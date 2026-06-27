from torch import nn
import torch
import triton
import triton.language as tl

from .autotune_config4linear import _small_m_autotune_configs, _gemm_autotune_configs

_SMALL_M_THRESHOLD = 128
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None) # 工程规范写法，为了把这个参数注册到_parameters字典里，即使是个None，以便后续代码如model.state_dict()、model.parameters()等能统一规范地处理这个参数

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.in_features, f"continuous batching need to flaten the input to 2D, and the last dimension should be {self.in_features}, but got {x.shape}"
        return myLinearInterface(x, self.weight, self.bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_param: torch.Tensor):
        param.data.copy_(loaded_param)
    
class MergedLinear(Linear):
    def __init__(self, in_features, out_features_list: list[int], bias=False):
        self.out_features_list = out_features_list
        self.merged_dim = 0 # 默认(out, in)维度的，在out维度上合并
        super().__init__(in_features, sum(out_features_list), bias)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_param: torch.Tensor, shard_id):
        param_data = param.data

        shard_start = sum(self.out_features_list[:shard_id])
        shard_size = self.out_features_list[shard_id]
        param_data = param_data.narrow(self.merged_dim, shard_start, shard_size) # 取出这个shard对应的那一块参数
        param_data.copy_(loaded_param) 

class QKVMergedLinear(Linear):
    def __init__(self, in_features, out_features_list: list[int], bias=False):
        assert len(out_features_list) == 3
        self.out_features_list = out_features_list
        self.merged_dim = 0
        super().__init__(in_features, sum(out_features_list), bias)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id):
        param_data = param.data
        assert shard_id in ['q', 'k', 'v']

        if shard_id == 'q':
            shard_size = self.out_features_list[0]
            shard_offset = 0
        elif shard_id == 'k':
            shard_size = self.out_features_list[1]
            shard_offset = self.out_features_list[0]
        else:
            shard_size = self.out_features_list[2]
            shard_offset = self.out_features_list[0] + self.out_features_list[1]

        param_data = param_data.narrow(self.merged_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

def _m_bucket(M: int) -> int:
    if M <= 1:
        return 1
    if M <= 16:
        return 16
    if M <= 64:
        return 64
    if M <= 128:
        return 128
    if M <= 512:
        return 512
    if M <= 2048:
        return 2048
    return 8192

# 语义跟F.linear相似，weight是(out, in)维度的，且部分序列长度，根据x的shape自动选择small_m_kernel还是gemm_kernel
def myLinearInterface(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor | None = None,
):
    """
    x:
        [M, K]
    weight:
        [N, K]
    bias:
        [N] or None
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")

    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")

    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("this Triton linear only supports CUDA tensors")

    if bias is not None and not bias.is_cuda:
        raise RuntimeError("bias must be CUDA tensor")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported x dtype: {x.dtype}")

    if weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported weight dtype: {weight.dtype}")

    if bias is not None and bias.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported bias dtype: {bias.dtype}")

    if x.device != weight.device:
        raise RuntimeError(f"x and weight must be on same device, got {x.device} and {weight.device}")

    if bias is not None and bias.device != x.device:
        raise RuntimeError(f"x and bias must be on same device, got {x.device} and {bias.device}")

    # 推理框架里不要偷偷 contiguous()
    # if not x.is_contiguous():
    #     raise RuntimeError("x must be contiguous. Do x.contiguous() explicitly before calling linear.") 
    assert x.stride(1) == 1 # x可以不连续，但必须保证最后一个维度的stride是1

    if not weight.is_contiguous():
        raise RuntimeError("weight must be contiguous. Use Linear.pack_weight() for packed inference.")
    
    M, K = x.shape
    N, K_w = weight.shape
    if K != K_w:
        raise ValueError(f"input feature dimension of x and weight must match, got {K} and {K_w}")
    stride_wn, stride_wk = weight.stride()

    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != N:
            raise ValueError(f"bias must be shape [{N}], got {tuple(bias.shape)}")
        if not bias.is_contiguous():
            raise RuntimeError("bias must be contiguous")
        
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    bias_for_kernel = bias if has_bias else out  # dummy pointer; HAS_BIAS=False 时不会读取

    m_bucket = _m_bucket(M)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    if M <= _SMALL_M_THRESHOLD:
        linear_small_m_kernel[grid](
            x,
            weight,
            bias_for_kernel,
            out,
            M,
            N,
            K,
            m_bucket,
            x.stride(0),
            x.stride(1),
            stride_wk,
            stride_wn,
            out.stride(0),
            out.stride(1),
            HAS_BIAS=has_bias,
        )
    else:
        linear_gemm_kernel[grid](
            x,
            weight,
            bias_for_kernel,
            out,
            M,
            N,
            K,
            m_bucket,
            x.stride(0),
            x.stride(1),
            stride_wk,
            stride_wn,
            out.stride(0),
            out.stride(1),
            HAS_BIAS=has_bias,
        )

    return out

@triton.autotune(
    configs=_small_m_autotune_configs(),
    key=["M_BUCKET", "N", "K"],
    cache_results=True,
)
@triton.jit
def linear_small_m_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    M_BUCKET,  # only used by autotune key
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # x: logical [M, K]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # weight is logical B: [K, N]
    w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_iter * BLOCK_K

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(x_ptrs, mask=x_mask, other=0.0)
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # LEARN: tl.dot(a, b, acc) 要求M >= 16, N >= 16 and K >= 16
        acc = tl.dot(a, b, acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # acc 是 fp32，store 到 fp16/bf16/fp32 out 时会按 out_ptr dtype 转换。
    tl.store(out_ptrs, acc, mask=out_mask)

@triton.autotune(
    configs=_gemm_autotune_configs(),
    key=["M_BUCKET", "N", "K"],
    cache_results=True,
)
@triton.jit
def linear_gemm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    M_BUCKET,  # only used by autotune key
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped swizzle, same as Triton matmul tutorial style
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_iter * BLOCK_K

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(x_ptrs, mask=x_mask, other=0.0)
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(a, b, acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, acc, mask=out_mask)
