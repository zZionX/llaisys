from torch import nn
import torch
import triton 
import triton.language as tl

class RMSnorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__() # 补上 super 初始化
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        num_tokens, hidden_size = x.shape
        grid = (num_tokens, )
        BLOCK_SIZE = triton.next_power_of_2(hidden_size) # 一行一个kernel
        BLOCK_SIZE = max(BLOCK_SIZE, 128)

        # 分别启动kernel
        if residual is not None:
            fused_add_rmsnorm_kernel[grid](
                x, residual, self.weight, 
                x.stride(0), residual.stride(0), # LEARN：kernel中为了避免不连续tensor，需要传入步长参数来正确加载数据！！！
                self.eps, hidden_size, BLOCK_SIZE
            )
            return x, residual
        else:
            out = torch.empty_like(x) # LEARN: 不确定这里创的这个tensor会不会影响cuda graph的优化？？？？先不考虑cuda gragh的实现
            rmsnorm_kernel[grid](
                x, out, self.weight, 
                x.stride(0), out.stride(0),
                self.eps, hidden_size, BLOCK_SIZE
            )
            return out
        
@triton.jit
def fused_add_rmsnorm_kernel(# LEARN: 算子融合,把RMSNorm之前的残差加法融合在一起,减少内存访问
    x_ptr, 
    residual_ptr, 
    weight_ptr, 
    x_stride_0,       # <--- 新增：x 的行步长
    res_stride_0,     # <--- 新增：residual 的行步长
    eps, 
    hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 核心修改：使用 stride_0 计算每一行的起始物理地址
    x_start_ptr = x_ptr + pid * x_stride_0
    res_start_ptr = residual_ptr + pid * res_stride_0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    x = tl.load(x_start_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(res_start_ptr + offsets, mask=mask, other=0.0)
    original_type = x.dtype # LEARN：这是个编译期的绝对常量，在kernel内获取类型反而更优
    
    x_res = x.to(tl.float32) + res.to(tl.float32)
    tl.store(res_start_ptr + offsets, x_res.to(original_type), mask=mask)

    x_sq = x_res * x_res
    mean_sq = tl.sum(x_sq, axis=0) / hidden_size
    rms_scale = tl.math.rsqrt(mean_sq + eps) # 推荐用 tl.math.rsqrt

    # 注意：weight 是 1D 的 Parameter，物理上绝对连续，所以不需要 stride，直接按 offset 读
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    out_f32 = x_res * rms_scale * w_f32

    out = out_f32.to(original_type)
    tl.store(x_start_ptr + offsets, out, mask=mask)


@triton.jit
def rmsnorm_kernel(
    x_ptr, 
    out_ptr,
    weight_ptr, 
    x_stride_0,       # <--- 新增：x 的行步长
    out_stride_0,     # <--- 新增：out 的行步长
    eps, 
    hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 核心修改：使用 stride_0 计算每一行的起始物理地址
    x_start_ptr = x_ptr + pid * x_stride_0
    out_start_ptr = out_ptr + pid * out_stride_0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(x_start_ptr + offsets, mask=mask, other=0.0)
    original_type = x.dtype # 这是个编译期的绝对常量，在kernel内获取类型反而更优

    # 在FP32下计算RMSNorm
    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    mean_sq = tl.sum(x_sq, axis=0) / hidden_size
    rms_scale = tl.math.rsqrt(mean_sq + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    out_f32 = x_f32 * rms_scale * w_f32

    out = out_f32.to(original_type)
    tl.store(out_start_ptr + offsets, out, mask=mask)
