import torch
import triton
import triton.language as tl
from torch import nn
from functools import lru_cache

@triton.jit
def rope_kernel(
    # Data Pointers
    in_ptr, out_ptr,
    pos_ptr, cos_sin_ptr,
    # Strides
    in_stride_tok, in_stride_head, in_stride_dim,
    out_stride_tok, out_stride_head, out_stride_dim,
    pos_stride_tok, cache_stride_pos,
    # Dimensions & Meta-parameters
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (num_tokens, num_heads)
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    # 1. 获取当前 Token 的位置 ID
    pos = tl.load(pos_ptr + pid_tok * pos_stride_tok)

    # 2. 设置线程块偏移量 (计算 head_dim 的前半部分)
    half_dim = head_dim // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    # 3. 从 cache 中直接加载当前位置的 cos 和 sin
    # cos_sin_cache 的 shape 是 [max_pos, 1, head_dim]
    cache_offset = pos * cache_stride_pos
    cos = tl.load(cos_sin_ptr + cache_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(cos_sin_ptr + cache_offset + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    # 4. 加载输入张量 x 的前半部分 (x1) 和后半部分 (x2)
    in_offset = pid_tok * in_stride_tok + pid_head * in_stride_head
    in1_ptrs = in_ptr + in_offset + offsets * in_stride_dim
    in2_ptrs = in_ptr + in_offset + (half_dim + offsets) * in_stride_dim

    # 转为 float32 防止精度溢出，对齐 PyTorch 中的 .float() 行为
    x1 = tl.load(in1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 5. 计算旋转 (RoPE 数学公式)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    # 6. 将结果写回内存，并转换回原本的数据类型 (如 float16/bfloat16)
    out_offset = pid_tok * out_stride_tok + pid_head * out_stride_head
    out1_ptrs = out_ptr + out_offset + offsets * out_stride_dim
    out2_ptrs = out_ptr + out_offset + (half_dim + offsets) * out_stride_dim
    
    tl.store(out1_ptrs, y1.to(in_ptr.dtype.element_ty), mask=mask)
    tl.store(out2_ptrs, y2.to(in_ptr.dtype.element_ty), mask=mask)


def apply_rotary_emb_triton(
    x: torch.Tensor, 
    positions: torch.Tensor, 
    cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """
    Triton 版本的 RoPE 算子。
    x: shape [num_tokens, num_heads, head_dim]
    positions: shape [seq_len]
    cos_sin_cache: shape [max_pos, 1, head_dim]
    """
    num_tokens = x.shape[0]
    num_heads = x.shape[1]
    head_dim = x.shape[2]

    # 分配输出内存 (Triton 也可以做 In-place 操作省内存，这里保持 Out-of-place 语义)
    out = torch.empty_like(x)

    # BLOCK_SIZE 取大于 head_dim // 2 的最小 2 的幂次方
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    # 并行网格: 为每个 Token 的每个 Head 启动一个 Block
    grid = (num_tokens, num_heads)

    rope_kernel[grid](
        x, out,
        positions, cos_sin_cache,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        positions.stride(0), cos_sin_cache.stride(0),
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # 预计算频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # 外积操作，得到一个 [max_position_embeddings, rotary_dim//2] 的矩阵
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Q、K的计算没融合在一起，主要因为GQA的head数目不同，kernel要特殊处理索引，目前先不这么麻烦
        query_out = apply_rotary_emb_triton(query, positions, self.cos_sin_cache)
        key_out = apply_rotary_emb_triton(key, positions, self.cos_sin_cache)
        
        return query_out, key_out

@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)