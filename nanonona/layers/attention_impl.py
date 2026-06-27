import torch
import triton
import triton.language as tl

try:
    from .attention_decode import flash_attn_decode
    from .attention_prefill import flash_attn_varlen_compact, flash_attn_varlen_paged
except ImportError:
    from attention_decode import flash_attn_decode
    from attention_prefill import flash_attn_varlen_compact, flash_attn_varlen_paged


@triton.jit
def store_kvcache_kernel(
    src_k_ptr, src_v_ptr, dst_k_cache_ptr, dst_v_cache_ptr, slots_ptr,
    stride_src_k_0, stride_src_v_0,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.load(slots_ptr + pid)

    offsets = tl.arange(0, BLOCK_D)
    valid = (slot >= 0) & (offsets < D)
    src_ks_prt = src_k_ptr + pid * stride_src_k_0 + offsets
    src_vs_prt = src_v_ptr + pid * stride_src_v_0 + offsets

    src_ks = tl.load(src_ks_prt, mask=valid, other=0.0)
    src_vs = tl.load(src_vs_prt, mask=valid, other=0.0)

    cache_offsets = slot * D + offsets
    tl.store(dst_k_cache_ptr + cache_offsets, src_ks, mask=valid)
    tl.store(dst_v_cache_ptr + cache_offsets, src_vs, mask=valid)


def store_kvcache(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k_cache: torch.Tensor,
    dst_v_cache: torch.Tensor,
    slots: torch.Tensor,
):
    N, num_head, head_dim = src_k.shape
    D = num_head * head_dim
    assert src_k.stride(-1) == 1 and src_v.stride(-1) == 1
    assert src_k.stride(1) == head_dim and src_v.stride(1) == head_dim
    assert dst_k_cache.stride(1) == D and dst_v_cache.stride(1) == D
    assert slots.numel() == N
    block_d = 1 << (D - 1).bit_length()
    store_kvcache_kernel[(N,)](
        src_k, src_v, dst_k_cache, dst_v_cache, slots, 
        src_k.stride(0), src_v.stride(0), D, BLOCK_D=block_d
    )



# 变长Flattention，用于prefill
def flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q, cu_seqlens_q,
    max_seqlen_k, cu_seqlens_k,
    softmax_scale, causal,
    block_table
):
    if block_table is None: # 没有prefix cache，就使用最原始的flashAttn
        return flash_attn_varlen_compact(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    # 否则，使用paged Attn，主要是要利用block_table读取cached key/value
    return flash_attn_varlen_paged( 
        q=q,
        k_cache=k,
        v_cache=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
    )



# decode
def flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens, block_table, 
    softmax_scale, causal
):
    return flash_attn_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )
