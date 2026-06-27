import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    return triton.cdiv(a, b)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def _bucket_seqlen_k(x: int) -> int:
    if x <= 128:
        return 128
    if x <= 256:
        return 256
    if x <= 512:
        return 512
    if x <= 768:
        return 768
    if x <= 1024:
        return 1024
    if x <= 1536:
        return 1536
    if x <= 2048:
        return 2048
    return _next_power_of_2(x)

# RTX 3090 / GA102(sm86) tuned static heuristic.
# 注意：这个 kernel 会把 head_dim round up 到 BLOCK_D。
# 所以 select 时应该按 BLOCK_D 选，而不是只按原始 head_dim 选。
def _select_prefill_config(
    head_dim: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    paged: bool,
) -> tuple[int, int]:
    block_d = _next_power_of_2(head_dim)

    # paged KV 有 block_table 间接寻址，K/V 空间局部性弱于 compact。
    # RTX 3090 上 paged 路径建议保守使用 BLOCK_N=64。
    if paged:
        block_n = 64

        if block_d <= 64:
            block_m = 32 if max_seqlen_q <= 256 else 64
        elif block_d <= 128:
            # 覆盖 head_dim = 80/96/128
            block_m = 16 if max_seqlen_q <= 128 else 32
        else:
            # D=256 级别寄存器压力很高
            block_m = 16 if max_seqlen_q <= 512 else 32

        return block_m, block_n

    # compact KV 是连续内存，D<=64 时可以用更宽的 K/V tile 提高复用。
    if block_d <= 64:
        block_m = 32 if max_seqlen_q <= 256 else 64
        block_n = 64 if max_seqlen_k <= 128 else 128
    elif block_d <= 128:
        # 覆盖 head_dim = 80/96/128。
        # 不建议再用 D<=64 的 64x128 大 tile。
        block_m = 16 if max_seqlen_q <= 128 else 32
        block_n = 64
    else:
        block_m = 16 if max_seqlen_q <= 512 else 32
        block_n = 64

    return block_m, block_n


def _select_num_warps(head_dim: int, block_m: int, block_n: int) -> int:
    block_d = _next_power_of_2(head_dim)

    # GA102 上 FA forward tile 到 D=128 通常 4 warps 更稳；
    # D=256 级别或显式很大的 D=128 tile 再用 8 warps。
    if block_d <= 64:
        return 4
    if block_d <= 128:
        return 8 if block_m >= 64 else 4
    return 8


def _select_num_stages(head_dim: int, block_n: int, paged: bool) -> int:
    block_d = _next_power_of_2(head_dim)

    # paged 路径有间接访存，过深 pipeline 不一定能稳定收益，
    # 还会增加 shared/register 压力。
    if paged:
        return 3

    # compact + D<=64 + BLOCK_N=128 是最容易受 K/V load latency 影响的组合，
    # RTX 3090 上 4 stages 通常更合适。
    if block_d <= 64 and block_n >= 128:
        return 4

    return 3

@triton.jit
def _flash_attn_varlen_compact_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    softmax_scale,
    max_seqlen_k_bucket: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + pid_b)
    q_end = tl.load(cu_seqlens_q_ptr + pid_b + 1)
    k_start = tl.load(cu_seqlens_k_ptr + pid_b)
    k_end = tl.load(cu_seqlens_k_ptr + pid_b + 1)

    q_len = q_end - q_start
    k_len = k_end - k_start

    # 因都对齐到max_seqlen_q，对于空白tiling，直接early return
    if pid_m * BLOCK_M >= q_len:
        return

    prefix_len = k_len - q_len
    kv_head = pid_h // group_size # GQA

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + offs_d[None, :],
        mask=q_mask,
        other=0.0,
    )

    valid_m = offs_m < q_len
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634 # log2 scale

    for block_start in range(0, max_seqlen_k_bucket, BLOCK_N):
        if block_start < k_len:
            k_pos = block_start + offs_n
            k_mask = (k_pos[:, None] < k_len) & (offs_d[None, :] < head_dim)
            k = tl.load(
                k_ptr + (k_start + k_pos[:, None]) * stride_k_t + kv_head * stride_k_h + offs_d[None, :],
                mask=k_mask,
                other=0.0,
            )

            qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * qk_scale
            valid = k_pos[None, :] < k_len
            if CAUSAL:
                q_abs_pos = prefix_len + offs_m
                valid = valid & (k_pos[None, :] <= q_abs_pos[:, None])
            valid = valid & valid_m[:, None]
            qk = tl.where(valid, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp2(qk - m_new_safe[:, None])
            alpha = tl.exp2(m_i - m_new_safe)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_ptr + (k_start + k_pos[:, None]) * stride_v_t + kv_head * stride_v_h + offs_d[None, :],
                mask=k_mask,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))
            m_i = m_new
            l_i = l_new

    out = acc / tl.maximum(l_i[:, None], 1.0e-20)
    o_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    tl.store(
        o_ptr + (q_start + offs_m[:, None]) * stride_o_t + pid_h * stride_o_h + offs_d[None, :],
        out,
        mask=o_mask,
    )


@triton.jit
def _flash_attn_varlen_paged_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    block_table_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_s: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_bt_b: tl.constexpr,
    softmax_scale,
    # LEARN：了解到一个知识，triton会编译时会把constexpr参数当作编译时常量，直接展开到代码里，当这些常量去不同值时，triton会重新编译kernel，会有编译开销，所以这些参数尽量不要频繁变化，尤其是BLOCK_M/N/D这种kernel tile size参数。容易变化的就像下面这个参数bucket化。
    # LEARN：constexpr编译和autotune的区别在于，autotune只选择性能最优的配置参数，而constexpr才是真的会编译成对应值的kernel机器码。
    max_seqlen_k_bucket: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + pid_b)
    q_end = tl.load(cu_seqlens_q_ptr + pid_b + 1)
    k_start = tl.load(cu_seqlens_k_ptr + pid_b)
    k_end = tl.load(cu_seqlens_k_ptr + pid_b + 1)

    q_len = q_end - q_start
    k_len = k_end - k_start
    
    # 因都对齐到max_seqlen_q，对于空白tiling，直接early return
    if pid_m * BLOCK_M >= q_len:
        return
    
    prefix_len = k_len - q_len
    kv_head = pid_h // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + offs_d[None, :],
        mask=q_mask,
        other=0.0,
    )

    valid_m = offs_m < q_len
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634 # log2 scale

    for block_start in range(0, max_seqlen_k_bucket, BLOCK_N):
        if block_start < k_len:
            k_pos = block_start + offs_n
            logical_block = k_pos // cache_block_size
            block_offset = k_pos - logical_block * cache_block_size
            physical_block = tl.load(
                block_table_ptr + pid_b * stride_bt_b + logical_block,
                mask=k_pos < k_len,
                other=0,
            )

            k_mask = (k_pos[:, None] < k_len) & (offs_d[None, :] < head_dim)
            k = tl.load(
                k_cache_ptr
                + physical_block[:, None] * stride_k_b
                + block_offset[:, None] * stride_k_s
                + kv_head * stride_k_h
                + offs_d[None, :],
                mask=k_mask,
                other=0.0,
            )

            qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * qk_scale
            valid = k_pos[None, :] < k_len
            if CAUSAL:
                q_abs_pos = prefix_len + offs_m
                valid = valid & (k_pos[None, :] <= q_abs_pos[:, None])
            valid = valid & valid_m[:, None]
            qk = tl.where(valid, qk, -float("inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp2(qk - m_new_safe[:, None])
            alpha = tl.exp2(m_i - m_new_safe)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_cache_ptr
                + physical_block[:, None] * stride_v_b
                + block_offset[:, None] * stride_v_s
                + kv_head * stride_v_h
                + offs_d[None, :],
                mask=k_mask,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))
            m_i = m_new
            l_i = l_new

    out = acc / tl.maximum(l_i[:, None], 1.0e-20)
    o_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    tl.store(
        o_ptr + (q_start + offs_m[:, None]) * stride_o_t + pid_h * stride_o_h + offs_d[None, :],
        out,
        mask=o_mask,
    )


def flash_attn_varlen_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    assert q.shape[-2] % k.shape[-2] == 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1

    total_q, num_q_heads, head_dim = q.shape
    
    out = torch.empty(
        (total_q, num_q_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    if q.numel() == 0:
        return out

    num_kv_heads = k.shape[1]
    batch_size = cu_seqlens_q.numel() - 1
    group_size = num_q_heads // num_kv_heads
    block_m, block_n = _select_prefill_config(head_dim, max_seqlen_q, max_seqlen_k, paged=False)
    block_d = _next_power_of_2(head_dim)
    max_seqlen_k_bucket = _bucket_seqlen_k(max_seqlen_k)

    num_warps = _select_num_warps(head_dim, block_m, block_n)
    num_stages = _select_num_stages(head_dim, block_n, paged=False)

    grid = (_ceil_div(max_seqlen_q, block_m), batch_size, num_q_heads)
    _flash_attn_varlen_compact_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        softmax_scale,
        max_seqlen_k_bucket,
        group_size,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def flash_attn_varlen_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor,
) -> torch.Tensor:
    assert q.ndim == 3 and k_cache.ndim == 4 and v_cache.ndim == 4
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]
    assert k_cache.shape[-2] == v_cache.shape[-2]
    assert q.shape[-2] % k_cache.shape[-2] == 0
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1

    total_q, num_q_heads, head_dim = q.shape
    
    out = torch.empty(
        (total_q, num_q_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    if q.numel() == 0:
        return out

    num_kv_heads = k_cache.shape[2]
    cache_block_size = k_cache.shape[1]
    batch_size = cu_seqlens_q.numel() - 1
    group_size = num_q_heads // num_kv_heads

    # LEARN: 本来这个BLOCK_N、BLOCK_M之类的kernel的超参数都需要autone来找到最优值，但之前觉得会跑很久，就采用heuristic select的方式来选择了。但其实真运行起来linear部分的autotune并没占用很久时间，而且还能保存上一次调出来的结果。后续可以改成完全autotune，更加稳健。

    block_m, block_n = _select_prefill_config(head_dim, max_seqlen_q, max_seqlen_k, paged=True)
    block_d = _next_power_of_2(head_dim)
    max_seqlen_k_bucket = _bucket_seqlen_k(max_seqlen_k)

    num_warps = _select_num_warps(head_dim, block_m, block_n)
    num_stages = _select_num_stages(head_dim, block_n, paged=True)

    grid = (_ceil_div(max_seqlen_q, block_m), batch_size, num_q_heads)
    _flash_attn_varlen_paged_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        block_table.stride(0),
        softmax_scale,
        max_seqlen_k_bucket,
        group_size,
        head_dim,
        cache_block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
