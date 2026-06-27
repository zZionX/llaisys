import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    return triton.cdiv(a, b)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def _bucket_context_len(x: int) -> int:
    if x <= 128:
        return 128
    if x <= 256:
        return 256
    if x <= 512:
        return 512
    if x <= 1024:
        return 1024
    if x <= 2048:
        return 2048
    if x <= 4096:
        return 4096
    if x <= 8192:
        return 8192
    if x <= 16384:
        return 16384
    return triton.next_power_of_2(x)

RTX3090_NUM_SMS = 82
MMA_MIN_DIM = 16


def _round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _pad_mma_dim(x: int) -> int:
    return max(MMA_MIN_DIM, _next_power_of_2(x))


def _select_q_heads_per_block(num_q_heads: int, num_kv_heads: int, head_dim: int) -> int:
    """
    RTX 3090 / Ampere heuristic.

    约束：
    1. Q_HEADS_PER_BLOCK 不能跨 kv head，因为 kernel 里 kv_head 是单个标量：
       kv_head = (pid_hb * Q_HEADS_PER_BLOCK) // group_size
    2. 因此只能选 group_size 的因子。
    3. head_dim 越大，acc/qk/p 的寄存器压力越大，需要降低 Q heads 聚合数。
    """
    group_size = num_q_heads // num_kv_heads
    if group_size <= 1:
        return 1

    # D<=64: 复用 K/V 的收益很高，8 heads 一般能接受
    # 65<=D<=128: 4 heads 是 GQA/MQA 下的甜点
    # D>128: 寄存器压力明显增加，保守压到 2
    if head_dim <= 64:
        max_q_heads = 8
    elif head_dim <= 128:
        max_q_heads = 4
    else:
        max_q_heads = 2

    for c in (8, 4, 2, 1):
        if c <= max_q_heads and c <= group_size and group_size % c == 0 and num_q_heads % c == 0:
            return c
    return 1


def _select_block_n(head_dim: int, max_context_len: int) -> int:
    """
    BLOCK_N 控制 qk/p 的 tile N 维。
    大 BLOCK_N：循环次数少，K/V load amortization 好；
    小 BLOCK_N：qk/p 临时矩阵小，寄存器压力低，占用率更好。
    """
    if head_dim <= 64:
        # D=64 时 128 通常更好；长上下文下为了 occupancy 和寄存器压力降到 64
        return 128 if max_context_len <= 4096 else 64

    if head_dim < 128:
        # D=80/96 会 bucket 到 BLOCK_D=128，但真实 D 小于 128，短上下文可用 128
        return 128 if max_context_len <= 2048 else 64

    if head_dim == 128:
        # LLaMA/Qwen 类常见 D=128，RTX 3090 上 64 更稳，避免 QH=4 时 qk/p/acc 过大
        return 64

    # D=160/192/256：长上下文建议 32，避免 spills
    return 64 if max_context_len <= 2048 else 32


def _select_split_n(
    max_context_len: int,
    batch_size: int,
    num_head_blocks: int,
    block_n: int,
    sm_count: int = RTX3090_NUM_SMS,
) -> int:
    """
    Split-K 的关键不是固定 SPLIT_N，而是让 stage1 的 program 数够填满 RTX 3090。

    stage1 programs = batch_size * num_head_blocks * num_splits

    对 RTX 3090，目标先取 2 wave: 2 * 82 = 164 个 programs。
    同时限制每个 split 不要太短，否则 stage2 combine 和 partial 写回开销会上来。
    """
    base_work = max(1, batch_size * num_head_blocks)
    target_work = 2 * sm_count

    desired_splits = max(1, _ceil_div(target_work, base_work))

    # 每个 split 至少覆盖若干个 BLOCK_N，避免 split 太碎。
    # <=8K 时为了小 batch latency，可以允许 2 个 BLOCK_N 一个 split。
    min_blocks_per_split = 2 if max_context_len <= 8192 else 4
    min_split_n = max(256, min_blocks_per_split * block_n)

    # 限制 num_splits，避免 partial_acc/partial_m/partial_l 和 stage2 成本膨胀。
    max_splits_by_len = max(1, max_context_len // min_split_n)
    max_splits = min(32, max_splits_by_len)

    desired_splits = min(desired_splits, max_splits)

    # 用少量 split 桶降低 Triton JIT 变体数量。
    split_choices = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    num_splits = split_choices[-1]
    for s in split_choices:
        if s >= desired_splits:
            num_splits = s
            break
    num_splits = min(num_splits, max_splits)

    split_n = _ceil_div(max_context_len, num_splits)
    split_n = _round_up_to_multiple(split_n, block_n)
    return split_n


def _use_split_k(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_len: int,
) -> bool:
    if max_context_len < 4096:
        return False

    q_heads_per_block = _select_q_heads_per_block(num_q_heads, num_kv_heads, head_dim)
    num_head_blocks = _ceil_div(num_q_heads, q_heads_per_block)

    # 3090: stage1 program 数不足 2 wave 时，split-k 通常更有价值
    base_work = batch_size * num_head_blocks
    return base_work < 2 * RTX3090_NUM_SMS


@triton.jit
def _decode_single_pass_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    stride_q_b: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_s: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_bt_b: tl.constexpr,
    softmax_scale,
    max_context_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    Q_HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hb = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    qh_offsets = pid_hb * Q_HEADS_PER_BLOCK + offs_m
    kv_head = (pid_hb * Q_HEADS_PER_BLOCK) // group_size
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    cache_len = tl.load(cache_seqlens_ptr + pid_b)
    valid_h = (offs_m < Q_HEADS_PER_BLOCK) & (qh_offsets < num_q_heads)
    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + 0 * stride_q_s
        + qh_offsets[:, None] * stride_q_h
        + offs_d[None, :],
        mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634 # log2 scale
    for block_start in range(0, max_context_len, BLOCK_N):
        if block_start < cache_len:
            k_pos = block_start + offs_n
            logical_block = k_pos // cache_block_size
            block_offset = k_pos - logical_block * cache_block_size
            physical_block = tl.load(
                block_table_ptr + pid_b * stride_bt_b + logical_block,
                mask=k_pos < cache_len,
                other=0,
            )

            k_mask = (k_pos[:, None] < cache_len) & (offs_d[None, :] < head_dim)
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
            qk = tl.where((k_pos[None, :] < cache_len) & valid_h[:, None], qk, -float("inf"))

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
    tl.store(
        o_ptr
        + pid_b * stride_o_b
        + 0 * stride_o_s
        + qh_offsets[:, None] * stride_o_h
        + offs_d[None, :],
        out,
        mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
    )


@triton.jit
def _decode_split_k_stage1_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    stride_q_b: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_s: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_pa_b: tl.constexpr,
    stride_pa_hb: tl.constexpr,
    stride_pa_split: tl.constexpr,
    stride_pa_qh: tl.constexpr,
    stride_pm_b: tl.constexpr,
    stride_pm_hb: tl.constexpr,
    stride_pm_split: tl.constexpr,
    stride_pm_qh: tl.constexpr,
    stride_bt_b: tl.constexpr,
    softmax_scale,
    num_q_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    Q_HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_split = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    qh_offsets = pid_hb * Q_HEADS_PER_BLOCK + offs_m
    kv_head = (pid_hb * Q_HEADS_PER_BLOCK) // group_size
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    split_start = pid_split * SPLIT_N

    cache_len = tl.load(cache_seqlens_ptr + pid_b)
    valid_h = (offs_m < Q_HEADS_PER_BLOCK) & (qh_offsets < num_q_heads)

    if split_start >= cache_len:
        zero_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        zero_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)
        neg_inf_vec = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

        tl.store(
            partial_acc_ptr
            + pid_b * stride_pa_b
            + pid_hb * stride_pa_hb
            + pid_split * stride_pa_split
            + offs_m[:, None] * stride_pa_qh
            + offs_d[None, :],
            zero_acc,
            mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
        )

        tl.store(
            partial_m_ptr
            + pid_b * stride_pm_b
            + pid_hb * stride_pm_hb
            + pid_split * stride_pm_split
            + offs_m * stride_pm_qh,
            neg_inf_vec,
            mask=valid_h,
        )

        tl.store(
            partial_l_ptr
            + pid_b * stride_pm_b
            + pid_hb * stride_pm_hb
            + pid_split * stride_pm_split
            + offs_m * stride_pm_qh,
            zero_vec,
            mask=valid_h,
        )
        return

    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + 0 * stride_q_s
        + qh_offsets[:, None] * stride_q_h
        + offs_d[None, :],
        mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634 # log2 scale
    for local_start in range(0, SPLIT_N, BLOCK_N):
        if split_start + local_start < cache_len:
            k_pos = split_start + local_start + offs_n
            logical_block = k_pos // cache_block_size
            block_offset = k_pos - logical_block * cache_block_size
            physical_block = tl.load(
                block_table_ptr + pid_b * stride_bt_b + logical_block,
                mask=k_pos < cache_len,
                other=0,
            )

            k_mask = (k_pos[:, None] < cache_len) & (offs_d[None, :] < head_dim)
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
            qk = tl.where((k_pos[None, :] < cache_len) & valid_h[:, None], qk, -float("inf"))

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

    tl.store(
        partial_acc_ptr
        + pid_b * stride_pa_b
        + pid_hb * stride_pa_hb
        + pid_split * stride_pa_split
        + offs_m[:, None] * stride_pa_qh
        + offs_d[None, :],
        acc,
        mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
    )
    tl.store(
        partial_m_ptr
        + pid_b * stride_pm_b
        + pid_hb * stride_pm_hb
        + pid_split * stride_pm_split
        + offs_m * stride_pm_qh,
        m_i,
        mask=valid_h,
    )
    tl.store(
        partial_l_ptr
        + pid_b * stride_pm_b
        + pid_hb * stride_pm_hb
        + pid_split * stride_pm_split
        + offs_m * stride_pm_qh,
        l_i,
        mask=valid_h,
    )


@triton.jit
def _decode_split_k_stage2_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    o_ptr,
    stride_pa_b: tl.constexpr,
    stride_pa_hb: tl.constexpr,
    stride_pa_split: tl.constexpr,
    stride_pa_qh: tl.constexpr,
    stride_pm_b: tl.constexpr,
    stride_pm_hb: tl.constexpr,
    stride_pm_split: tl.constexpr,
    stride_pm_qh: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_splits: tl.constexpr,
    head_dim: tl.constexpr,
    Q_HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hb = tl.program_id(1)

    qh_offsets = pid_hb * Q_HEADS_PER_BLOCK + tl.arange(0, Q_HEADS_PER_BLOCK)
    offs_d = tl.arange(0, BLOCK_D)
    valid_h = qh_offsets < num_q_heads

    m_i = tl.full((Q_HEADS_PER_BLOCK,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_HEADS_PER_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((Q_HEADS_PER_BLOCK, BLOCK_D), dtype=tl.float32)

    for split_idx in range(0, num_splits):
        m_part = tl.load(
            partial_m_ptr
            + pid_b * stride_pm_b
            + pid_hb * stride_pm_hb
            + split_idx * stride_pm_split
            + tl.arange(0, Q_HEADS_PER_BLOCK) * stride_pm_qh,
            mask=valid_h,
            other=-float("inf"),
        )
        l_part = tl.load(
            partial_l_ptr
            + pid_b * stride_pm_b
            + pid_hb * stride_pm_hb
            + split_idx * stride_pm_split
            + tl.arange(0, Q_HEADS_PER_BLOCK) * stride_pm_qh,
            mask=valid_h,
            other=0.0,
        )
        acc_part = tl.load(
            partial_acc_ptr
            + pid_b * stride_pa_b
            + pid_hb * stride_pa_hb
            + split_idx * stride_pa_split
            + tl.arange(0, Q_HEADS_PER_BLOCK)[:, None] * stride_pa_qh
            + offs_d[None, :],
            mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        m_new = tl.maximum(m_i, m_part)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_new_safe)
        beta = tl.exp2(m_part - m_new_safe)
        l_new = l_i * alpha + l_part * beta
        acc = acc * alpha[:, None] + acc_part * beta[:, None]
        m_i = m_new
        l_i = l_new

    out = acc / tl.maximum(l_i[:, None], 1.0e-20)
    tl.store(
        o_ptr
        + pid_b * stride_o_b
        + 0 * stride_o_s
        + qh_offsets[:, None] * stride_o_h
        + offs_d[None, :],
        out,
        mask=valid_h[:, None] & (offs_d[None, :] < head_dim),
    )


def _run_decode_single_pass(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    max_context_len: int,
) -> torch.Tensor:
    batch_size, _, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads
    q_heads_per_block = _select_q_heads_per_block(num_q_heads, num_kv_heads, head_dim)
    block_n = _select_block_n(head_dim, max_context_len)
    block_m = _pad_mma_dim(q_heads_per_block)
    block_d = _pad_mma_dim(head_dim)
    cache_block_size = k_cache.shape[1]
    out = torch.empty(
        (batch_size, 1, num_q_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )

    grid = (batch_size, _ceil_div(num_q_heads, q_heads_per_block))
    _decode_single_pass_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        cache_seqlens,
        block_table,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        softmax_scale,
        max_context_len,
        num_q_heads,
        group_size,
        head_dim,
        cache_block_size,
        Q_HEADS_PER_BLOCK=q_heads_per_block,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=3,
    )
    return out


def _run_decode_split_k(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    max_context_len: int,
) -> torch.Tensor:
    batch_size, _, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads

    q_heads_per_block = _select_q_heads_per_block(num_q_heads, num_kv_heads, head_dim)
    num_head_blocks = _ceil_div(num_q_heads, q_heads_per_block)
    block_n = _select_block_n(head_dim, max_context_len)
    split_n = _select_split_n(
        max_context_len=max_context_len,
        batch_size=batch_size,
        num_head_blocks=num_head_blocks,
        block_n=block_n,
    )
    num_splits = _ceil_div(max_context_len, split_n)
    block_m = _pad_mma_dim(q_heads_per_block)
    block_d = _pad_mma_dim(head_dim)
    cache_block_size = k_cache.shape[1]

    partial_acc = torch.empty(
        (batch_size, num_head_blocks, num_splits, q_heads_per_block, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    partial_m = torch.empty(
        (batch_size, num_head_blocks, num_splits, q_heads_per_block),
        device=q.device,
        dtype=torch.float32,
    )
    partial_l = torch.empty_like(partial_m)
    out = torch.empty(
        (batch_size, 1, num_q_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )

    grid_stage1 = (batch_size, num_head_blocks, num_splits)
    _decode_split_k_stage1_kernel[grid_stage1](
        q,
        k_cache,
        v_cache,
        partial_acc,
        partial_m,
        partial_l,
        cache_seqlens,
        block_table,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        block_table.stride(0),
        softmax_scale,
        num_q_heads,
        group_size,
        head_dim,
        cache_block_size,
        Q_HEADS_PER_BLOCK=q_heads_per_block,
        BLOCK_M=block_m,
        SPLIT_N=split_n,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=3,
    )

    grid_stage2 = (batch_size, num_head_blocks)
    _decode_split_k_stage2_kernel[grid_stage2](
        partial_acc,
        partial_m,
        partial_l,
        out,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_q_heads,
        num_splits,
        head_dim,
        Q_HEADS_PER_BLOCK=q_heads_per_block,
        BLOCK_D=block_d,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=3,
    )
    return out


def flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del causal

    original_ndim = q.ndim
    if q.ndim == 3:
        q = q.unsqueeze(1)

    assert q.ndim == 4 and q.shape[1] == 1
    assert k_cache.ndim == 4 and v_cache.ndim == 4
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]
    assert k_cache.shape[-2] == v_cache.shape[-2]
    assert q.shape[-2] % k_cache.shape[-2] == 0
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert block_table is not None

    max_context_len = int(cache_seqlens.max().item())
    max_context_len_bucket = _bucket_context_len(max_context_len)
    if max_context_len == 0:
        out = torch.zeros_like(q)
    elif _use_split_k(
        q.shape[0],
        q.shape[2],
        k_cache.shape[2],
        q.shape[3],
        max_context_len,
    ):
        out = _run_decode_split_k(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, max_context_len_bucket)
    else:
        out = _run_decode_single_pass(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, max_context_len_bucket)

    if original_ndim == 3:
        return out.squeeze(1)
    return out
