import triton

def _small_m_autotune_configs():
    """
    针对 M <= 128 的 decode / continuous batching 小 M 场景。

    设计目标：
    - 降低单 CTA 的资源占用，减少小 M 时的浪费
    - BLOCK_M 不要太大，否则 M=1/16/64 时空算严重
    - BLOCK_N 以 64/128 为主，提高 N 维吞吐
    - BLOCK_K 以 32/64 为主，适配 Ampere tensor core
    """
    return [
        # ---- M 极小：M=1/16 常见 decode 场景 ----
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=3,
        ),

        # ---- M=32/64：仍然偏 latency，但可以稍微拉大 tile ----
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=4,
            num_stages=3,
        ),

        # ---- M=64/128：小 M 上界，开始接近普通 GEMM ----
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),

        # ---- K 很大时可能更好，但资源占用也更高 ----
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=4,
        ),
    ]


def _gemm_autotune_configs():
    """
    针对 M > 128 的普通 GEMM 场景。

    RTX 3090 上通常：
    - 128x64x32 / 64x128x32 是比较稳的基础配置
    - 128x128x32 吞吐强，但寄存器/共享内存压力更大
    - BLOCK_K=64 对大 K 可能更好，但不一定总赢
    - num_warps 4/8 都要测，3090 上大 tile 常见 4 或 8 warps 最优
    """
    return [
        # ---- 通用稳妥配置：优先级最高 ----
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),

        # ---- 大 N / 大 M 吞吐配置 ----
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),

        # ---- K 较大时尝试 BLOCK_K=64 ----
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),

        # ---- 更激进 pipeline，K 很大时可能赢 ----
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=4,
        ),

        # ---- M 很大时增加 M tile，提高复用，但只适合足够大的 M/N/K ----
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),
    ]