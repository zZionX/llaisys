import torch
import triton
import triton.language as tl

def _swiglu_m_bucket(num_tokens: int) -> int:
    if num_tokens <= 1:
        return 1
    if num_tokens <= 4:
        return 4
    if num_tokens <= 8:
        return 8
    if num_tokens <= 16:
        return 16
    if num_tokens <= 32:
        return 32
    if num_tokens <= 64:
        return 64
    if num_tokens <= 128:
        return 128
    if num_tokens <= 256:
        return 256
    if num_tokens <= 512:
        return 512
    return 1024

def _swiglu_autotune_configs():
    return [
        # decode / very small M: 多切 hidden 维，增加 program 数量
        triton.Config(
            {"BLOCK_N": 256},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 512},
            num_warps=4,
        ),

        # 通用主力
        triton.Config(
            {"BLOCK_N": 1024},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 1024},
            num_warps=8,
        ),

        # prefill / large M: 减少 program 数量，提升单 program 吞吐
        triton.Config(
            {"BLOCK_N": 2048},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 2048},
            num_warps=8,
        ),

        # 可以试，但不一定总赢；RTX 3090 上有些 hidden_size 会选它
        triton.Config(
            {"BLOCK_N": 4096},
            num_warps=8,
        ),
    ]

def SwiGLU(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute:
        out = up * silu(gate)
            = up * gate * sigmoid(gate)

    gate: [num_tokens, hidden_size]
    up:   [num_tokens, hidden_size]
    out:  [num_tokens, hidden_size] or None

    默认 out=None 时，原地覆盖 gate。
    """

    assert gate.dim() == 2, f"gate must be a 2D tensor, got {tuple(gate.shape)}"
    assert up.dim() == 2, f"up must be a 2D tensor, got {tuple(up.shape)}"
    assert gate.shape == up.shape, (
        f"gate and up shape mismatch: gate={tuple(gate.shape)}, up={tuple(up.shape)}"
    )

    assert gate.is_cuda and up.is_cuda, "gate and up must be CUDA tensors"
    assert gate.device == up.device, "gate and up must be on the same device"
    assert gate.dtype == up.dtype, "gate and up must have the same dtype"

    if out is None:
        # 默认原地覆盖 gate，节省一个中间 buffer。
        out = gate
    else:
        assert out.dim() == 2, f"out must be a 2D tensor, got {tuple(out.shape)}"
        assert out.shape == gate.shape, (
            f"out shape mismatch: out={tuple(out.shape)}, gate={tuple(gate.shape)}"
        )
        assert out.is_cuda, "out must be CUDA tensor"
        assert out.device == gate.device, "out must be on the same device as gate"
        assert out.dtype == gate.dtype, "out dtype must match gate dtype"

        # 不建议 out=up。数学上 elementwise 可以做，但 autotune 首次调参时会多次运行，
        # 如果 out 覆盖 up，需要额外 restore up。这里先禁止，保持实现简单。
        if out.data_ptr() == up.data_ptr():
            raise RuntimeError(
                "out=up is not supported in this autotuned SwiGLU. "
                "Use out=None to overwrite gate, or pass a separate out tensor."
            )

    num_tokens, hidden_size = gate.shape

    if num_tokens == 0 or hidden_size == 0:
        return out

    m_bucket = _swiglu_m_bucket(num_tokens)

    grid = lambda META: (
        num_tokens,
        triton.cdiv(hidden_size, META["BLOCK_N"]),
    )

    swiglu_kernel[grid](
        gate,
        up,
        out,
        hidden_size,
        m_bucket,
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out

@triton.autotune(
    configs=_swiglu_autotune_configs(),
    key=["M_BUCKET", "hidden_size"],
    restore_value=["gate_ptr"],
    cache_results=True,
)
@triton.jit
def swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    hidden_size,
    M_BUCKET,  # only used by autotune key
    gate_stride_0,
    gate_stride_1,
    up_stride_0,
    up_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < hidden_size

    gate_offsets = gate_ptr + pid_m * gate_stride_0 + cols * gate_stride_1
    up_offsets = up_ptr + pid_m * up_stride_0 + cols * up_stride_1
    out_offsets = out_ptr + pid_m * out_stride_0 + cols * out_stride_1

    gate = tl.load(gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_offsets, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU:
    #   out = up * silu(gate)
    #       = up * gate * sigmoid(gate)
    #
    # tl.sigmoid 是 elementwise sigmoid。
    out = up * gate * tl.sigmoid(gate)

    # tl.store 会根据 out_ptr 的元素类型自动 cast。
    # 如果 out 是 bf16，那么这里会从 fp32 cast 到 bf16 写回。
    tl.store(out_offsets, out, mask=mask)