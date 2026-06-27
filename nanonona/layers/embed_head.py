import triton
import torch
from torch import nn
import triton.language as tl

from nanonona.utils.context import get_context
from nanonona.layers.linear import myLinearInterface

class VocabEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        y = myEmbeddingInterface(x, self.weight)
        return y

class ParallelLMHead(VocabEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = myLinearInterface(x, self.weight, None)
        return logits
    

@triton.jit
def _embedding_fwd_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    embedding_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < embedding_dim

    token_id = tl.load(input_ptr + pid_s)

    weight_offsets = token_id * embedding_dim + offs_d
    output_offsets = pid_s * embedding_dim + offs_d

    values = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offsets, values, mask=mask)


def myEmbeddingInterface(input: torch.Tensor, weight: torch.Tensor):
    """
    input:  [seq_len], int32 / int64, CUDA tensor
    weight: [vocab_size, embedding_dim], CUDA tensor
    output: [seq_len, embedding_dim]
    """
    assert input.is_cuda
    assert weight.is_cuda
    assert input.dim() == 1
    assert weight.dim() == 2
    assert input.dtype in (torch.int32, torch.int64)

    input = input.contiguous()
    weight = weight.contiguous()

    seq_len = input.shape[0]
    embedding_dim = weight.shape[1]

    output = torch.empty(
        (seq_len, embedding_dim),
        device=weight.device,
        dtype=weight.dtype,
    )

    block_d = triton.next_power_of_2(embedding_dim)

    _embedding_fwd_kernel[(seq_len,)](
        input,
        weight,
        output,
        embedding_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )

    return output
