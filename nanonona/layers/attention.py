import torch
import torch.nn as nn

from nanonona.utils.context import get_context
from nanonona.layers.attention_impl import flash_attn_varlen_func, flash_attn_with_kvcache, store_kvcache

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 推理第n+1个token时，k_cache和v_cache已经有n-1个token的kv了，第n个token的特征值是在这次decode时计算的，所以要把第n个token的kv也存到cache里
        if k_cache.numel() and v_cache.numel(): # cache里没元素时代表还没分配物理显存，也就是warmup阶段，不能存储kv，也方便warmup后计算扣除推理时的激活值显存后能有多少空余显存拿来分配cache
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables
            )
        else:    # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens, block_table=context.block_tables, 
                softmax_scale=self.scale, causal=True
            )
        return o