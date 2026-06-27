import math
import os
import unittest


try:
    import torch
except Exception:  # pragma: no cover - exercised only on machines without torch
    torch = None


HAS_CUDA = bool(torch is not None and torch.cuda.is_available())
DTYPE = torch.float16 if torch is not None else None


def cuda_only(fn):
    return unittest.skipUnless(HAS_CUDA, "CUDA is required for Triton operator tests")(fn)


def _sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def _m_values():
    if os.getenv("NANONONA_EXTENSIVE_OP_TESTS") == "1":
        return [1, 4, 16, 64, 128, 129, 256, 512]
    return [1, 16, 129]


def _assert_close(actual, expected, *, atol=2e-2, rtol=2e-2):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _repeat_kv_heads(x, num_q_heads):
    """Expand [T, Hkv, D] into [T, Hq, D] using GQA head mapping."""
    num_kv_heads = x.shape[1]
    repeat = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeat, dim=1)


def _reference_prefill_attention(q, k, v, cu_q, cu_k, scale, causal=True):
    """Reference for compact or prefix-cached prefill attention.

    q: [sum(q_lens), num_q_heads, head_dim]
    k/v: [sum(k_lens), num_kv_heads, head_dim]
    cu_q/cu_k: CPU lists of cumulative lengths.
    """
    outs = []
    num_q_heads = q.shape[1]
    for batch_idx in range(len(cu_q) - 1):
        q_start, q_end = cu_q[batch_idx], cu_q[batch_idx + 1]
        k_start, k_end = cu_k[batch_idx], cu_k[batch_idx + 1]
        q_seq = q[q_start:q_end].float()
        k_seq = _repeat_kv_heads(k[k_start:k_end], num_q_heads).float()
        v_seq = _repeat_kv_heads(v[k_start:k_end], num_q_heads).float()
        q_len = q_seq.shape[0]
        k_len = k_seq.shape[0]
        prefix_len = k_len - q_len
        seq_out = []
        for q_pos in range(q_len):
            q_abs_pos = prefix_len + q_pos
            per_head = []
            for head in range(num_q_heads):
                scores = torch.matmul(q_seq[q_pos, head], k_seq[:, head].T) * scale
                if causal:
                    mask = torch.arange(k_len, device=q.device) <= q_abs_pos
                    scores = scores.masked_fill(~mask, -float("inf"))
                probs = torch.softmax(scores, dim=-1)
                per_head.append(torch.matmul(probs, v_seq[:, head]))
            seq_out.append(torch.stack(per_head, dim=0))
        outs.append(torch.stack(seq_out, dim=0))
    return torch.cat(outs, dim=0).to(q.dtype)


def _gather_paged_sequence(cache, block_table_row, seq_len, block_size):
    chunks = []
    remaining = seq_len
    for block_id in block_table_row.tolist():
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        chunks.append(cache[block_id, :take])
        remaining -= take
    return torch.cat(chunks, dim=0)


class TestTritonElementwiseAndMatmul(unittest.TestCase):
    @cuda_only
    def test_linear_matches_torch_reference(self):
        from nanonona.layers.linear import myLinearInterface

        torch.manual_seed(0)
        for m in _m_values():
            with self.subTest(m=m):
                k = 64
                n = 96
                x = torch.randn((m, k), device="cuda", dtype=DTYPE)
                weight = torch.randn((n, k), device="cuda", dtype=DTYPE)
                bias = torch.randn((n,), device="cuda", dtype=DTYPE)

                out = myLinearInterface(x, weight, bias)
                expected = torch.nn.functional.linear(x.float(), weight.float(), bias.float()).to(DTYPE)
                _sync()
                _assert_close(out, expected, atol=3e-2, rtol=3e-2)

    @cuda_only
    def test_swiglu_matches_torch_reference(self):
        from nanonona.layers.activation import SwiGLU

        torch.manual_seed(1)
        for m in _m_values():
            with self.subTest(m=m):
                hidden = 257
                gate = torch.randn((m, hidden), device="cuda", dtype=DTYPE)
                up = torch.randn((m, hidden), device="cuda", dtype=DTYPE)
                expected = (up.float() * torch.nn.functional.silu(gate.float())).to(DTYPE)

                out = SwiGLU(gate.clone(), up)
                _sync()
                _assert_close(out, expected, atol=2e-2, rtol=2e-2)

    @cuda_only
    def test_rmsnorm_matches_torch_reference(self):
        from nanonona.layers.rmsnorm import RMSnorm

        torch.manual_seed(2)
        m = 17
        hidden = 80
        eps = 1e-6
        mod = RMSnorm(hidden, eps=eps).cuda().to(DTYPE)
        mod.weight.data.uniform_(0.5, 1.5)
        x = torch.randn((m, hidden), device="cuda", dtype=DTYPE)

        out = mod(x.clone())
        expected = x.float()
        expected = expected * torch.rsqrt(expected.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = (expected * mod.weight.float()).to(DTYPE)
        _sync()
        _assert_close(out, expected, atol=2e-2, rtol=2e-2)

    @cuda_only
    def test_fused_add_rmsnorm_updates_residual_and_output(self):
        from nanonona.layers.rmsnorm import RMSnorm

        torch.manual_seed(3)
        m = 9
        hidden = 96
        eps = 1e-6
        mod = RMSnorm(hidden, eps=eps).cuda().to(DTYPE)
        mod.weight.data.uniform_(0.5, 1.5)
        x0 = torch.randn((m, hidden), device="cuda", dtype=DTYPE)
        r0 = torch.randn((m, hidden), device="cuda", dtype=DTYPE)
        x = x0.clone()
        residual = r0.clone()

        out, new_residual = mod(x, residual)
        expected_residual = (x0.float() + r0.float()).to(DTYPE)
        expected_out = expected_residual.float()
        expected_out = expected_out * torch.rsqrt(expected_out.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected_out = (expected_out * mod.weight.float()).to(DTYPE)
        _sync()
        _assert_close(new_residual, expected_residual, atol=2e-2, rtol=2e-2)
        _assert_close(out, expected_out, atol=2e-2, rtol=2e-2)

    @cuda_only
    def test_rotary_embedding_matches_reference(self):
        from nanonona.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb_triton

        torch.manual_seed(4)
        num_tokens = 11
        num_heads = 3
        head_dim = 64
        positions = torch.tensor([0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64], device="cuda", dtype=torch.int64)
        x = torch.randn((num_tokens, num_heads, head_dim), device="cuda", dtype=DTYPE)
        rope = RotaryEmbedding(head_dim, head_dim, max_position_embeddings=128, base=10000.0).cuda()

        out = apply_rotary_emb_triton(x, positions, rope.cos_sin_cache)

        cache = rope.cos_sin_cache[positions].squeeze(1).to(x.device)
        half = head_dim // 2
        cos = cache[:, :half].float()
        sin = cache[:, half:].float()
        x1 = x[..., :half].float()
        x2 = x[..., half:].float()
        expected = torch.empty_like(x.float())
        expected[..., :half] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        expected[..., half:] = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        expected = expected.to(DTYPE)
        _sync()
        _assert_close(out, expected, atol=2e-2, rtol=2e-2)


class TestKVCacheAndAttention(unittest.TestCase):
    @cuda_only
    def test_store_kvcache_writes_expected_slots(self):
        from nanonona.layers.attention_impl import store_kvcache

        torch.manual_seed(5)
        num_blocks = 4
        block_size = 4
        num_kv_heads = 2
        head_dim = 32
        num_tokens = 5
        src_k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        src_v = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        v_cache = torch.zeros_like(k_cache)
        slots = torch.tensor([7, 0, 12, 3, 9], device="cuda", dtype=torch.int32)

        store_kvcache(src_k, src_v, k_cache, v_cache, slots)
        _sync()

        flat_k = k_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
        flat_v = v_cache.view_as(flat_k)
        for token_idx, slot in enumerate(slots.tolist()):
            _assert_close(flat_k[slot], src_k[token_idx], atol=0, rtol=0)
            _assert_close(flat_v[slot], src_v[token_idx], atol=0, rtol=0)

    @cuda_only
    def test_compact_prefill_attention_matches_reference(self):
        from nanonona.layers.attention_impl import flash_attn_varlen_func

        torch.manual_seed(6)
        lengths = [3, 5]
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        total = cu[-1]
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 32
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn((total, num_q_heads, head_dim), device="cuda", dtype=DTYPE)
        k = torch.randn((total, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        v = torch.randn((total, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        cu_tensor = torch.tensor(cu, device="cuda", dtype=torch.int32)

        out = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max(lengths),
            cu_seqlens_q=cu_tensor,
            max_seqlen_k=max(lengths),
            cu_seqlens_k=cu_tensor,
            softmax_scale=scale,
            causal=True,
            block_table=None,
        )
        expected = _reference_prefill_attention(q, k, v, cu, cu, scale, causal=True)
        _sync()
        _assert_close(out, expected, atol=4e-2, rtol=4e-2)

    @cuda_only
    def test_paged_prefill_attention_matches_reference_with_prefix_cache(self):
        from nanonona.layers.attention_impl import flash_attn_varlen_func

        torch.manual_seed(7)
        block_size = 4
        num_blocks = 4
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 32
        q_len = 2
        k_len = 6
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn((q_len, num_q_heads, head_dim), device="cuda", dtype=DTYPE)
        compact_k = torch.randn((k_len, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        compact_v = torch.randn((k_len, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
        k_cache[2, :4] = compact_k[:4]
        v_cache[2, :4] = compact_v[:4]
        k_cache[0, :2] = compact_k[4:]
        v_cache[0, :2] = compact_v[4:]
        cu_q = [0, q_len]
        cu_k = [0, k_len]

        out = flash_attn_varlen_func(
            q, k_cache, v_cache,
            max_seqlen_q=q_len,
            cu_seqlens_q=torch.tensor(cu_q, device="cuda", dtype=torch.int32),
            max_seqlen_k=k_len,
            cu_seqlens_k=torch.tensor(cu_k, device="cuda", dtype=torch.int32),
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
        )
        expected = _reference_prefill_attention(q, compact_k, compact_v, cu_q, cu_k, scale, causal=True)
        _sync()
        _assert_close(out, expected, atol=4e-2, rtol=4e-2)

    @cuda_only
    def test_decode_attention_matches_reference_with_noncontiguous_blocks(self):
        from nanonona.layers.attention_impl import flash_attn_with_kvcache

        torch.manual_seed(8)
        batch_size = 2
        block_size = 4
        num_blocks = 5
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 32
        scale = 1.0 / math.sqrt(head_dim)
        context_lens = torch.tensor([5, 3], device="cuda", dtype=torch.int32)
        block_table = torch.tensor([[3, 1], [4, -1]], device="cuda", dtype=torch.int32)
        q = torch.randn((batch_size, 1, num_q_heads, head_dim), device="cuda", dtype=DTYPE)
        k_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device="cuda", dtype=DTYPE)
        v_cache = torch.randn_like(k_cache)

        out = flash_attn_with_kvcache(q, k_cache, v_cache, context_lens, block_table, scale, causal=True)

        expected_rows = []
        for batch_idx in range(batch_size):
            seq_len = int(context_lens[batch_idx].item())
            k_seq = _gather_paged_sequence(k_cache, block_table[batch_idx], seq_len, block_size)
            v_seq = _gather_paged_sequence(v_cache, block_table[batch_idx], seq_len, block_size)
            ref = _reference_prefill_attention(
                q[batch_idx, 0:1],
                k_seq,
                v_seq,
                [0, 1],
                [0, seq_len],
                scale,
                causal=False,
            )
            expected_rows.append(ref.unsqueeze(0))
        expected = torch.cat(expected_rows, dim=0)
        _sync()
        _assert_close(out, expected, atol=4e-2, rtol=4e-2)


if __name__ == "__main__":
    unittest.main()
