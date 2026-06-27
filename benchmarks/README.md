# Nanonona Benchmarks

These scripts are additive benchmark helpers. They do not modify the inference
source code.

## Correctness Tests

Run control-plane and CUDA operator tests:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 python -m unittest discover -s test -p "test_mechanisms.py"
TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 python -m unittest discover -s test -p "test_ops_correctness.py"
```

For broader operator bucket coverage:

```bash
NANONONA_EXTENSIVE_OP_TESTS=1 python -m unittest discover -s test -p "test_ops_correctness.py"
```

`test.test_mechanisms` includes one `expectedFailure` that documents a scheduler
edge case: cached prefill admission should ideally be checked against uncached
tokens rather than total prompt length.

## Engine Benchmarks

Fixed prompt lengths, continuous batching:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario fixed \
  --mode continuous \
  --num-requests 32 \
  --prompt-len 512 \
  --output-len 128 \
  --output-dir benchmarks/results/fixed_continuous
```

Serial baseline:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario fixed \
  --mode serial \
  --num-requests 32 \
  --prompt-len 512 \
  --output-len 128 \
  --output-dir benchmarks/results/fixed_serial
```

Prefix-cache ablation:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario shared-prefix \
  --mode continuous \
  --num-requests 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --output-len 128 \
  --output-dir benchmarks/results/prefix_on

TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario shared-prefix \
  --mode continuous \
  --num-requests 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --output-len 128 \
  --disable-prefix-cache \
  --output-dir benchmarks/results/prefix_off
```

Each run writes:

- `summary.json`: throughput, TTFT, latency, cache-hit estimate, block usage.
- `requests.csv`: per-request TTFT and latency.
- `steps.csv`: per-step prefill/decode timing, scheduled tokens, cached blocks.

Use the same model, GPU, dtype, prompt/output lengths, and warmup policy when
comparing against nano-vllm or vLLM.



## Transformers Correctness Comparison

Compare nanonona against HuggingFace Transformers on next-token logits and
fixed-length greedy generation:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/compare_transformers.py \
  --model-path ./DS-R1-Distill-Qwen-1.5B \
  --num-prompts 4 \
  --prompt-len 64 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/compare_transformers
```

The script loads nanonona first, saves its logits/generated token ids on CPU,
releases CUDA memory, then loads Transformers. This avoids OOM when nanonona's
KV cache allocation uses most of the configured GPU memory.

Outputs:

- `summary.json`: top-1 match rate, top-k overlap, cosine similarity, greedy
  token match rate, and backend timings. ("greedy" means always selecting the most probable token)
- `logits_metrics.json`: per-prompt next-token logits comparison.
- `greedy_metrics.json`: per-prompt generated token/text comparison.

Use `--skip-greedy` for logits-only checks or `--skip-logits` for generation-only
checks.


## vLLM Performance Comparison

Install vLLM on the server in the same Python environment first:

```bash
pip install vllm
```

Then compare nanonona with vLLM on the same prompts:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/compare_vllm.py \
  --model-path ./DS-R1-Distill-Qwen-1.5B \
  --scenario fixed \
  --mode continuous \
  --num-requests 32 \
  --prompt-len 512 \
  --max-new-tokens 128 \
  --output-dir benchmarks/results/compare_vllm_fixed
```

Shared-prefix comparison with prefix cache enabled:

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/compare_vllm.py \
  --model-path ./DS-R1-Distill-Qwen-1.5B \
  --scenario shared-prefix \
  --mode continuous \
  --num-requests 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --max-new-tokens 128 \
  --output-dir benchmarks/results/compare_vllm_prefix
```

Disable the CUDA graph of vllm and then conduct the comparison

```bash
TRITON_CACHE_DIR=./triton_cache_sm86 \
CUDA_VISIBLE_DEVICES=7 \
python benchmarks/compare_vllm.py \
  --model-path ./DS-R1-Distill-Qwen-1.5B \
  --scenario fixed \
  --mode continuous \
  --num-requests 32 \
  --prompt-len 512 \
  --max-new-tokens 128 \
  --enforce-eager \
  --output-dir benchmarks/results/compare_vllm_eager
```

Outputs:

- `summary.json`: nanonona/vLLM throughput, total latency, memory, and relative
  throughput ratio.
- `requests.csv`: per-backend request lengths and measured latency where
  available.
- `token_comparison.json`: optional greedy generated-token comparison.

Note: vLLM offline batch inference returns full request outputs after generation,
so this script compares offline throughput directly. TTFT is not available from
the simple offline `LLM.generate` API in the same way as nanonona's internal
scheduler loop.
