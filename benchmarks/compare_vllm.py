import argparse
import csv
import gc
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from nanonona.engine.blockManager import BlockManager
from nanonona.engine.engine import llm_engine
from nanonona.engine.scheduler import Scheduler
from nanonona.engine.sequence import Sequence
from nanonona.utils.sample_params import SamplingParams as NanononaSamplingParams


class GreedySampler(torch.nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        del temperatures
        return torch.argmax(logits.float(), dim=-1)


@dataclass
class RequestMetric:
    backend: str
    request_id: int
    prompt_len: int
    output_len: int
    arrival_s: float
    latency_s: float | None
    ttft_s: float | None
    num_cached_tokens: int | None


@dataclass
class BackendSummary:
    backend: str
    mode: str
    scenario: str
    num_requests: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    total_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    latency_avg_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    ttft_avg_s: float | None
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    peak_cuda_memory_bytes: int


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _avg(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _tokens_from_text(tokenizer, target_len: int, salt: int = 0) -> list[int]:
    base = (
        "This benchmark compares nanonona with vLLM on the same Qwen2 checkpoint. "
        f"Request salt {salt}. "
    )
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        token_ids.extend(tokenizer.encode(base, add_special_tokens=False))
    return token_ids[:target_len]


def build_prompts(args, tokenizer) -> tuple[list[str], list[list[int]]]:
    if args.prompts_file is not None:
        texts = [
            line.strip()
            for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.num_requests is not None:
            texts = texts[: args.num_requests]
        token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        return texts, token_ids

    if args.scenario == "shared-prefix":
        prefix = _tokens_from_text(tokenizer, args.prefix_len, salt=0)
        token_ids = []
        for i in range(args.num_requests):
            suffix = _tokens_from_text(tokenizer, args.suffix_len, salt=i + 1)
            token_ids.append(prefix + suffix)
        return [tokenizer.decode(ids) for ids in token_ids], token_ids

    token_ids = [_tokens_from_text(tokenizer, args.prompt_len, salt=i) for i in range(args.num_requests)]
    return [tokenizer.decode(ids) for ids in token_ids], token_ids


def _nanonona_no_prefix_context(enabled: bool):
    if not enabled:
        return nullcontext()

    def no_prefix_cache(self, seq):
        seq.block_table.clear()
        seq.num_cached_blocks = 0
        return 0

    from unittest import mock

    return mock.patch.object(BlockManager, "_num_cached_blocks", no_prefix_cache)


def _run_nanonona_until_idle(engine, metrics: dict[int, RequestMetric], start_s: float):
    while not engine.scheduler.is_finished():
        seqs, is_prefill = engine.scheduler.schedule()
        _sync()
        token_start = time.perf_counter()
        token_ids = engine.modelrunner.run(seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids)
        _sync()
        now_s = time.perf_counter() - start_s
        del token_start
        for seq in seqs:
            metric = metrics[seq.seq_id]
            if metric.ttft_s is None and seq.num_completion_tokens > 0:
                metric.ttft_s = now_s - metric.arrival_s
            if seq.is_completed and metric.latency_s is None:
                metric.latency_s = now_s - metric.arrival_s
                metric.output_len = seq.num_completion_tokens


def run_nanonona(args, prompt_token_ids: list[list[int]]):
    _cleanup_cuda()
    with _nanonona_no_prefix_context(args.disable_prefix_cache):
        engine = llm_engine(model_path=args.model_path)
        if args.temperature <= 0:
            engine.modelrunner.sampler = GreedySampler()
            temperature = 1.0
        else:
            temperature = args.temperature
        params = NanononaSamplingParams(
            temperature=temperature,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
        )

        request_metrics: dict[int, RequestMetric] = {}
        generated_tokens: dict[int, list[int]] = {}
        start_s = time.perf_counter()

        if args.mode == "continuous":
            for prompt in prompt_token_ids:
                seq = Sequence(prompt, params)
                engine.scheduler.add_sequence(seq)
                request_metrics[seq.seq_id] = RequestMetric(
                    backend="nanonona",
                    request_id=seq.seq_id,
                    prompt_len=len(prompt),
                    output_len=0,
                    arrival_s=0.0,
                    latency_s=None,
                    ttft_s=None,
                    num_cached_tokens=None,
                )
            _run_nanonona_until_idle(engine, request_metrics, start_s)
        else:
            for prompt in prompt_token_ids:
                arrival_s = time.perf_counter() - start_s
                seq = Sequence(prompt, params)
                engine.scheduler.add_sequence(seq)
                request_metrics[seq.seq_id] = RequestMetric(
                    backend="nanonona",
                    request_id=seq.seq_id,
                    prompt_len=len(prompt),
                    output_len=0,
                    arrival_s=arrival_s,
                    latency_s=None,
                    ttft_s=None,
                    num_cached_tokens=None,
                )
                _run_nanonona_until_idle(engine, request_metrics, start_s)

        for seq in list(engine.scheduler.running) + list(engine.scheduler.waiting):
            generated_tokens[seq.seq_id] = seq.generated_token_ids

        # Completed sequences are no longer retained by the scheduler, so collect
        # generated tokens through a second deterministic run only when requested
        # would be too expensive. Instead, use engine.generate-style outputs by
        # recording them from request metrics in the scheduler loop is not possible
        # without touching source code. We keep token-level comparison optional and
        # derive it from a direct generate pass below when requested.
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        total_s = time.perf_counter() - start_s
        if args.compare_tokens:
            engine.scheduler = Scheduler()
            outputs = engine.generate(prompt_token_ids, params)
            generated_by_order = [out["token_ids"] for out in outputs]
        else:
            generated_by_order = []

        del engine
        _cleanup_cuda()
        return list(request_metrics.values()), generated_by_order, total_s, peak


def _make_vllm_sampling_params(args):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install it on the server with `pip install vllm`.") from exc

    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }
    if args.temperature > 0:
        kwargs["top_p"] = args.top_p
        kwargs["seed"] = args.seed
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        # Keep compatibility with older vLLM wheels whose SamplingParams may not
        # expose every newer keyword.
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def _make_vllm_llm(args):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install it on the server with `pip install vllm`.") from exc

    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "tokenizer": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.vllm_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "generation_config": "vllm",
        "enable_prefix_caching": not args.disable_prefix_cache,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        kwargs["enforce_eager"] = True

    try:
        return LLM(**kwargs)
    except TypeError:
        # Some older vLLM versions do not expose these convenience arguments.
        # Retry with the most portable constructor surface.
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("generation_config", None)
        fallback_kwargs.pop("enable_prefix_caching", None)
        fallback_kwargs.pop("enforce_eager", None)
        return LLM(**fallback_kwargs)


def _vllm_prompts_from_token_ids(prompt_token_ids: list[list[int]]):
    return [{"prompt_token_ids": ids} for ids in prompt_token_ids]


def _run_vllm_generate(llm, prompts, sampling_params, *, use_tqdm: bool):
    try:
        return llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    except (TypeError, ValueError):
        # Older vLLM versions may be less happy with token-prompt dictionaries.
        # Fall back to prompt strings through vLLM's own tokenizer.
        tokenizer = llm.get_tokenizer()
        text_prompts = [tokenizer.decode(p["prompt_token_ids"]) for p in prompts]
        return llm.generate(text_prompts, sampling_params, use_tqdm=use_tqdm)


def run_vllm(args, prompt_token_ids: list[list[int]]):
    _cleanup_cuda()
    llm = _make_vllm_llm(args)
    sampling_params = _make_vllm_sampling_params(args)
    prompts = _vllm_prompts_from_token_ids(prompt_token_ids)
    request_metrics: list[RequestMetric] = []
    generated_by_order: list[list[int]] = []

    _sync()
    start_s = time.perf_counter()
    if args.mode == "continuous":
        outputs = _run_vllm_generate(llm, prompts, sampling_params, use_tqdm=args.vllm_tqdm)
        _sync()
        total_s = time.perf_counter() - start_s
        for idx, output in enumerate(outputs):
            token_ids = list(output.outputs[0].token_ids)
            generated_by_order.append(token_ids)
            request_metrics.append(
                RequestMetric(
                    backend="vllm",
                    request_id=idx,
                    prompt_len=len(prompt_token_ids[idx]),
                    output_len=len(token_ids),
                    arrival_s=0.0,
                    latency_s=total_s,
                    ttft_s=None,
                    num_cached_tokens=getattr(output, "num_cached_tokens", None),
                )
            )
    else:
        total_s = 0.0
        for idx, prompt in enumerate(prompts):
            arrival_s = time.perf_counter() - start_s
            req_start = time.perf_counter()
            outputs = _run_vllm_generate(llm, [prompt], sampling_params, use_tqdm=False)
            _sync()
            latency_s = time.perf_counter() - req_start
            total_s = time.perf_counter() - start_s
            token_ids = list(outputs[0].outputs[0].token_ids)
            generated_by_order.append(token_ids)
            request_metrics.append(
                RequestMetric(
                    backend="vllm",
                    request_id=idx,
                    prompt_len=len(prompt_token_ids[idx]),
                    output_len=len(token_ids),
                    arrival_s=arrival_s,
                    latency_s=latency_s,
                    ttft_s=None,
                    num_cached_tokens=getattr(outputs[0], "num_cached_tokens", None),
                )
            )

    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    del llm
    _cleanup_cuda()
    return request_metrics, generated_by_order, total_s, peak


def summarize_backend(backend: str, args, metrics: list[RequestMetric], total_s: float, peak: int):
    prompt_tokens = sum(m.prompt_len for m in metrics)
    output_tokens = sum(m.output_len for m in metrics)
    latencies = [m.latency_s for m in metrics if m.latency_s is not None]
    ttfts = [m.ttft_s for m in metrics if m.ttft_s is not None]
    return BackendSummary(
        backend=backend,
        mode=args.mode,
        scenario=args.scenario,
        num_requests=len(metrics),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        total_s=total_s,
        output_tok_per_s=output_tokens / total_s if total_s > 0 else 0.0,
        total_tok_per_s=(prompt_tokens + output_tokens) / total_s if total_s > 0 else 0.0,
        latency_avg_s=_avg(latencies),
        latency_p50_s=_percentile(latencies, 0.50),
        latency_p95_s=_percentile(latencies, 0.95),
        ttft_avg_s=_avg(ttfts),
        ttft_p50_s=_percentile(ttfts, 0.50),
        ttft_p95_s=_percentile(ttfts, 0.95),
        peak_cuda_memory_bytes=peak,
    )


def compare_tokens(nanonona_tokens: list[list[int]], vllm_tokens: list[list[int]]):
    rows = []
    if not nanonona_tokens or not vllm_tokens:
        return rows
    for idx, (a, b) in enumerate(zip(nanonona_tokens, vllm_tokens)):
        n = min(len(a), len(b))
        matches = [a[i] == b[i] for i in range(n)]
        first_mismatch = None
        for i, matched in enumerate(matches):
            if not matched:
                first_mismatch = i
                break
        rows.append(
            {
                "request_id": idx,
                "generated_len_nanonona": len(a),
                "generated_len_vllm": len(b),
                "exact_match": a == b,
                "token_match_rate": sum(matches) / max(1, n),
                "first_mismatch_index": first_mismatch,
                "nanonona_tokens": a,
                "vllm_tokens": b,
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    summaries: list[BackendSummary],
    request_metrics: list[RequestMetric],
    token_comparison: list[dict[str, Any]],
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "config": {
            "model_path": args.model_path,
            "scenario": args.scenario,
            "mode": args.mode,
            "num_requests": args.num_requests,
            "prompt_len": args.prompt_len,
            "prefix_len": args.prefix_len,
            "suffix_len": args.suffix_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "disable_prefix_cache": args.disable_prefix_cache,
        },
        "summaries": [asdict(summary) for summary in summaries],
    }
    if len(summaries) == 2:
        by_backend = {s.backend: s for s in summaries}
        if "nanonona" in by_backend and "vllm" in by_backend:
            nn = by_backend["nanonona"]
            vv = by_backend["vllm"]
            summary_payload["relative"] = {
                "nanonona_output_tok_per_s_vs_vllm": nn.output_tok_per_s / vv.output_tok_per_s
                if vv.output_tok_per_s > 0
                else None,
                "nanonona_total_tok_per_s_vs_vllm": nn.total_tok_per_s / vv.total_tok_per_s
                if vv.total_tok_per_s > 0
                else None,
            }
    if token_comparison:
        summary_payload["token_comparison"] = {
            "exact_match_rate": sum(row["exact_match"] for row in token_comparison) / len(token_comparison),
            "token_match_rate_avg": sum(row["token_match_rate"] for row in token_comparison) / len(token_comparison),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(request_metrics[0]).keys()) if request_metrics else ["backend"])
        writer.writeheader()
        for row in request_metrics:
            writer.writerow(asdict(row))

    if token_comparison:
        (output_dir / "token_comparison.json").write_text(
            json.dumps(token_comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare nanonona offline inference against vLLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["both", "nanonona", "vllm"], default="both")
    parser.add_argument("--scenario", choices=["fixed", "shared-prefix"], default="fixed")
    parser.add_argument("--mode", choices=["continuous", "serial"], default="continuous")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--respect-eos", action="store_false", dest="ignore_eos")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--compare-tokens", action="store_true")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/compare_vllm"))
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--vllm-tqdm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.compare_tokens and args.temperature > 0:
        raise ValueError("--compare-tokens requires greedy decoding, so use --temperature 0.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    _, prompt_token_ids = build_prompts(args, tokenizer)
    args.num_requests = len(prompt_token_ids)

    summaries: list[BackendSummary] = []
    request_rows: list[RequestMetric] = []
    nanonona_tokens: list[list[int]] = []
    vllm_tokens: list[list[int]] = []

    if args.backend in ("both", "nanonona"):
        nn_metrics, nanonona_tokens, nn_total_s, nn_peak = run_nanonona(args, prompt_token_ids)
        summaries.append(summarize_backend("nanonona", args, nn_metrics, nn_total_s, nn_peak))
        request_rows.extend(nn_metrics)

    if args.backend in ("both", "vllm"):
        vv_metrics, vllm_tokens, vv_total_s, vv_peak = run_vllm(args, prompt_token_ids)
        summaries.append(summarize_backend("vllm", args, vv_metrics, vv_total_s, vv_peak))
        request_rows.extend(vv_metrics)

    token_comparison = compare_tokens(nanonona_tokens, vllm_tokens) if args.compare_tokens else []
    write_outputs(args.output_dir, summaries, request_rows, token_comparison, args)
    print(json.dumps({"summaries": [asdict(s) for s in summaries]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
