import argparse
import csv
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from unittest import mock

import torch

from nanonona.engine.blockManager import BlockManager
from nanonona.engine.engine import llm_engine
from nanonona.engine.scheduler import Scheduler
from nanonona.engine.sequence import Sequence
from nanonona.utils.config import Config
from nanonona.utils.sample_params import SamplingParams


@dataclass
class RequestMetric:
    request_id: int
    prompt_len: int
    output_len: int
    arrival_s: float
    first_token_s: float | None = None
    finished_s: float | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.arrival_s

    @property
    def latency_s(self) -> float | None:
        if self.finished_s is None:
            return None
        return self.finished_s - self.arrival_s


@dataclass
class StepMetric:
    step_idx: int
    is_prefill: bool
    batch_size: int
    scheduled_tokens: int
    cached_blocks: int
    elapsed_s: float
    running: int
    waiting: int
    free_blocks: int
    allocated_blocks: int


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _tokens_from_text(tokenizer, target_len: int, salt: int = 0) -> list[int]:
    base = (
        "The quick brown fox studies paged attention, prefix caching, and continuous batching. "
        f"Request salt {salt}. "
    )
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        token_ids.extend(tokenizer.encode(base, add_special_tokens=False))
    return token_ids[:target_len]


def build_prompts(args, tokenizer) -> list[list[int]]:
    if args.scenario == "shared-prefix":
        prefix = _tokens_from_text(tokenizer, args.prefix_len, salt=0)
        prompts = []
        for i in range(args.num_requests):
            suffix = _tokens_from_text(tokenizer, args.suffix_len, salt=i + 1)
            prompts.append(prefix + suffix)
        return prompts

    prompts = []
    for i in range(args.num_requests):
        prompts.append(_tokens_from_text(tokenizer, args.prompt_len, salt=i))
    return prompts


def _disable_prefix_cache_context(enabled: bool):
    if not enabled:
        return nullcontext()

    def no_prefix_cache(self, seq):
        seq.block_table.clear()
        seq.num_cached_blocks = 0
        return 0

    return mock.patch.object(BlockManager, "_num_cached_blocks", no_prefix_cache)


def _add_sequence(engine: llm_engine, prompt: list[int], params: SamplingParams) -> Sequence:
    seq = Sequence(prompt, params)
    engine.scheduler.add_sequence(seq)
    return seq


def _run_until_idle(engine: llm_engine, metrics: dict[int, RequestMetric], step_metrics: list[StepMetric], start_s: float):
    step_idx = len(step_metrics)
    while not engine.scheduler.is_finished():
        seqs, is_prefill = engine.scheduler.schedule()
        scheduled_tokens = 0
        cached_blocks = 0
        if is_prefill:
            for seq in seqs:
                cached_blocks += seq.num_cached_blocks
                scheduled_tokens += len(seq) - seq.num_cached_blocks * Config.engine.block_size
        else:
            scheduled_tokens = len(seqs)

        _sync()
        step_start = time.perf_counter()
        token_ids = engine.modelrunner.run(seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids)
        _sync()
        step_end = time.perf_counter()

        now = step_end - start_s
        for seq in seqs:
            metric = metrics[seq.seq_id]
            if metric.first_token_s is None and seq.num_completion_tokens > 0:
                metric.first_token_s = now
            if seq.is_completed and metric.finished_s is None:
                metric.finished_s = now
                metric.output_len = seq.num_completion_tokens

        bm = engine.scheduler.blockManager
        step_metrics.append(
            StepMetric(
                step_idx=step_idx,
                is_prefill=is_prefill,
                batch_size=len(seqs),
                scheduled_tokens=scheduled_tokens,
                cached_blocks=cached_blocks,
                elapsed_s=step_end - step_start,
                running=len(engine.scheduler.running),
                waiting=len(engine.scheduler.waiting),
                free_blocks=len(bm.free_blocks),
                allocated_blocks=len(bm.allocated_blocks),
            )
        )
        step_idx += 1


def run_continuous(engine: llm_engine, prompts: list[list[int]], params: SamplingParams):
    metrics: dict[int, RequestMetric] = {}
    step_metrics: list[StepMetric] = []
    start_s = time.perf_counter()
    for prompt in prompts:
        seq = _add_sequence(engine, prompt, params)
        metrics[seq.seq_id] = RequestMetric(
            request_id=seq.seq_id,
            prompt_len=len(prompt),
            output_len=0,
            arrival_s=0.0,
        )
    _run_until_idle(engine, metrics, step_metrics, start_s)
    return metrics, step_metrics, time.perf_counter() - start_s


def run_serial(engine: llm_engine, prompts: list[list[int]], params: SamplingParams):
    metrics: dict[int, RequestMetric] = {}
    step_metrics: list[StepMetric] = []
    start_s = time.perf_counter()
    for prompt in prompts:
        arrival_s = time.perf_counter() - start_s
        seq = _add_sequence(engine, prompt, params)
        metrics[seq.seq_id] = RequestMetric(
            request_id=seq.seq_id,
            prompt_len=len(prompt),
            output_len=0,
            arrival_s=arrival_s,
        )
        _run_until_idle(engine, metrics, step_metrics, start_s)
    return metrics, step_metrics, time.perf_counter() - start_s


def summarize(metrics: Iterable[RequestMetric], step_metrics: list[StepMetric], total_s: float, args) -> dict:
    rows = list(metrics)
    ttfts = [m.ttft_s for m in rows if m.ttft_s is not None]
    latencies = [m.latency_s for m in rows if m.latency_s is not None]
    prompt_tokens = sum(m.prompt_len for m in rows)
    output_tokens = sum(m.output_len for m in rows)
    prefill_steps = [s for s in step_metrics if s.is_prefill]
    decode_steps = [s for s in step_metrics if not s.is_prefill]
    peak_allocated_blocks = max((s.allocated_blocks for s in step_metrics), default=0)
    total_cached_blocks = sum(s.cached_blocks for s in prefill_steps)
    total_prefill_tokens = sum(s.scheduled_tokens for s in prefill_steps)

    return {
        "scenario": args.scenario,
        "mode": args.mode,
        "disable_prefix_cache": args.disable_prefix_cache,
        "num_requests": len(rows),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "total_s": total_s,
        "output_tok_per_s": output_tokens / total_s if total_s > 0 else 0.0,
        "total_tok_per_s": (prompt_tokens + output_tokens) / total_s if total_s > 0 else 0.0,
        "ttft_avg_s": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "prefill_steps": len(prefill_steps),
        "decode_steps": len(decode_steps),
        "scheduled_prefill_tokens": total_prefill_tokens,
        "cached_blocks": total_cached_blocks,
        "cache_saved_tokens_estimate": total_cached_blocks * Config.engine.block_size,
        "peak_allocated_blocks": peak_allocated_blocks,
        "max_kvcache_blocks": Config.engine.max_num_kvcache_blocks,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    }


def write_outputs(output_dir: Path, summary: dict, request_metrics: dict[int, RequestMetric], step_metrics: list[StepMetric]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["request_id", "prompt_len", "output_len", "arrival_s", "first_token_s", "finished_s", "ttft_s", "latency_s"],
        )
        writer.writeheader()
        for metric in sorted(request_metrics.values(), key=lambda m: m.request_id):
            row = asdict(metric)
            row["ttft_s"] = metric.ttft_s
            row["latency_s"] = metric.latency_s
            writer.writerow(row)

    with (output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(step_metrics[0]).keys()) if step_metrics else ["step_idx"])
        writer.writeheader()
        for metric in step_metrics:
            writer.writerow(asdict(metric))


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end benchmark for nanonona inference.")
    parser.add_argument("--scenario", choices=["fixed", "shared-prefix"], default="fixed")
    parser.add_argument("--mode", choices=["continuous", "serial"], default="continuous")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/latest"))
    parser.add_argument("--model-path", default=None, help="Overrides Config.model.path when provided.")
    parser.add_argument("--warmup-requests", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for engine benchmarks.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.cuda.reset_peak_memory_stats()

    with _disable_prefix_cache_context(args.disable_prefix_cache):
        engine = llm_engine(model_path=args.model_path)
        params = SamplingParams(temperature=args.temperature, max_tokens=args.output_len, ignore_eos=True)

        if args.warmup_requests > 0:
            warmup_prompts = build_prompts(
                argparse.Namespace(
                    scenario="fixed",
                    prompt_len=min(args.prompt_len, 64),
                    num_requests=args.warmup_requests,
                    prefix_len=args.prefix_len,
                    suffix_len=args.suffix_len,
                ),
                engine.tokenizer,
            )
            run_serial(engine, warmup_prompts, SamplingParams(temperature=args.temperature, max_tokens=1, ignore_eos=True))
            engine.scheduler = Scheduler()
            torch.cuda.reset_peak_memory_stats()

        prompts = build_prompts(args, engine.tokenizer)
        if args.mode == "serial":
            request_metrics, step_metrics, total_s = run_serial(engine, prompts, params)
        else:
            request_metrics, step_metrics, total_s = run_continuous(engine, prompts, params)

    summary = summarize(request_metrics.values(), step_metrics, total_s, args)
    write_outputs(args.output_dir, summary, request_metrics, step_metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
