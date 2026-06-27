import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanonona.engine.engine import llm_engine
from nanonona.engine.scheduler import Scheduler
from nanonona.engine.sequence import Sequence
from nanonona.utils.context import reset_context
from nanonona.utils.sample_params import SamplingParams


class GreedySampler(torch.nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        del temperatures
        return torch.argmax(logits.float(), dim=-1)


@dataclass
class LogitsMetric:
    prompt_id: int
    prompt_len: int
    top1_match: bool
    nanonona_top1: int
    transformers_top1: int
    topk_overlap: float
    cosine_similarity: float
    max_abs_error: float
    mean_abs_error: float


@dataclass
class GreedyMetric:
    prompt_id: int
    prompt_len: int
    generated_len: int
    exact_match: bool
    token_match_rate: float
    first_mismatch_index: int | None
    nanonona_tokens: list[int]
    transformers_tokens: list[int]
    nanonona_text: str
    transformers_text: str


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _dtype_from_arg(dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def _tokens_from_text(tokenizer, target_len: int, salt: int = 0) -> list[int]:
    base = (
        "Nanonona compares logits against the reference Transformers implementation. "
        f"Prompt salt {salt}. "
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
        if args.num_prompts is not None:
            texts = texts[: args.num_prompts]
        token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        return texts, token_ids

    texts = []
    token_ids = []
    for i in range(args.num_prompts):
        ids = _tokens_from_text(tokenizer, args.prompt_len, salt=i)
        token_ids.append(ids)
        texts.append(tokenizer.decode(ids))
    return texts, token_ids


def run_nanonona(args, prompt_token_ids: list[list[int]]):
    _cleanup_cuda()
    engine = llm_engine(model_path=args.model_path)
    engine.modelrunner.sampler = GreedySampler()

    logits_cpu = None
    generation_outputs = None
    timings = {}

    if not args.skip_logits:
        rows = []
        _sync()
        start = time.perf_counter()
        for prompt in prompt_token_ids:
            engine.scheduler = Scheduler()
            seq = Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True))
            engine.scheduler.add_sequence(seq)
            seqs, is_prefill = engine.scheduler.schedule()
            if not is_prefill or seqs != [seq]:
                raise RuntimeError("Expected one prefill sequence for logits comparison.")
            try:
                input_ids, positions = engine.modelrunner.prepare_prefill(seqs)
                with torch.inference_mode():
                    logits = engine.modelrunner.model(input_ids=input_ids, positions=positions)
                rows.append(logits.detach().float().cpu())
            finally:
                reset_context()
                if seq.block_table:
                    engine.scheduler.blockManager.deallocate(seq)
        _sync()
        timings["nanonona_logits_s"] = time.perf_counter() - start
        logits_cpu = torch.cat(rows, dim=0)

    if not args.skip_greedy:
        params = SamplingParams(temperature=1.0, max_tokens=args.max_new_tokens, ignore_eos=True)
        engine.scheduler = Scheduler()
        _sync()
        start = time.perf_counter()
        outputs = engine.generate(prompt_token_ids, params)
        _sync()
        timings["nanonona_greedy_s"] = time.perf_counter() - start
        generation_outputs = [out["token_ids"] for out in outputs]

    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    del engine
    _cleanup_cuda()
    return logits_cpu, generation_outputs, timings, peak_memory


def load_transformers_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_dtype_from_arg(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval().to(args.device)
    return tokenizer, model


def run_transformers(args, prompt_token_ids: list[list[int]]):
    _cleanup_cuda()
    tokenizer, model = load_transformers_model(args)
    logits_cpu = None
    generation_outputs = None
    timings = {}

    if not args.skip_logits:
        rows = []
        _sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for prompt in prompt_token_ids:
                input_ids = torch.tensor([prompt], device=args.device, dtype=torch.long)
                logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
                rows.append(logits.detach().float().cpu())
        _sync()
        timings["transformers_logits_s"] = time.perf_counter() - start
        logits_cpu = torch.cat(rows, dim=0)

    if not args.skip_greedy:
        generated = []
        _sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for prompt in prompt_token_ids:
                input_ids = torch.tensor([prompt], device=args.device, dtype=torch.long)
                past_key_values = None
                out_tokens = []
                for _ in range(args.max_new_tokens):
                    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                    next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1)
                    out_tokens.append(int(next_token.item()))
                    past_key_values = outputs.past_key_values
                    input_ids = next_token[:, None]
                generated.append(out_tokens)
        _sync()
        timings["transformers_greedy_s"] = time.perf_counter() - start
        generation_outputs = generated

    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    del model
    _cleanup_cuda()
    return tokenizer, logits_cpu, generation_outputs, timings, peak_memory


def compare_logits(nanonona_logits: torch.Tensor, transformers_logits: torch.Tensor, prompt_lens: list[int], topk: int):
    metrics = []
    for i in range(nanonona_logits.shape[0]):
        a = nanonona_logits[i].float()
        b = transformers_logits[i].float()
        a_top = torch.topk(a, topk).indices.tolist()
        b_top = torch.topk(b, topk).indices.tolist()
        overlap = len(set(a_top) & set(b_top)) / topk
        diff = (a - b).abs()
        metrics.append(
            LogitsMetric(
                prompt_id=i,
                prompt_len=prompt_lens[i],
                top1_match=a_top[0] == b_top[0],
                nanonona_top1=a_top[0],
                transformers_top1=b_top[0],
                topk_overlap=overlap,
                cosine_similarity=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                max_abs_error=diff.max().item(),
                mean_abs_error=diff.mean().item(),
            )
        )
    return metrics


def compare_greedy(nanonona_tokens, transformers_tokens, tokenizer, prompt_lens: list[int]):
    metrics = []
    for i, (a, b) in enumerate(zip(nanonona_tokens, transformers_tokens)):
        matches = [x == y for x, y in zip(a, b)]
        first_mismatch = None
        for idx, matched in enumerate(matches):
            if not matched:
                first_mismatch = idx
                break
        metrics.append(
            GreedyMetric(
                prompt_id=i,
                prompt_len=prompt_lens[i],
                generated_len=min(len(a), len(b)),
                exact_match=a == b,
                token_match_rate=sum(matches) / max(1, min(len(a), len(b))),
                first_mismatch_index=first_mismatch,
                nanonona_tokens=a,
                transformers_tokens=b,
                nanonona_text=tokenizer.decode(a),
                transformers_text=tokenizer.decode(b),
            )
        )
    return metrics


def summarize(args, logits_metrics, greedy_metrics, timings, nanonona_peak, transformers_peak):
    summary = {
        "model_path": args.model_path,
        "num_prompts": args.num_prompts,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "topk": args.topk,
        "timings": timings,
        "nanonona_peak_cuda_memory_bytes": nanonona_peak,
        "transformers_peak_cuda_memory_bytes": transformers_peak,
    }
    if logits_metrics:
        summary.update(
            {
                "logits_top1_match_rate": sum(m.top1_match for m in logits_metrics) / len(logits_metrics),
                "logits_topk_overlap_avg": sum(m.topk_overlap for m in logits_metrics) / len(logits_metrics),
                "logits_cosine_similarity_avg": sum(m.cosine_similarity for m in logits_metrics) / len(logits_metrics),
                "logits_max_abs_error_max": max(m.max_abs_error for m in logits_metrics),
                "logits_mean_abs_error_avg": sum(m.mean_abs_error for m in logits_metrics) / len(logits_metrics),
            }
        )
    if greedy_metrics:
        summary.update(
            {
                "greedy_exact_match_rate": sum(m.exact_match for m in greedy_metrics) / len(greedy_metrics),
                "greedy_token_match_rate_avg": sum(m.token_match_rate for m in greedy_metrics) / len(greedy_metrics),
            }
        )
    return summary


def write_outputs(output_dir: Path, summary, logits_metrics, greedy_metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if logits_metrics:
        (output_dir / "logits_metrics.json").write_text(
            json.dumps([asdict(m) for m in logits_metrics], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if greedy_metrics:
        (output_dir / "greedy_metrics.json").write_text(
            json.dumps([asdict(m) for m in greedy_metrics], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare nanonona against HuggingFace Transformers.")
    parser.add_argument("--model-path", required=True, help="Local model directory used by both backends.")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/compare_transformers"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-logits", action="store_true")
    parser.add_argument("--skip-greedy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_logits and args.skip_greedy:
        raise ValueError("At least one of logits or greedy comparison must be enabled.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device cuda is used.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    prompt_texts, prompt_token_ids = build_prompts(args, tokenizer)
    prompt_lens = [len(ids) for ids in prompt_token_ids]
    args.num_prompts = len(prompt_token_ids)

    nanonona_logits, nanonona_tokens, nanonona_timings, nanonona_peak = run_nanonona(args, prompt_token_ids)
    hf_tokenizer, hf_logits, hf_tokens, hf_timings, hf_peak = run_transformers(args, prompt_token_ids)

    logits_metrics = []
    if not args.skip_logits:
        logits_metrics = compare_logits(nanonona_logits, hf_logits, prompt_lens, args.topk)

    greedy_metrics = []
    if not args.skip_greedy:
        greedy_metrics = compare_greedy(nanonona_tokens, hf_tokens, hf_tokenizer, prompt_lens)

    timings = {**nanonona_timings, **hf_timings}
    summary = summarize(args, logits_metrics, greedy_metrics, timings, nanonona_peak, hf_peak)
    summary["prompt_texts"] = prompt_texts
    write_outputs(args.output_dir, summary, logits_metrics, greedy_metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
