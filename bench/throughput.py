"""Offline throughput: N requests submitted at once, output tokens per second.

Same workload and flags as bench/vllm_throughput.py: random prompts, greedy,
ignore_eos, bf16, max 256 concurrent sequences.
"""

import argparse
import time

import torch

from bench.common import save_result
from bench.workload import random_requests
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams


def run(engine: LLMEngine, reqs) -> dict:
    seqs = [engine.add_request(p, SamplingParams(max_tokens=n, ignore_eos=True)) for p, n in reqs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while engine.has_unfinished():
        engine.step()
        steps += 1
    elapsed = time.perf_counter() - t0
    out_tokens = sum(len(s.output_token_ids) for s in seqs)
    in_tokens = sum(len(s.prompt_token_ids) for s in seqs)
    return {"elapsed_s": elapsed, "output_tokens": out_tokens, "input_tokens": in_tokens,
            "output_tok_per_s": out_tokens / elapsed, "total_tok_per_s": (in_tokens + out_tokens) / elapsed,
            "requests_per_s": len(reqs) / elapsed, "engine_steps": steps,
            "preemptions": engine.scheduler.num_preemptions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--attn-backend", default="triton")
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = EngineConfig(max_num_seqs=args.max_num_seqs, attn_backend=args.attn_backend,
                       use_cuda_graphs=not args.no_cuda_graphs)
    engine = LLMEngine(cfg)
    run(engine, random_requests(32, args.input_len, 16, seed=1))  # warmup
    engine.scheduler.num_preemptions = 0
    res = run(engine, random_requests(args.num_prompts, args.input_len, args.output_len))
    print(f"ours: {res['output_tok_per_s']:.0f} output tok/s ({res['elapsed_s']:.1f}s, "
          f"{res['preemptions']} preemptions, {engine.num_blocks} KV blocks)")
    save_result(f"throughput_ours{args.tag}", {
        "system": "ours", "workload": vars(args), "num_kv_blocks": engine.num_blocks,
        "engine_config": {k: str(v) for k, v in vars(cfg).items()}, **res})


if __name__ == "__main__":
    main()
