"""Concurrent requests that fit in a fixed KV-cache budget: paged blocks vs per-request preallocation.

Both runs use the same engine, the same KV pool (same bytes of GPU memory) and the same
burst of requests, submitted at once so memory is the only limit on concurrency. The
preallocating allocator reserves max_model_len tokens per request at admission (like a
contiguous per-request KV buffer); the paged allocator takes one 16-token block at a time.
We also report an oracle preallocator that reserves exactly prompt + max_tokens per request
(it knows each output length in advance, which a real server does not).
"""

import argparse
import time

import numpy as np

from bench.common import save_result
from bench.workload import random_requests
from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.scheduler import Scheduler
from engine.sequence import SamplingParams


def run(engine: LLMEngine, mode: str, reqs) -> dict:
    engine.cfg.kv_allocation = mode
    engine.block_manager = BlockManager(engine.num_blocks, engine.cfg.block_size, mode, engine.cfg.max_model_len)
    engine.scheduler = Scheduler(engine.cfg, engine.block_manager, engine.model_cfg.eos_token_ids)
    seqs = [engine.add_request(p, SamplingParams(max_tokens=n, ignore_eos=True)) for p, n in reqs]
    running, used_blocks, durations = [], [], []
    t0 = time.perf_counter()
    while engine.has_unfinished():
        t = time.perf_counter()
        engine.step()
        durations.append(time.perf_counter() - t)
        running.append(len(engine.scheduler.running))
        used_blocks.append(engine.block_manager.num_usable_blocks - engine.block_manager.num_free_blocks)
    elapsed = time.perf_counter() - t0
    # tokens actually stored vs blocks held, at the step with peak concurrency
    out_tokens = sum(len(s.output_token_ids) for s in seqs)
    return {"mode": mode, "max_concurrent": int(max(running)), "mean_concurrent": float(np.average(running, weights=durations)),  # time-weighted
            "p50_concurrent": float(np.median(running)), "peak_blocks_used": int(max(used_blocks)),
            "preemptions": engine.scheduler.num_preemptions, "elapsed_s": elapsed,
            "output_tok_per_s": out_tokens / elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-budget-gb", type=float, default=4.0)
    ap.add_argument("--num-prompts", type=int, default=2000)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--range-ratio", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    block_bytes = 2 * 16 * 16 * 8 * 64 * 2  # K+V x layers x block_size x kv_heads x head_dim x bf16
    num_blocks = int(args.kv_budget_gb * 2**30 // block_bytes) + 1
    engine = LLMEngine(EngineConfig(num_kv_blocks=num_blocks, max_num_seqs=4096, max_model_len=args.max_model_len))
    reqs = random_requests(args.num_prompts, args.input_len, args.output_len, range_ratio=args.range_ratio)
    run(engine, "paged", random_requests(64, args.input_len, 16, seed=1))  # warmup
    results = {}
    for mode in ("prealloc", "prealloc_oracle", "paged"):
        results[mode] = run(engine, mode, reqs)
        print(results[mode])
    ratio = results["paged"]["mean_concurrent"] / results["prealloc"]["mean_concurrent"]
    print(f"paged fits {ratio:.2f}x more concurrent requests on average "
          f"({results['paged']['max_concurrent']} vs {results['prealloc']['max_concurrent']} at peak)")
    save_result("memory", {"workload": vars(args), "num_kv_blocks": num_blocks, "kv_budget_bytes": num_blocks * block_bytes,
                           "results": results, "mean_concurrency_ratio": ratio,
                           "mean_concurrency_ratio_vs_oracle":
                               results["paged"]["mean_concurrent"] / results["prealloc_oracle"]["mean_concurrent"],
                           "peak_concurrency_ratio": results["paged"]["max_concurrent"] / results["prealloc"]["max_concurrent"]})


if __name__ == "__main__":
    main()
