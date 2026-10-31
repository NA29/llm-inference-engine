"""Per-token decode latency (time per decode step) at fixed batch sizes.

eager:         PyTorch attention that gathers each sequence's KV blocks, every kernel launched from Python
triton:        Triton paged-attention kernel, still launched kernel by kernel
triton+graphs: Triton kernel with the whole decode step replayed as one CUDA graph
"""

import argparse
import time

import numpy as np
import torch

from bench.common import save_result
from bench.workload import random_requests
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams

CONFIGS = {"eager": ("torch", False), "triton": ("triton", False), "triton+graphs": ("triton", True)}


def decode_step_ms(engine: LLMEngine, batch: int, input_len: int, output_len: int) -> float:
    for prompt, n in random_requests(batch, input_len, output_len, seed=batch):
        engine.add_request(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
    times = []
    while engine.has_unfinished():
        seqs, is_prefill = engine.scheduler.schedule()
        torch.cuda.synchronize()
        t = time.perf_counter()
        tokens = engine._run(seqs, is_prefill)  # .tolist() inside waits for the GPU
        dt = time.perf_counter() - t
        engine.scheduler.postprocess(seqs, tokens)
        if not is_prefill and len(seqs) == batch:
            times.append(dt)
    return float(np.median(times) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,8,32,64,128")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    args = ap.parse_args()
    batches = [int(b) for b in args.batch_sizes.split(",")]

    table = {}
    for name, (backend, graphs) in CONFIGS.items():
        engine = LLMEngine(EngineConfig(attn_backend=backend, use_cuda_graphs=graphs, num_kv_blocks=8192,
                                        max_num_seqs=max(batches), cuda_graph_batch_sizes=batches))
        decode_step_ms(engine, 4, 64, 8)  # warmup
        table[name] = {b: decode_step_ms(engine, b, args.input_len, args.output_len) for b in batches}
        print(name, {b: f"{ms:.2f} ms" for b, ms in table[name].items()})
        del engine
        torch.cuda.empty_cache()

    speedup = {b: table["eager"][b] / table["triton+graphs"][b] for b in batches}
    print("speedup eager -> triton+graphs:", {b: f"{s:.2f}x" for b, s in speedup.items()})
    save_result("latency", {"workload": vars(args), "decode_step_ms": table,
                            "speedup_vs_eager": speedup})


if __name__ == "__main__":
    main()
