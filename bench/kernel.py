"""Microbenchmark: Triton paged decode attention vs the eager PyTorch gather version."""

import argparse

import torch

from bench.common import save_result
from engine.attention import paged_decode_attention_torch
from engine.triton_attention import paged_decode_attention
from tests.test_paged_attention import make_inputs


def time_ms(fn, iters=100) -> float:
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()
    rows = []
    for batch in (1, 8, 32, 64, 128, 256):
        inputs = make_inputs(batch, [args.ctx] * batch, torch.bfloat16, num_blocks=batch * (args.ctx // 16 + 1) + 1)
        scale = 64 ** -0.5
        row = {"batch": batch, "ctx": args.ctx,
               "torch_ms": time_ms(lambda: paged_decode_attention_torch(*inputs, scale))}
        for tile in (32, 64, 128):
            row[f"triton_tile{tile}_ms"] = time_ms(lambda: paged_decode_attention(*inputs, scale, tile=tile))
        kv_bytes = 2 * batch * args.ctx * 8 * 64 * 2
        best = min(v for k, v in row.items() if k.startswith("triton"))
        row["triton_best_GBps"] = kv_bytes / best / 1e6
        print(row)
        rows.append(row)
    save_result("kernel_microbench", {"rows": rows})
