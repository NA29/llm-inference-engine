"""torch.profiler traces of decode steps, eager vs Triton + CUDA graphs.

Writes results/traces/decode_{eager,graphs}.json.gz; open them in https://ui.perfetto.dev.
The eager trace shows hundreds of short kernels separated by launch gaps; with graphs
the same step is one graph launch and the GPU stays busy.
"""

import gzip
import os
import shutil
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench.common import RESULTS_DIR, save_result
from bench.workload import random_requests
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams


def trace(name: str, backend: str, graphs: bool, batch: int = 8) -> dict:
    engine = LLMEngine(EngineConfig(attn_backend=backend, use_cuda_graphs=graphs, num_kv_blocks=2048,
                                    max_num_seqs=batch, cuda_graph_batch_sizes=[batch]))
    for prompt, _ in random_requests(batch, 512, 64):
        engine.add_request(prompt, SamplingParams(max_tokens=64, ignore_eos=True))
    for _ in range(10):  # prefill + warm decode steps
        engine.step()
    out_dir = RESULTS_DIR / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"decode_{name}.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            engine.step()
    prof.export_chrome_trace(str(path))
    with open(path, "rb") as f, gzip.open(f"{path}.gz", "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(path)
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    del engine
    torch.cuda.empty_cache()
    return {"trace": f"traces/{Path(path).name}.gz", "gpu_kernel_events_in_5_steps": len(kernels)}


if __name__ == "__main__":
    res = {"eager": trace("eager", "torch", False), "graphs": trace("graphs", "triton", True)}
    print(res)
    save_result("profile_decode", res)
