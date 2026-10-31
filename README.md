# llm-inference-engine

An LLM inference engine written from scratch in Python, PyTorch and Triton. It serves
**Llama-3.2-1B** (bf16) on one **NVIDIA A10G** using continuous batching, a paged KV cache,
a Triton paged-attention kernel and CUDA-graph decode.

About 950 lines of engine code (`engine/`), plus benchmarks (`bench/`) and tests (`tests/`).

## Results

All numbers below were measured on an NVIDIA A10G (24 GB, Modal cloud) with Llama-3.2-1B in bf16.
The raw data for each number is in [`results/`](results/) (see [Proof of runs](#proof-of-runs)).

| | Result | Data |
|---|---|---|
| Output throughput | **5,019 tok/s**, **99.6% of vLLM 0.10.1.1** (5,039 tok/s) on the same benchmark | [ours](results/throughput_ours.json), [vLLM](results/throughput_vllm.json) |
| Request rate under a 500 ms p99 TTFT SLO | continuous batching: **23 req/s**; static batching misses the SLO at **every** tested rate from 0.125 to 16 req/s (p99 TTFT ≥ 858 ms) | [continuous](results/serving_continuous.json), [static](results/serving_static.json) |
| Concurrent requests in the same 4 GB of KV cache | paged: **212** vs per-request preallocation: **63** (**3.4×**; 3.8× at peak) | [memory](results/memory.json) |
| Per-token (decode step) latency | Triton kernel + CUDA graphs vs eager PyTorch: **2.5× faster at batch 1**, 2.3×–4.6× across batch 1–128 | [latency](results/latency.json) |
| Correctness vs Hugging Face Transformers | greedy outputs identical on **999/1,000 prompts** (fp32, 64 new tokens each); the remaining prompt diverges at a numerical tie (HF's own top-2 logits differ by 1.3e-5) | [fp32](results/hf_parity_float32.json), [bf16](results/hf_parity_bfloat16.json) |

Static batching never meets a 500 ms p99 TTFT: a request that arrives while a batch is running
waits for that whole batch (up to 192 decode steps) before its prefill. The gap narrows as the SLO
loosens:

| p99 TTFT SLO | continuous batching | static batching |
|---|---|---|
| 500 ms | 23 req/s | none of the tested rates (≥ 0.125 req/s) |
| 1 s | ≥ 24 req/s | 0.25 req/s |
| 2 s | ≥ 24 req/s | 4 req/s |
| 5 s | ≥ 24 req/s | 16 req/s (highest rate tested) |

Above 23 req/s, continuous batching was measured at 24 req/s (p99 TTFT 647 ms) and at 32 req/s,
where it is overloaded (p99 TTFT 13 s).

<p>
<img src="results/plots/throughput.png" width="49%">
<img src="results/plots/serving_slo.png" width="49%">
<img src="results/plots/memory.png" width="49%">
<img src="results/plots/latency.png" width="49%">
</p>

## Architecture

```
add_request ──► Scheduler.waiting ──┐
                                    ▼
             ┌──────────── Scheduler.schedule() — every step ────────────┐
             │ 1. admit waiting requests while KV blocks are free         │
             │    (BlockManager.allocate)              -> prefill step    │
             │ 2. otherwise grow every running sequence by one token      │
             │    (BlockManager.append_slot; preempt newest if out)       │
             │                                         -> decode step     │
             └────────────────────────────┬────────────────────────────────┘
                                          ▼
     LLMEngine._run: flatten tokens, build slot mapping / block tables
       prefill: Llama forward, SDPA over padded prompts, K/V written to paged cache
       decode : CUDA graph replay (bucketed batch) -> Triton paged attention
                                          ▼
     sample (greedy / temperature / top-p) -> Scheduler.postprocess:
       append token, retire finished sequences, free their blocks
```

| File | What it does |
|---|---|
| [`engine/model.py`](engine/model.py) | Llama 3 from scratch: RMSNorm, RoPE with Llama-3 frequency scaling, GQA, SwiGLU, tied embeddings; loads HF safetensors into fused QKV / gate-up projections |
| [`engine/block_manager.py`](engine/block_manager.py) | Paged KV allocator: 16-token blocks, free pool, per-sequence block tables. Also the two preallocation baselines |
| [`engine/scheduler.py`](engine/scheduler.py) | Continuous (iteration-level) batching with admission control and recompute preemption. Also the static-batching baseline |
| [`engine/attention.py`](engine/attention.py) | Writes K/V into the paged cache, prefill attention, and the eager PyTorch paged-attention reference |
| [`engine/triton_attention.py`](engine/triton_attention.py) | Triton paged decode-attention kernel: GQA-grouped tensor-core tiles, online softmax, split-K |
| [`engine/cuda_graph.py`](engine/cuda_graph.py) | Captures the decode forward pass as one CUDA graph per batch-size bucket |
| [`engine/engine.py`](engine/engine.py) | Step loop, input preparation, KV pool sizing |

### Design notes

**Paged KV cache.** Each layer's cache is `[num_blocks, 16, 8 kv_heads, 64]`. A sequence holds a
block table: logical block *i* maps to a physical block anywhere in the pool. Blocks are taken one
at a time as the sequence grows and returned the moment it finishes, so memory is never reserved
for tokens that might be generated later. A KV token costs 32 KB in bf16
(2 × 16 layers × 8 heads × 64 dims × 2 bytes), so a 16-token block is 512 KB. Block 0 is a reserved
scratch block that padded rows of a CUDA-graph batch write into.

**Scheduling.** Each step is either a prefill step or a decode step. Waiting requests are admitted
when the pool has room for their prompt, with a 1% watermark kept free for running sequences to
grow, so new requests join a running batch without waiting for it to finish. When the pool runs dry
during decode, the most recently admitted sequence is preempted: its blocks are freed and it is
later recomputed from its tokens. Tests check that preemption does not change the output.

**Triton kernel.** One program handles one (sequence, KV head, context split). The 4 query heads
that share a KV head (GQA) are computed together, padded to 16 rows, so QKᵀ and PV run on tensor
cores (`tl.dot`). Each K/V tile is read from HBM once per group instead of once per head. Every
token's physical address comes from the block table, and an online softmax (running max and
running sum) avoids materializing the score row. At small batch sizes, (batch × KV heads) programs
leave SMs idle, so the context is split across programs (FlashDecoding-style split-K) and merged by
a second kernel. With the default 64-token tile, at batch ≥ 32 the kernel reads K/V at
415–435 GB/s, against the A10G's 600 GB/s peak. At batch 256 (512-token contexts) one attention
call takes 0.65 ms, vs 6.58 ms for the eager gather version
([microbenchmark](results/kernel_microbench.json)).

**CUDA graphs.** An eager decode step of this model launches about 950 small GPU kernels
(4,750 in 5 steps in the profiler trace). At small batch sizes, the GPU finishes them faster than
Python can launch them. Decode is captured once per batch bucket on persistent input buffers:
1, 2, 4, 8, then every 16 up to 256. A batch is padded to the next bucket. The profiler traces
are in `results/traces/` (open them in [Perfetto](https://ui.perfetto.dev)). At batch 8, five
decode steps take 134 ms of CPU time eager vs 40 ms with Triton + graphs, where one graph replay
replaces the launches. GPU time also drops, from 57.9 to 38.3 ms, because the Triton kernel
replaces the gather-based attention.

**Correctness.** `tests/test_model.py` checks the model's prefill logits against HF (within 1e-3 in fp32)
and its single-sequence greedy output token by token. The parity benchmark runs 1,000
Alpaca prompts through the full production path at once: continuous batching, paged cache, Triton
kernel and CUDA graphs. It compares every generated token with HF `generate(do_sample=False)` run
one prompt at a time. The parity check is done in fp32 because in bf16 two correct kernels round
differently (different summation orders), so near-ties between the top two logits can flip.
In bf16, 599/1,000 prompts are identical, and 396 of the 401 divergences happen where HF's own
top-2 logits are within 0.25 (one or two bf16 rounding steps).

## Benchmark methodology

- **vs vLLM** ([`bench/throughput.py`](bench/throughput.py), [`bench/vllm_throughput.py`](bench/vllm_throughput.py)):
  - workload: 1,000 requests with 512 random prompt tokens and 256 output tokens each
  - identical settings for both engines: same token IDs, greedy, `ignore_eos`, bf16,
    `max_num_seqs=256`, `max_model_len=2048`, 90% GPU memory
  - vLLM: prefix caching off, detokenization off
  - both on an A10G, with one warmup batch first
- **SLO sweep** ([`bench/serving.py`](bench/serving.py)): open-loop Poisson arrivals.
  - Workload: prompts of 512 ± 50% tokens, outputs of 128 ± 50% tokens, at least 200 requests per rate.
  - TTFT runs from arrival to first token and includes queueing.
  - Continuous batching: the rate doubles until p99 TTFT breaks 500 ms, then a bisection narrows it down.
  - Static batching: same engine, kernels and cache. A batch is the set of requests waiting when the previous batch finished, and it runs to completion. Swept on a fixed rate grid.
- **Memory** ([`bench/memory.py`](bench/memory.py)): one engine with a fixed 4 GB KV pool, 2,000 requests submitted at once.
  - Baseline: preallocation reserves `max_model_len` (2,048) tokens per request, like a contiguous per-request KV buffer.
  - Oracle variant: reserves exactly prompt + max_tokens; paged still fits 1.1× more than it, and real servers don't know output lengths in advance.
  - Concurrency is the time-weighted mean of running requests.
- **Latency** ([`bench/latency.py`](bench/latency.py)): median decode-step wall time (including sampling and host overhead) with all sequences of the batch decoding. Prompts are 512 tokens.
  - eager: PyTorch attention that gathers blocks, no graphs
  - triton: the Triton kernel, no graphs
  - triton+graphs: the Triton kernel with CUDA graphs

## Proof of runs

Every result file embeds the environment it was produced on:
- the full `nvidia-smi` output, with GPU name, UUID and driver;
- CUDA, torch, Triton, transformers and vLLM versions;
- the git commit the code ran from;
- the Modal task ID.

The full stdout of each run is in `results/logs/` with the same header. Modal's "A10G" pool also
contains A10s (72 SMs, 150 W). [`modal_app.py`](modal_app.py) refuses to run on anything but an
A10G, so every number comes from the same GPU model.

## Reproduce

```bash
pip install modal && modal setup
modal run modal_app.py --cmd "pytest tests -v"                     # unit + GPU tests
modal run modal_app.py --cmd bench.throughput
modal run modal_app.py --cmd bench.vllm_throughput --vllm
modal run modal_app.py --cmd "bench.serving --scheduling continuous"
modal run modal_app.py --cmd "bench.serving --scheduling static --rates 0.125,0.25,0.5,1,2,4,8,16"
modal run modal_app.py --cmd bench.memory
modal run modal_app.py --cmd bench.latency
modal run modal_app.py --cmd "bench.hf_parity --dtype float32"
modal run modal_app.py --cmd bench.profile_decode
python -m bench.plots                                              # results/*.json -> results/plots/
```

## Limitations and next steps

- Prefill attention pads prompts to the longest in the step and uses PyTorch SDPA. A varlen
  FlashAttention kernel (or chunked prefill) would remove the padding.
- Only pure decode steps are graphed. Prefill and decode are never mixed in one step.
- No prefix caching, speculative decoding, quantization or tensor parallelism.
- Scheduling and input preparation are synchronous Python. vLLM overlaps them with GPU execution.
