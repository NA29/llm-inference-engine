"""vLLM baseline for bench/throughput.py: same GPU, model, dtype, requests and limits.

Run with: modal run modal_app.py --vllm --cmd "bench.vllm_throughput"
"""

import argparse
import os
import time

from bench.common import MODEL_ID, save_result
from bench.workload import random_requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=os.environ.get("MODEL_ID", MODEL_ID), dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.90, max_num_seqs=args.max_num_seqs, enable_prefix_caching=False,
              seed=0)

    def run(reqs):
        prompts = [TokensPrompt(prompt_token_ids=p) for p, _ in reqs]
        params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, detokenize=False) for _, n in reqs]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        in_tokens = sum(len(p) for p, _ in reqs)
        return {"elapsed_s": elapsed, "output_tokens": out_tokens, "input_tokens": in_tokens,
                "output_tok_per_s": out_tokens / elapsed, "total_tok_per_s": (in_tokens + out_tokens) / elapsed,
                "requests_per_s": len(reqs) / elapsed}

    run(random_requests(32, args.input_len, 16, seed=1))  # warmup
    res = run(random_requests(args.num_prompts, args.input_len, args.output_len))
    print(f"vllm: {res['output_tok_per_s']:.0f} output tok/s ({res['elapsed_s']:.1f}s)")
    save_result(f"throughput_vllm{args.tag}", {"system": "vllm", "workload": vars(args), **res})


if __name__ == "__main__":
    main()
