"""Greedy-output parity with Hugging Face Transformers on N real prompts.

Our engine runs all prompts at once through the production path (continuous batching,
paged KV cache, Triton kernel, CUDA graphs). HF runs each prompt alone with
`generate(do_sample=False)`. A prompt matches if every generated token is identical.

fp32 is the exactness check: in bf16 two different (both correct) kernels round
differently, and a near-tie between the top two logits can flip, so for bf16 we also
report where outputs first diverge and how close the top-2 HF logits were there.
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.common import save_result
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams


def load_prompts(tok, n: int) -> list[list[int]]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        text = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
        ids = tok(text).input_ids
        if len(ids) <= 512:
            prompts.append(ids)
        if len(prompts) == n:
            break
    return prompts


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    engine = LLMEngine(EngineConfig(dtype=dtype, max_model_len=1024, gpu_memory_utilization=0.6))
    tok = AutoTokenizer.from_pretrained(engine.model_dir)
    prompts = load_prompts(tok, args.num_prompts)
    params = SamplingParams(max_tokens=args.max_tokens)
    t0 = time.perf_counter()
    ours = engine.generate(prompts, params)
    engine_s = time.perf_counter() - t0
    eos = engine.model_cfg.eos_token_ids

    hf = AutoModelForCausalLM.from_pretrained(engine.model_dir, torch_dtype=dtype).cuda().eval()
    matches, mismatches = 0, []
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        x = torch.tensor([ids], device="cuda")
        res = hf.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_tokens, do_sample=False,
                          eos_token_id=eos, pad_token_id=eos[0], output_scores=True, return_dict_in_generate=True,
                          temperature=None, top_p=None)
        ref = res.sequences[0, len(ids):].tolist()
        if ours[i] == ref:
            matches += 1
        else:
            j = next((k for k, (a, b) in enumerate(zip(ours[i], ref)) if a != b), min(len(ours[i]), len(ref)))
            margin = None
            if j < len(res.scores):
                top2 = res.scores[j][0].float().topk(2).values
                margin = float(top2[0] - top2[1])
            mismatches.append({"prompt_index": i, "first_divergence": j, "hf_top2_logit_margin": margin,
                               "ours": tok.decode(ours[i]), "hf": tok.decode(ref)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(prompts)} prompts checked, {matches} exact matches")
    hf_s = time.perf_counter() - t0

    # A divergence where HF's own top-2 logits are within `tie` is a numerical tie: either
    # token is a correct greedy choice, and which one wins depends on summation order.
    tie = 1e-4 if dtype == torch.float32 else 0.25
    ties = sum(1 for m in mismatches if m["hf_top2_logit_margin"] is not None and m["hf_top2_logit_margin"] < tie)
    print(f"{args.dtype}: {matches}/{len(prompts)} prompts match HF exactly; "
          f"{ties} of the {len(mismatches)} divergences are at a near-tie (top-2 margin < {tie})")
    save_result(f"hf_parity_{args.dtype}", {
        "num_prompts": len(prompts), "max_new_tokens": args.max_tokens, "dtype": args.dtype,
        "exact_matches": matches, "tie_threshold": tie, "divergences_at_near_tie": ties,
        "matches_or_near_tie": matches + ties, "engine_seconds": engine_s, "hf_seconds": hf_s,
        "engine_config": "triton paged attention + CUDA graphs + continuous batching (all prompts in flight)",
        "mismatches": mismatches,
    })


if __name__ == "__main__":
    main()
