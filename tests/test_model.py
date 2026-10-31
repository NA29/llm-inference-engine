"""Milestone 0: one sequence through the from-scratch model matches Hugging Face."""

import pytest
import torch

from engine.attention import AttentionMetadata
from engine.model import load_model
from tests.conftest import requires_cuda

PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1969, the first person to walk on the moon"]


def run_greedy(model, ids: list[int], n_new: int, backend: str) -> tuple[list[int], torch.Tensor]:
    """Greedy decode one sequence over a trivially paged cache (blocks 0, 1, 2, ...)."""
    cfg = model.cfg
    block_size, num_blocks = 16, 64
    for layer in model.layers:
        shape = (num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        layer.self_attn.k_cache = torch.zeros(shape, dtype=model.embed_tokens.weight.dtype, device="cuda")
        layer.self_attn.v_cache = torch.zeros_like(layer.self_attn.k_cache)
    n = len(ids)
    meta = AttentionMetadata(is_prefill=True, slot_mapping=torch.arange(n, device="cuda"), max_seqlen=n,
                             num_seqs=1, pad_index=torch.arange(n, device="cuda"))
    h = model(torch.tensor(ids, device="cuda"), torch.arange(n, device="cuda"), meta)
    prefill_logits = model.compute_logits(h).float()
    out = [int(prefill_logits[-1].argmax())]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None]
    for _ in range(n_new - 1):
        pos = n + len(out) - 1
        meta = AttentionMetadata(is_prefill=False, slot_mapping=torch.tensor([pos], device="cuda"),
                                 block_tables=block_table,
                                 context_lens=torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
                                 backend=backend)
        h = model(torch.tensor([out[-1]], device="cuda"), torch.tensor([pos], device="cuda"), meta)
        out.append(int(model.compute_logits(h)[-1].argmax()))
    return out, prefill_logits


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_matches_hf(model_dir, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).cuda().eval()
    ours = load_model(model_dir, dtype, "cuda", max_positions=4096)
    for prompt in PROMPTS:
        ids = tok(prompt).input_ids
        hf_logits = hf(torch.tensor([ids], device="cuda")).logits[0].float()
        hf_out = hf.generate(torch.tensor([ids], device="cuda"), max_new_tokens=32, do_sample=False,
                             attention_mask=torch.ones(1, len(ids), device="cuda"), pad_token_id=tok.eos_token_id)
        hf_out = hf_out[0, len(ids):].tolist()
        out, logits = run_greedy(ours, ids, 32, backend="torch")
        err = (logits - hf_logits).abs().max().item()
        print(f"{dtype} {prompt!r}: max |logit diff| = {err:.4f}, tokens equal = {out == hf_out}")
        assert err < (1e-3 if dtype == torch.float32 else 0.5)
        if dtype == torch.float32:
            assert out == hf_out, (tok.decode(out), tok.decode(hf_out))
        else:
            # bf16 kernels accumulate in different orders, so near-ties can flip later on;
            # only the first token is required to agree (see tests/test_hf_parity.py).
            assert out[0] == hf_out[0]
