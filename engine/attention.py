"""Attention over the paged KV cache.

A step is either a prefill step (whole prompts) or a decode step (one new token per
sequence). Tokens of all sequences in a step are flattened into one [T, ...] batch.

KV cache layout per layer: [num_blocks, block_size, num_kv_heads, head_dim]. A
sequence's tokens live in the blocks listed in its block table, so token i of a
sequence is at slot block_table[i // block_size] * block_size + i % block_size.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    is_prefill: bool
    slot_mapping: torch.Tensor            # [T] int64: cache slot each new token's K/V is written to
    # prefill
    max_seqlen: int = 0
    num_seqs: int = 0
    pad_index: torch.Tensor | None = None  # [T] int64: position of each token in a [B * max_seqlen] padded batch
    # decode
    block_tables: torch.Tensor | None = None  # [B, max_blocks_per_seq] int32
    context_lens: torch.Tensor | None = None  # [B] int32, including the token being decoded
    backend: str = "triton"


def store_kv(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    n_kv, d = k.shape[1], k.shape[2]
    k_cache.view(-1, n_kv, d).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, n_kv, d).index_copy_(0, slot_mapping, v)


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, meta: AttentionMetadata,
                      scale: float) -> torch.Tensor:
    """Causal attention over each prompt's own tokens.

    Prompts are scattered into a right-padded [B, L] batch so one SDPA (FlashAttention)
    call covers the whole step. Right padding is safe under a causal mask: real tokens
    never attend to the padding that follows them.
    """
    t, n_heads, d = q.shape
    n_kv = k.shape[1]
    b, length = meta.num_seqs, meta.max_seqlen

    def pad(x: torch.Tensor, h: int) -> torch.Tensor:
        out = x.new_zeros(b * length, h, d)
        out[meta.pad_index] = x
        return out.view(b, length, h, d).transpose(1, 2)

    qp, kp, vp = pad(q, n_heads), pad(k, n_kv), pad(v, n_kv)
    group = n_heads // n_kv
    kp, vp = kp.repeat_interleave(group, dim=1), vp.repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True, scale=scale)
    return out.transpose(1, 2).reshape(b * length, n_heads, d)[meta.pad_index]


def paged_decode_attention_torch(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                 block_tables: torch.Tensor, context_lens: torch.Tensor,
                                 scale: float) -> torch.Tensor:
    """Eager PyTorch reference: gather each sequence's blocks into a padded tensor.

    q: [B, n_heads, d] -> [B, n_heads, d]. Materializes [B, max_ctx, n_kv, d] copies of
    K and V, which is what the Triton kernel avoids.
    """
    b, n_heads, d = q.shape
    block_size, n_kv = k_cache.shape[1], k_cache.shape[2]
    group = n_heads // n_kv
    k = k_cache[block_tables.long()].reshape(b, -1, n_kv, d)  # [B, max_ctx, n_kv, d]
    v = v_cache[block_tables.long()].reshape(b, -1, n_kv, d)
    qg = q.view(b, n_kv, group, d)
    scores = torch.einsum("bkgd,btkd->bkgt", qg.float(), k.float()) * scale
    mask = torch.arange(k.shape[1], device=q.device)[None, :] >= context_lens[:, None]
    scores.masked_fill_(mask[:, None, None, :], float("-inf"))
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bkgt,btkd->bkgd", probs, v.float())
    return out.reshape(b, n_heads, d).to(q.dtype)


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     meta: AttentionMetadata, scale: float) -> torch.Tensor:
    return paged_decode_attention_torch(q, k_cache, v_cache, meta.block_tables, meta.context_lens, scale)
