"""Triton paged-attention kernel for decode (one query token per sequence).

One program handles one (sequence, KV head, context split). It computes attention for
all GROUP query heads that share the KV head (GQA), so each K/V tile is read from HBM
once per group rather than once per query head. The group is padded to 16 rows so the
QK^T and PV products run on tensor cores via tl.dot.

The program walks its part of the context in tiles of TILE tokens, finding each token's
physical block through the block table, with an online softmax (running max m, running
sum l) so the full score row is never materialized.

Split-K (FlashDecoding): with few sequences, (batch x KV heads) programs cannot fill the
GPU's SMs, so each sequence's context is also split into NUM_SPLITS partitions processed
in parallel. Each split writes its partial (acc, m, l) and a second kernel merges them.
"""

import torch
import triton
import triton.language as tl

_NUM_SMS: int | None = None


@triton.jit
def _paged_decode_kernel(
    Q, K, V, O, M_partial, L_partial, block_tables, context_lens, scale, partition,
    stride_qb, stride_qh,
    stride_kblk, stride_kslot, stride_kh,
    stride_bt,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr, NUM_SPLITS: tl.constexpr, IEEE: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    ctx_len = tl.load(context_lens + seq)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, TILE)
    head = kv_head * GROUP + offs_g
    row_valid = offs_g < GROUP

    q = tl.load(Q + seq * stride_qb + head[:, None] * stride_qh + offs_d[None, :],
                mask=row_valid[:, None], other=0.0)                  # [GROUP_PAD, HEAD_DIM]

    m = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)        # running max
    l = tl.zeros([GROUP_PAD], dtype=tl.float32)                      # running sum of exp
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)          # running weighted sum of V

    start = split * partition
    end = tl.minimum(start + partition, ctx_len)
    for t0 in range(start, end, TILE):
        tok = t0 + offs_t
        valid = tok < end
        # logical token -> physical block (block table) -> slot inside that block
        block = tl.load(block_tables + seq * stride_bt + tok // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        kv_off = block * stride_kblk + (tok % BLOCK_SIZE) * stride_kslot + kv_head * stride_kh
        k = tl.load(K + kv_off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0)  # [TILE, HEAD_DIM]
        v = tl.load(V + kv_off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0)

        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))                               # [GROUP_PAD, TILE], fp32 accumulate
        s = tl.where(valid[None, :], s * scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)                                    # rescale old stats to the new max
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    if NUM_SPLITS == 1:
        out = acc / l[:, None]
        tl.store(O + seq * stride_qb + head[:, None] * stride_qh + offs_d[None, :],
                 out.to(O.dtype.element_ty), mask=row_valid[:, None])
    else:
        # unnormalized partial result; [B, n_heads, NUM_SPLITS, HEAD_DIM] fp32 scratch
        part = (seq * tl.num_programs(1) * GROUP + head) * NUM_SPLITS + split
        tl.store(O + part[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=row_valid[:, None])
        tl.store(M_partial + part, m, mask=row_valid)
        tl.store(L_partial + part, l, mask=row_valid)


@triton.jit
def _merge_splits_kernel(O, O_partial, M_partial, L_partial, stride_ob, stride_oh,
                         N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)
    base = (seq * N_HEADS + h) * NUM_SPLITS
    m = tl.load(M_partial + base + offs_s)                           # empty splits have m = -inf
    l = tl.load(L_partial + base + offs_s)
    acc = tl.load(O_partial + (base + offs_s)[:, None] * HEAD_DIM + offs_d[None, :])
    m_max = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp(m - m_max), 0.0)
    out = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    tl.store(O + seq * stride_ob + h * stride_oh + offs_d, out.to(O.dtype.element_ty))


def _num_splits(batch: int, n_kv: int, max_ctx: int, partition: int) -> int:
    """Split the context only when (batch x KV heads) programs leave SMs idle."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    if batch * n_kv >= 2 * _NUM_SMS:
        return 1
    return triton.next_power_of_2(max(1, triton.cdiv(max_ctx, partition)))


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float,
                           tile: int = 64, partition: int = 256, num_splits: int | None = None) -> torch.Tensor:
    """q: [B, n_heads, head_dim]; caches: [num_blocks, block_size, n_kv_heads, head_dim].

    The split count depends only on static shapes (batch size and block-table width), so a
    captured CUDA graph always replays the same launch configuration.
    """
    b, n_heads, d = q.shape
    block_size, n_kv = k_cache.shape[1], k_cache.shape[2]
    group = n_heads // n_kv
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and k_cache.stride() == v_cache.stride()
    max_ctx = block_tables.shape[1] * block_size
    if num_splits is None:
        num_splits = _num_splits(b, n_kv, max_ctx, partition)
    if num_splits > 1:
        partition = triton.cdiv(triton.cdiv(max_ctx, num_splits), tile) * tile
    out = torch.empty_like(q)
    if num_splits == 1:
        o_partial, m_partial, l_partial = out, out, out  # unused
    else:
        o_partial = torch.empty(b, n_heads, num_splits, d, dtype=torch.float32, device=q.device)
        m_partial = torch.empty(b, n_heads, num_splits, dtype=torch.float32, device=q.device)
        l_partial = torch.empty_like(m_partial)
    _paged_decode_kernel[(b, n_kv, num_splits)](
        q, k_cache, v_cache, o_partial, m_partial, l_partial, block_tables, context_lens, scale,
        partition if num_splits > 1 else max_ctx,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        BLOCK_SIZE=block_size, HEAD_DIM=d, GROUP=group, GROUP_PAD=max(16, triton.next_power_of_2(group)),
        TILE=tile, NUM_SPLITS=num_splits,
        IEEE=q.dtype == torch.float32, num_warps=4, num_stages=1,
    )
    if num_splits > 1:
        _merge_splits_kernel[(b, n_heads)](out, o_partial, m_partial, l_partial, out.stride(0), out.stride(1),
                                           N_HEADS=n_heads, HEAD_DIM=d, NUM_SPLITS=num_splits)
    return out
