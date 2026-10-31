import pytest
import torch

from engine.attention import paged_decode_attention_torch
from tests.conftest import requires_cuda


def make_inputs(batch, ctx_lens, dtype, n_heads=32, n_kv=8, d=64, block_size=16, num_blocks=1024, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, n_heads, d, device="cuda", dtype=dtype, generator=g)
    k_cache = torch.randn(num_blocks, block_size, n_kv, d, device="cuda", dtype=dtype, generator=g)
    v_cache = torch.randn(num_blocks, block_size, n_kv, d, device="cuda", dtype=dtype, generator=g)
    max_blocks = max(-(-c // block_size) for c in ctx_lens)
    # scattered, non-contiguous physical blocks, like a real fragmented pool
    perm = torch.randperm(num_blocks - 1, device="cuda", generator=g)[: batch * max_blocks] + 1
    block_tables = perm.view(batch, max_blocks).to(torch.int32)
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    return q, k_cache, v_cache, block_tables, context_lens


@requires_cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("ctx_lens", [[1], [16], [17, 5, 300], [1000, 1, 33, 64, 128, 7, 512, 2000]])
def test_triton_matches_reference(dtype, tol, ctx_lens):
    from engine.triton_attention import paged_decode_attention
    args = make_inputs(len(ctx_lens), ctx_lens, dtype)
    scale = 64 ** -0.5
    ref = paged_decode_attention_torch(*args, scale)
    for tile in (16, 64):
        for num_splits in (None, 1, 4):
            out = paged_decode_attention(*args, scale, tile=tile, num_splits=num_splits)
            torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
