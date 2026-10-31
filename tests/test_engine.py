"""Continuous batching must not change results: batched output == one-at-a-time output."""

import random

import pytest
import torch

from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams
from tests.conftest import requires_cuda


def random_prompts(n: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[128000] + [rng.randrange(1000, 100000) for _ in range(rng.randrange(4, 200))] for _ in range(n)]


@pytest.fixture(scope="module")
def engine():
    return LLMEngine(EngineConfig(dtype=torch.float32, attn_backend="torch", use_cuda_graphs=False,
                                  num_kv_blocks=2048, max_model_len=1024))


def reset(engine: LLMEngine, **overrides):
    for k, v in overrides.items():
        setattr(engine.cfg, k, v)


@requires_cuda
def test_batched_equals_sequential(engine):
    prompts = random_prompts(24)
    params = SamplingParams(max_tokens=24, ignore_eos=True)
    batched = engine.generate(prompts, params)
    sequential = [engine.generate([p], params)[0] for p in prompts[:6]]
    assert batched[:6] == sequential
    assert all(len(o) == 24 for o in batched)


@requires_cuda
def test_preemption_recompute_gives_same_output(engine):
    prompts = random_prompts(16, seed=1)
    params = SamplingParams(max_tokens=48, ignore_eos=True)
    expected = engine.generate(prompts, params)
    # Shrink the pool so sequences must be preempted and recomputed mid-generation.
    engine.block_manager.free_blocks = type(engine.block_manager.free_blocks)(range(1, 60))
    watermark, engine.block_manager.watermark_blocks = engine.block_manager.watermark_blocks, 0
    before = engine.scheduler.num_preemptions
    got = engine.generate(prompts, params)
    assert engine.scheduler.num_preemptions > before
    assert got == expected
    engine.block_manager.free_blocks = type(engine.block_manager.free_blocks)(range(1, engine.num_blocks))
    engine.block_manager.watermark_blocks = watermark

