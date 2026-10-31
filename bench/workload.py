"""Synthetic workloads shared by our benchmarks and the vLLM baseline (same seed -> same requests)."""

import random


def random_requests(n: int, input_len: int, output_len: int, seed: int = 0,
                    range_ratio: float = 0.0) -> list[tuple[list[int], int]]:
    """n (prompt_token_ids, max_tokens) pairs. Lengths are uniform in [len*(1-r), len*(1+r)]."""
    rng = random.Random(seed)
    reqs = []

    def length(mean: int) -> int:
        return rng.randint(max(int(mean * (1 - range_ratio)), 1), max(int(mean * (1 + range_ratio)), 1))

    for _ in range(n):
        n_in, n_out = length(input_len), length(output_len)
        prompt = [128000] + [rng.randrange(1000, 120000) for _ in range(n_in - 1)]
        reqs.append((prompt, n_out))
    return reqs
