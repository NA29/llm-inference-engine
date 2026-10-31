"""Paged KV cache allocator: fixed-size blocks, a free pool, one block table per sequence.

Same idea as virtual memory: a sequence's logical KV blocks map to physical blocks
anywhere in the pool, so memory is claimed one block at a time as the sequence grows
and returned the moment it finishes. Nothing is reserved for tokens that are never
generated, and there is no external fragmentation.

`mode="prealloc"` is the baseline the paged design replaces: each request reserves
room for max_model_len tokens up front, like a contiguous per-request KV buffer.
`mode="prealloc_oracle"` is a generous variant that reserves exactly prompt + max_tokens.
"""

from collections import deque

from engine.sequence import Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, mode: str = "paged", max_model_len: int = 2048,
                 watermark: float = 0.01):
        assert mode in ("paged", "prealloc", "prealloc_oracle")
        self.block_size = block_size
        self.mode = mode
        self.max_model_len = max_model_len
        # Block 0 is a scratch block: padded rows of a CUDA-graph batch write their K/V there.
        self.free_blocks: deque[int] = deque(range(1, num_blocks))
        self.num_usable_blocks = num_blocks - 1
        # Admission keeps a few blocks spare so running sequences can grow without
        # immediately preempting each other (same 1% default as vLLM).
        self.watermark_blocks = int(watermark * num_blocks) if mode == "paged" else 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _blocks_for(self, seq: Sequence) -> int:
        if self.mode == "prealloc":
            return cdiv(self.max_model_len, self.block_size)
        if self.mode == "prealloc_oracle":
            return cdiv(len(seq.prompt_token_ids) + seq.params.max_tokens, self.block_size)
        return cdiv(seq.num_tokens, self.block_size)

    def can_allocate(self, seq: Sequence) -> bool:
        return self.num_free_blocks - self._blocks_for(seq) >= self.watermark_blocks

    def allocate(self, seq: Sequence) -> None:
        """Give a sequence the blocks for all its current tokens (called at prefill)."""
        assert not seq.block_table
        n = self._blocks_for(seq)
        assert self.num_free_blocks >= n, "allocate() without can_allocate()"
        seq.block_table = [self.free_blocks.popleft() for _ in range(n)]

    def can_append(self, seq: Sequence) -> bool:
        """Whether the token about to be decoded (position num_tokens - 1) has a slot."""
        needs_block = seq.num_tokens > len(seq.block_table) * self.block_size
        return not needs_block or self.num_free_blocks > 0

    def append_slot(self, seq: Sequence) -> None:
        if seq.num_tokens > len(seq.block_table) * self.block_size:
            seq.block_table.append(self.free_blocks.popleft())

    def free(self, seq: Sequence) -> None:
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []
