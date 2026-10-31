"""Iteration-level (continuous) batching.

Every engine step the scheduler builds a fresh batch: newly arrived requests are admitted
as soon as KV blocks are free (a prefill step), otherwise all running sequences advance
by one token (a decode step). Finished sequences leave immediately and free their blocks,
so new requests never wait for the slowest member of a batch.

`scheduling="static"` is the baseline: a batch is admitted only when the previous one
has completely finished.
"""

import time
from collections import deque

from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.sequence import Sequence, SeqStatus


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager, eos_token_ids: list[int]):
        self.cfg = cfg
        self.bm = block_manager
        self.eos = set(eos_token_ids)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0
        self._static_quota = 0  # static mode: requests still allowed into the current batch

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (sequences to run this step, is_prefill)."""
        # 1. Admit waiting requests (prefill) while sequence slots, token budget and KV blocks allow.
        static = self.cfg.scheduling == "static"
        if static and not self.running:
            # A static batch is the requests waiting when the previous batch finished; it may
            # take several prefill steps to admit them all (token budget per step).
            self._static_quota = min(len(self.waiting), self.cfg.max_num_seqs)
        batch, max_len = [], 0
        while self.waiting and len(self.running) + len(batch) < self.cfg.max_num_seqs:
            if static and self._static_quota == 0:
                break
            seq = self.waiting[0]
            # Prefill attention runs on a [batch, longest prompt] padded tensor, so the
            # token budget is charged for the padded size.
            if (len(batch) + 1) * max(max_len, seq.num_tokens) > self.cfg.max_num_batched_tokens:
                break
            if not self.bm.can_allocate(seq):
                break
            self.bm.allocate(seq)
            seq.status = SeqStatus.RUNNING
            self.waiting.popleft()
            batch.append(seq)
            max_len = max(max_len, seq.num_tokens)
            self._static_quota -= static
        if batch:
            self.running.extend(batch)
            return batch, True

        # 2. Decode every running sequence. If the pool runs dry, preempt the most recently
        #    admitted sequence (free its blocks, recompute it later from its tokens).
        self._static_quota = 0
        scheduled = []
        while self.running:
            seq = self.running.popleft()
            while not self.bm.can_append(seq):
                if self.running:
                    self._preempt(self.running.pop())
                else:
                    self._preempt(seq)
                    break
            else:
                self.bm.append_slot(seq)
                scheduled.append(seq)
        self.running.extend(scheduled)
        return scheduled, False

    def _preempt(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.bm.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        """Append each sequence's new token; retire finished ones. Returns the finished sequences."""
        now = time.perf_counter()
        finished = []
        for seq, tok in zip(seqs, token_ids):
            seq.output_token_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            hit_eos = not seq.params.ignore_eos and tok in self.eos
            if hit_eos or len(seq.output_token_ids) >= seq.params.max_tokens:
                seq.status = SeqStatus.FINISHED
                seq.finish_time = now
                self.bm.free(seq)
                self.running.remove(seq)
                finished.append(seq)
        return finished
