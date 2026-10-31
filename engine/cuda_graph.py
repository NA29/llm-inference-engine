"""CUDA Graphs for decode steps.

A 1B model's decode step is hundreds of small kernels (16 layers x ~15 ops), and at
small batch sizes the GPU finishes each one faster than Python can launch the next.
Recording the whole decode forward once as a CUDA graph and replaying it turns those
launches into a single one.

A graph replays fixed kernels on fixed memory addresses, so inputs live in persistent
buffers that are overwritten before every replay, and one graph is captured per
batch-size bucket. A batch is padded up to the next bucket; padded rows write to the
scratch KV block 0 and are discarded.
"""

import bisect

import numpy as np
import torch

from engine.attention import AttentionMetadata


class CUDAGraphRunner:
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model
        self.buckets = sorted(b for b in engine.cfg.cuda_graph_batch_sizes if b <= engine.cfg.max_num_seqs)
        max_bs = self.buckets[-1]
        self.input_ids = torch.zeros(max_bs, dtype=torch.long, device="cuda")
        self.positions = torch.zeros(max_bs, dtype=torch.long, device="cuda")
        self.slot_mapping = torch.zeros(max_bs, dtype=torch.long, device="cuda")
        self.context_lens = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        self.block_tables = torch.zeros(max_bs, engine.max_blocks_per_seq, dtype=torch.int32, device="cuda")
        # one logits buffer shared by every bucket's graph (a [bs, vocab] output per graph adds up)
        self.logits_buf = torch.empty(max_bs, engine.model_cfg.vocab_size, dtype=engine.cfg.dtype, device="cuda")
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, torch.Tensor] = {}
        self._capture_all()

    def _forward(self, bs: int) -> torch.Tensor:
        meta = AttentionMetadata(is_prefill=False, slot_mapping=self.slot_mapping[:bs],
                                 block_tables=self.block_tables[:bs], context_lens=self.context_lens[:bs],
                                 backend=self.engine.cfg.attn_backend)
        hidden = self.model(self.input_ids[:bs], self.positions[:bs], meta)
        return self.model.compute_logits(hidden, out=self.logits_buf[:bs])

    @torch.inference_mode()
    def _capture_all(self) -> None:
        pool = None
        for bs in reversed(self.buckets):  # largest first, so smaller graphs reuse its memory pool
            self._forward(bs)              # warmup: compiles Triton kernels, picks cuBLAS algorithms
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.logits[bs] = self._forward(bs)
            pool = graph.pool()
            self.graphs[bs] = graph
        torch.cuda.synchronize()

    def can_run(self, batch_size: int) -> bool:
        return batch_size <= self.buckets[-1]

    def run(self, seqs) -> torch.Tensor:
        n = len(seqs)
        bs = self.buckets[bisect.bisect_left(self.buckets, n)]
        arrays = self.engine.prepare_decode(seqs, pad_to=bs)
        for buf, arr in zip((self.input_ids, self.positions, self.slot_mapping, self.context_lens,
                             self.block_tables), arrays):
            buf[:bs].copy_(torch.from_numpy(np.ascontiguousarray(arr)), non_blocking=True)
        self.graphs[bs].replay()
        return self.logits[bs][:n]
