"""LLMEngine: owns the model, the paged KV pool, the scheduler, and the step loop."""

import time

import numpy as np
import torch

from engine.attention import AttentionMetadata
from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.model import download_model, load_model
from engine.sampling import sample
from engine.scheduler import Scheduler
from engine.sequence import SamplingParams, Sequence


def _to_gpu(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).pin_memory().cuda(non_blocking=True)


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model_dir = download_model(cfg.model)
        self.model = load_model(self.model_dir, cfg.dtype, "cuda", max_positions=cfg.max_model_len)
        self.model_cfg = self.model.cfg
        assert cfg.max_model_len <= cfg.max_num_batched_tokens, "a full-length prompt must fit one prefill step"
        self.max_blocks_per_seq = -(-cfg.max_model_len // cfg.block_size)
        self.num_blocks = cfg.num_kv_blocks or self._profile_num_blocks()
        assert self.num_blocks - 1 >= self.max_blocks_per_seq, "KV pool cannot hold one max_model_len sequence"
        self._allocate_kv_cache(self.num_blocks)
        self.block_manager = BlockManager(self.num_blocks, cfg.block_size, cfg.kv_allocation, cfg.max_model_len)
        self.scheduler = Scheduler(cfg, self.block_manager, self.model_cfg.eos_token_ids)
        self.graph_runner = None
        if cfg.use_cuda_graphs:
            from engine.cuda_graph import CUDAGraphRunner
            self.graph_runner = CUDAGraphRunner(self)

    # ---- memory ----------------------------------------------------------------------

    @property
    def block_bytes(self) -> int:
        c = self.model_cfg
        return 2 * c.num_layers * self.cfg.block_size * c.num_kv_heads * c.head_dim * self.cfg.dtype.itemsize

    @torch.inference_mode()
    def _profile_num_blocks(self) -> int:
        """Run the largest possible prefill once, then give the KV pool whatever memory is left.

        Also reserves room for what is allocated later (CUDA-graph logits buffer, sampling
        temporaries) and for memory PyTorch does not track (CUDA context, cuBLAS workspaces).
        """
        self._allocate_kv_cache(1)
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(self.cfg.max_model_len, self.cfg.max_num_batched_tokens)
        seqs = self.cfg.max_num_batched_tokens // seq_len
        n = seqs * seq_len  # the largest padded prefill the scheduler can admit
        meta = AttentionMetadata(
            is_prefill=True, slot_mapping=torch.zeros(n, dtype=torch.long, device="cuda"),
            max_seqlen=seq_len, num_seqs=seqs, pad_index=torch.arange(n, device="cuda"))
        positions = torch.arange(n, device="cuda") % seq_len
        hidden = self.model(torch.zeros(n, dtype=torch.long, device="cuda"), positions, meta)
        self.model.compute_logits(hidden[:seqs])
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        untracked = (total - free) - torch.cuda.memory_reserved()
        vocab = self.model_cfg.vocab_size
        later = self.cfg.max_num_seqs * vocab * (self.cfg.dtype.itemsize + 3 * 4)  # graph logits + fp32 sampling
        free_for_kv = total * self.cfg.gpu_memory_utilization - peak - untracked - later
        return int(free_for_kv // self.block_bytes)

    def _allocate_kv_cache(self, num_blocks: int) -> None:
        c = self.model_cfg
        for layer in self.model.layers:
            layer.self_attn.k_cache = layer.self_attn.v_cache = None
        torch.cuda.empty_cache()
        self.kv_cache = torch.zeros(2, c.num_layers, num_blocks, self.cfg.block_size, c.num_kv_heads,
                                    c.head_dim, dtype=self.cfg.dtype, device="cuda")
        for i, layer in enumerate(self.model.layers):
            layer.self_attn.k_cache, layer.self_attn.v_cache = self.kv_cache[0, i], self.kv_cache[1, i]

    # ---- requests ----------------------------------------------------------------------

    def add_request(self, prompt_token_ids: list[int], params: SamplingParams,
                    arrival_time: float | None = None) -> Sequence:
        assert len(prompt_token_ids) + params.max_tokens <= self.cfg.max_model_len, "request exceeds max_model_len"
        seq = Sequence(prompt_token_ids, params, arrival_time or time.perf_counter())
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[Sequence]:
        """Run one batch (prefill or decode). Returns sequences that finished this step."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        token_ids = self._run(seqs, is_prefill)
        return self.scheduler.postprocess(seqs, token_ids)

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[list[int]]:
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.has_unfinished():
            self.step()
        return [s.output_token_ids for s in seqs]

    # ---- model execution ---------------------------------------------------------------

    def _slot(self, seq: Sequence, pos: np.ndarray) -> np.ndarray:
        table = np.asarray(seq.block_table, dtype=np.int64)
        return table[pos // self.cfg.block_size] * self.cfg.block_size + pos % self.cfg.block_size

    @torch.inference_mode()
    def _run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            logits = self._run_prefill(seqs)
        else:
            logits = self._run_decode(seqs)
        temps = torch.tensor([s.params.temperature for s in seqs], device="cuda")
        top_ps = torch.tensor([s.params.top_p for s in seqs], device="cuda")
        return sample(logits, temps, top_ps).tolist()

    def _run_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        lens = np.array([s.num_tokens for s in seqs])
        max_len = int(lens.max())
        input_ids = np.concatenate([np.asarray(s.token_ids, dtype=np.int64) for s in seqs])
        positions = np.concatenate([np.arange(n) for n in lens])
        slots = np.concatenate([self._slot(s, np.arange(n)) for s, n in zip(seqs, lens)])
        pad_index = np.concatenate([i * max_len + np.arange(n) for i, n in enumerate(lens)])
        meta = AttentionMetadata(is_prefill=True, slot_mapping=_to_gpu(slots), max_seqlen=max_len,
                                 num_seqs=len(seqs), pad_index=_to_gpu(pad_index))
        hidden = self.model(_to_gpu(input_ids), _to_gpu(positions), meta)
        last = _to_gpu(np.cumsum(lens) - 1)
        return self.model.compute_logits(hidden[last])

    def prepare_decode(self, seqs: list[Sequence], pad_to: int | None = None):
        """Host-side inputs of a decode step, padded to `pad_to` rows (CUDA-graph bucket)."""
        b = pad_to or len(seqs)
        bs = self.cfg.block_size
        input_ids = np.zeros(b, dtype=np.int64)
        positions = np.zeros(b, dtype=np.int64)
        slots = np.zeros(b, dtype=np.int64)        # padded rows write to scratch block 0
        context_lens = np.ones(b, dtype=np.int32)  # and attend to one scratch token
        width = self.max_blocks_per_seq if pad_to else max(len(s.block_table) for s in seqs)
        block_tables = np.zeros((b, width), dtype=np.int32)
        for i, s in enumerate(seqs):
            pos = s.num_tokens - 1
            input_ids[i] = s.last_token_id
            positions[i] = pos
            slots[i] = s.block_table[pos // bs] * bs + pos % bs
            context_lens[i] = s.num_tokens
            block_tables[i, :len(s.block_table)] = s.block_table
        return input_ids, positions, slots, context_lens, block_tables

    def _run_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        if self.graph_runner is not None and self.graph_runner.can_run(len(seqs)):
            return self.graph_runner.run(seqs)
        input_ids, positions, slots, context_lens, block_tables = self.prepare_decode(seqs)
        meta = AttentionMetadata(is_prefill=False, slot_mapping=_to_gpu(slots),
                                 block_tables=_to_gpu(block_tables), context_lens=_to_gpu(context_lens),
                                 backend=self.cfg.attn_backend)
        hidden = self.model(_to_gpu(input_ids), _to_gpu(positions), meta)
        return self.model.compute_logits(hidden)
