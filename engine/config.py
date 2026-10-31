import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: list[int]

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ModelConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        eos = c.get("eos_token_id", [])
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            rope_scaling=c.get("rope_scaling"),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            eos_token_ids=eos if isinstance(eos, list) else [eos],
        )


@dataclass
class EngineConfig:
    model: str = os.environ.get("MODEL_ID", "unsloth/Llama-3.2-1B")
    dtype: torch.dtype = torch.bfloat16
    block_size: int = 16                 # tokens per KV block
    max_num_seqs: int = 256              # max sequences in one decode batch
    max_num_batched_tokens: int = 8192   # max prompt tokens in one prefill step
    max_model_len: int = 2048            # max prompt + output length of one request
    gpu_memory_utilization: float = 0.90
    num_kv_blocks: int | None = None     # fixed KV pool size; None = size it from free memory
    attn_backend: str = "torch"         # "triton" (paged kernel) or "torch" (eager gather)
    use_cuda_graphs: bool = False
    scheduling: str = "continuous"       # "continuous" or "static"
    kv_allocation: str = "paged"         # "paged" (on demand) or "prealloc" (max_model_len per request)
    cuda_graph_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 16)))
