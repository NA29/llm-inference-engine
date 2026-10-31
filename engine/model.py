"""Llama 3 decoder, written from scratch to run over a flattened, paged batch.

Matches Hugging Face's LlamaForCausalLM numerics: RMSNorm in fp32, rotate_half RoPE
with Llama 3 frequency scaling, SwiGLU MLP, grouped-query attention, tied embeddings.
"""

import glob
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from engine.attention import AttentionMetadata, decode_attention, prefill_attention, store_kv
from engine.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def llama3_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    d = cfg.head_dim
    arange = torch.arange(0, d, 2, dtype=torch.int64, device="cpu")  # computed on CPU, like HF
    inv_freq = 1.0 / (cfg.rope_theta ** (arange.float() / d))
    s = cfg.rope_scaling
    if not s or s.get("rope_type", s.get("type")) != "llama3":
        return inv_freq
    factor, low, high = s["factor"], s["low_freq_factor"], s["high_freq_factor"]
    old_len = s["original_max_position_embeddings"]
    low_freq_wavelen, high_freq_wavelen = old_len / low, old_len / high
    wavelen = 2 * math.pi / inv_freq
    # Long wavelengths (low frequencies) are slowed down by `factor`, short ones kept,
    # and the band in between is interpolated smoothly.
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_len / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding:
    """cos/sin tables [max_positions, head_dim], kept in fp32 (plain tensors, not module buffers,
    so model.to(bf16) does not round them)."""

    def __init__(self, cfg: ModelConfig, max_positions: int, device: str):
        inv_freq = llama3_inv_freq(cfg).to(device)
        freqs = torch.outer(torch.arange(max_positions, device=device).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def __call__(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        cos = self.cos[positions].to(q.dtype)[:, None, :]  # [T, 1, head_dim]
        sin = self.sin[positions].to(q.dtype)[:, None, :]
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.scale = self.d ** -0.5
        self.qkv_proj = nn.Linear(cfg.hidden_size, (self.n_heads + 2 * self.n_kv) * self.d, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d, cfg.hidden_size, bias=False)
        self.k_cache: torch.Tensor | None = None  # set by the engine once the KV pool is allocated
        self.v_cache: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, positions: torch.Tensor, rope: RotaryEmbedding,
                meta: AttentionMetadata) -> torch.Tensor:
        t = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.n_heads * self.d, self.n_kv * self.d, self.n_kv * self.d], -1)
        q, k, v = q.view(t, self.n_heads, self.d), k.view(t, self.n_kv, self.d), v.view(t, self.n_kv, self.d)
        q, k = rope(positions, q, k)
        store_kv(k, v, self.k_cache, self.v_cache, meta.slot_mapping)
        if meta.is_prefill:
            o = prefill_attention(q, k, v, meta, self.scale)
        else:
            o = decode_attention(q, self.k_cache, self.v_cache, meta, self.scale)
        return self.o_proj(o.reshape(t, self.n_heads * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, positions, rope, meta):
        x = x + self.self_attn(self.input_layernorm(x), positions, rope, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int, device: str):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(cfg, max_positions, device)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata) -> torch.Tensor:
        """input_ids, positions: [T] -> final hidden states [T, hidden]."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, self.rope, meta)
        return self.norm(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        w = self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        return F.linear(hidden, w)

    @torch.no_grad()
    def load_weights(self, model_dir: str) -> None:
        params = dict(self.named_parameters())
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            for name, w in load_file(path).items():
                name = name.removeprefix("model.")
                if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                    self._load_fused(params, name, w, "self_attn", ["q_proj", "k_proj", "v_proj"], "qkv_proj")
                elif name.endswith(("gate_proj.weight", "up_proj.weight")):
                    self._load_fused(params, name, w, "mlp", ["gate_proj", "up_proj"], "gate_up_proj")
                elif name in params:
                    params[name].copy_(w)
                elif name != "lm_head.weight":
                    raise KeyError(f"unexpected weight {name}")

    def _load_fused(self, params, name, w, module, parts, fused):
        """Copy q/k/v (or gate/up) weights into their row range of the fused projection."""
        prefix, part = name.rsplit(".", 2)[0], name.rsplit(".", 2)[1]
        prefix = prefix.removesuffix(f".{module}")
        dst = params[f"{prefix}.{module}.{fused}.weight"]
        sizes = {"q_proj": self.cfg.num_heads * self.cfg.head_dim,
                 "k_proj": self.cfg.num_kv_heads * self.cfg.head_dim,
                 "v_proj": self.cfg.num_kv_heads * self.cfg.head_dim,
                 "gate_proj": self.cfg.intermediate_size, "up_proj": self.cfg.intermediate_size}
        start = sum(sizes[p] for p in parts[:parts.index(part)])
        dst[start:start + sizes[part]].copy_(w)


def load_model(model_dir: str, dtype: torch.dtype, device: str, max_positions: int) -> LlamaModel:
    cfg = ModelConfig.from_pretrained(model_dir)
    with torch.device(device):
        model = LlamaModel(cfg, max_positions, device)
    model.to(dtype)
    model.load_weights(model_dir)
    return model.eval()


def download_model(model_id: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
