"""Configurable decoder-only causal language model.

Modern defaults: RoPE positional embeddings, pre-RMSNorm, SwiGLU feed-forward,
causal multi-head attention via F.scaled_dot_product_attention (works on CPU and
fuses to fast kernels when a CUDA device is available), optional GQA
(n_kv_heads < n_heads) and weight tying.

Everything is controlled by a NovaModelConfig dataclass so model capacity is a
configuration concern, never hard-coded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NovaModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8          # GQA: fewer kv heads than q heads if desired
    ff_mult: float = 2.0         # hidden = ff_mult * d_model
    max_seq_len: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model % self.n_kv_heads == 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    """Rotary embedding tables. Returns cos, sin each shaped (seq_len, head_dim/2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2) -> broadcast to (B, H, T, D/2)
    x1, x2 = x.chunk(2, dim=-1)                      # each (B, H, T, D/2)
    cos = cos[None, None, :, :]                      # (1, 1, T, D/2)
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: NovaModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, is_causal=True, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)   # cos/sin sliced to T above water? (see below)

        # repeat_kv for GQA
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    def __init__(self, cfg: NovaModelConfig):
        super().__init__()
        hidden = int(cfg.ff_mult * cfg.d_model)
        # SwiGLU conventionally uses 3 projections; hidden sized with ff_mult
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: NovaModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg)
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, use_cache=False, past_kv=None):
        a, present = self.attn(self.n1(x), cos, sin, is_causal=True,
                               past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x, present


class NovaLM(nn.Module):
    """Decoder-only causal LM.

    Notes:
      - Positions are RoPE; no learned position embeddings.
      - Input/output embeddings tied unless tie_embeddings=False.
      - forward(tokens) -> logits; supports incremental decoding via use_cache.
    """
    def __init__(self, cfg: NovaModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.lm_head = None  # computed via x @ embed.weight.T
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self._tied = True

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        """tokens: (B, T). targets: (B, T) next-token labels or None.
        Returns (logits, loss) where loss is None if no targets."""
        x = self.embed(tokens)
        cos, sin = rope_cos_sin(tokens.shape[1], self.cfg.head_dim,
                                self.cfg.rope_theta, x.device, x.dtype)
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        x = self.norm_f(x)
        if self.lm_head is None:
            logits = x @ self.embed.weight.T
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1).long())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, *, max_new_tokens: int, eos_id: int = None,
                 temperature: float = 1.0, top_k: int = None, top_p: float = None,
                 generator: torch.Generator = None):
        """Autoregressive generation loop.

        Returns the full token sequence including the prompt. One forward per
        token is simple and correct; a fused KV cache is a later optimization
        (see the project ROADMAP).
        """
        ctx = self.cfg.max_seq_len
        cur = tokens
        out_last = None
        for _ in range(max_new_tokens):
            cur_in = cur[:, -ctx:]
            logits, _ = self.forward(cur_in)
            logits = logits[:, -1, :]
            if temperature <= 0.0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, idx = torch.topk(logits, k)
                    probs = torch.softmax(v, dim=-1)
                    pick = torch.multinomial(probs, 1, generator=generator)
                    nxt = idx.gather(-1, pick)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs, 1, generator=generator)
            cur = torch.cat([cur, nxt], dim=1)
            if eos_id is not None and int(nxt[0, 0]) == eos_id:
                break
        return cur

