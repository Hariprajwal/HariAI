"""Neural network building blocks for UAIA-Nano in PyTorch."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hari_ai.model.config import UAIANanoConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scaling."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for relative sequence position encoding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""
    # xq: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim] -> unsqueeze to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads along the head dimension for Grouped-Query Attention."""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Latent Attention (GQLA) with KV Cache support."""

    def __init__(self, config: UAIANanoConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bs, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Handle KV Cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        current_kv = (k, v)

        # Expand KV heads to match query heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)

        # Reshape back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output), current_kv


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network: (Swish(x @ W_gate) * (x @ W_up)) @ W_down."""

    def __init__(self, config: UAIANanoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

