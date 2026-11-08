"""Complete UAIA-Nano Transformer model with Dual Language and Decision Heads.

Enhanced with Jev-style typed decision readouts (Choice, Score, Noul):
- Choice: Categorical action selection logits over discrete options
- Score: Raw sigmoid output in [0, 1], requiring task-specific calibration
- Noul: Binary permission / termination gate in [0, 1] (e.g. human approval required)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.layers import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


class UAIATransformerBlock(nn.Module):
    """A single UAIA-Nano residual block with Pre-RMSNorm, GQLA, and SwiGLU."""

    def __init__(self, config: UAIANanoConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-Norm Attention Residual
        normed_attn_in = self.attn_norm(x)
        attn_out, new_kv = self.attn(
            normed_attn_in, cos=cos, sin=sin, mask=mask, kv_cache=kv_cache
        )
        x = x + attn_out

        # Pre-Norm SwiGLU Residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class StructuredDecisionHead(nn.Module):
    """Dedicated Decision Engine head producing Jev-style typed readouts (Choice, Score, Noul)."""

    def __init__(self, config: UAIANanoConfig):
        super().__init__()
        # Choice: Categorical action selection
        self.action_proj = nn.Linear(config.d_model, config.n_actions, bias=False)
        # Score: raw task signal; calibration requires training and evaluation.
        self.confidence_proj = nn.Linear(config.d_model, 1, bias=True)
        # Noul: Binary permission / risk gating (requires human approval)
        self.permission_proj = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [batch_size, d_model]
        action_logits = self.action_proj(h)
        confidence = torch.sigmoid(self.confidence_proj(h))
        permission_required = torch.sigmoid(self.permission_proj(h))
        return action_logits, confidence, permission_required


class HariNanoModel(nn.Module):
    """HariNano: Custom Independent Foundation Model engineered by K R Hari Prajwal."""

    def __init__(self, config: Optional[HariNanoConfig] = None):
        super().__init__()
        self.config = config or HariNanoConfig()

        # Input token embeddings
        self.token_embeddings = nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.rope_theta,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [UAIATransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        # Final RMS Normalization
        self.final_norm = RMSNorm(self.config.d_model, eps=self.config.norm_eps)

        # Decision Head (Choice, Score, Noul)
        self.decision_head = StructuredDecisionHead(self.config)

        # Weight tying or separate head
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.config.d_model, self.config.vocab_size, bias=False
            )
        else:
            self.lm_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights from Gaussian distribution for from-scratch training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, non_embedding: bool = False) -> int:
        """Returns total trainable parameter count."""
        if non_embedding:
            return sum(
                p.numel()
                for n, p in self.named_parameters()
                if "token_embeddings" not in n and p.requires_grad
            )
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_size_mb(self, precision: str = "int4") -> float:
        """Calculates expected disk footprint in megabytes at given precision."""
        params = self.count_parameters(non_embedding=False)
        bits_per_param = {
            "fp32": 32.0,
            "fp16": 16.0,
            "bf16": 16.0,
            "int8": 8.0,
            "int4": 4.5,
            "bitnet": 1.58,
        }.get(precision.lower(), 4.5)
        bytes_total = (params * bits_per_param) / 8.0
        return bytes_total / (1024 * 1024)

    def get_lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Computes next token logits, reusing tied embeddings if configured."""
        if self.config.tie_word_embeddings:
            return F.linear(h, self.token_embeddings.weight)
        return self.lm_head(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        action_targets: Optional[torch.Tensor] = None,
        confidence_targets: Optional[torch.Tensor] = None,
        permission_targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
    ) -> Dict[str, Any]:
        bs, seq_len = input_ids.shape
        h = self.token_embeddings(input_ids)

        # Compute RoPE cos/sin
        cos, sin = self.rotary_emb(h, start_pos + seq_len)
        cos = cos[start_pos : start_pos + seq_len]
        sin = sin[start_pos : start_pos + seq_len]

        # Build causal mask if seq_len > 1
        mask = None
        if seq_len > 1:
            # Cached keys precede this chunk and must remain visible to every
            # new token; only future positions within the new chunk are masked.
            past_len = kv_caches[0][0].shape[2] if kv_caches is not None else 0
            mask = torch.full(
                (seq_len, past_len + seq_len), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=past_len + 1)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cached_kv = kv_caches[i] if kv_caches is not None else None
            h, updated_kv = block(
                h, cos=cos, sin=sin, mask=mask, kv_cache=cached_kv
            )
            new_kv_caches.append(updated_kv)

        h = self.final_norm(h)

        # 1. Language Head Logits
        lm_logits = self.get_lm_logits(h)

        # 2. Decision Head (Choice, Score, Noul evaluated on last token)
        last_hidden = h[:, -1, :]
        action_logits, confidence, permission_required = self.decision_head(last_hidden)

        # Loss calculation
        total_loss = None
        lm_loss = None
        action_loss = None
        conf_loss = None
        perm_loss = None

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = lm_loss

        if action_targets is not None:
            action_loss = F.cross_entropy(action_logits, action_targets)
            total_loss = (
                total_loss + 0.5 * action_loss if total_loss is not None else action_loss
            )

        if confidence_targets is not None:
            conf_loss = F.binary_cross_entropy(
                confidence.squeeze(-1), confidence_targets.float()
            )
            total_loss = (
                total_loss + 0.1 * conf_loss if total_loss is not None else conf_loss
            )

        if permission_targets is not None:
            perm_loss = F.binary_cross_entropy(
                permission_required.squeeze(-1), permission_targets.float()
            )
            total_loss = (
                total_loss + 0.1 * perm_loss if total_loss is not None else perm_loss
            )

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "action_loss": action_loss,
            "confidence_loss": conf_loss,
            "permission_loss": perm_loss,
            "logits": lm_logits,
            "action_logits": action_logits,
            "confidence": confidence,
            "permission_required": permission_required,
            "kv_caches": new_kv_caches,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        self.eval()
        bs, seq_len = prompt_ids.shape
        cur_tokens = prompt_ids
        kv_caches = None

        out = self.forward(cur_tokens, kv_caches=None, start_pos=0)
        kv_caches = out["kv_caches"]
        next_token_logits = out["logits"][:, -1, :]

        generated = prompt_ids

        for step in range(max_new_tokens):
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            out = self.forward(
                next_token,
                kv_caches=kv_caches,
                start_pos=seq_len + step,
            )
            kv_caches = out["kv_caches"]
            next_token_logits = out["logits"][:, -1, :]

        return generated


# Backwards compatibility alias
HariTransformerBlock = UAIATransformerBlock
UAIANanoModel = HariNanoModel
