"""Parameter and idealized storage accounting for the existing HariNano model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBudget:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_actions: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.n_layers, self.n_heads,
               self.n_kv_heads, self.d_ff, self.n_actions) <= 0:
            raise ValueError('All architecture dimensions must be positive.')
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError('Attention dimensions and grouped-query heads must divide evenly.')
        if (self.d_model // self.n_heads) % 2:
            raise ValueError('RoPE requires an even attention head dimension.')

    def breakdown(self) -> dict[str, int]:
        width = self.d_model
        kv_width = self.n_kv_heads * (width // self.n_heads)
        parts = {
            'token_embeddings': self.vocab_size * width,
            'attention': self.n_layers * (2*width*width + 2*width*kv_width),
            'feed_forward': self.n_layers * 3*width*self.d_ff,
            'layer_norms': self.n_layers * 2*width + width,
            'decision_head': width*(self.n_actions+2)+2,
            'untied_output_head': 0 if self.tie_word_embeddings else self.vocab_size*width,
        }
        return {**parts, 'total': sum(parts.values())}

    def ideal_weight_bytes(self, bits_per_weight: float) -> int:
        """Lower bound only: excludes quantization scales, tokenizer and format metadata."""
        if not 0 < bits_per_weight <= 32:
            raise ValueError('Bits per weight must be in (0, 32].')
        import math
        return math.ceil(self.breakdown()['total']*bits_per_weight/8)
