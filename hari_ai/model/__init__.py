"""Neural network architecture package for HariAI."""

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.layers import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_emb,
)
from hari_ai.model.transformer import (
    StructuredDecisionHead,
    UAIANanoModel,
    UAIATransformerBlock,
)

__all__ = [
    "UAIANanoConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "apply_rotary_emb",
    "StructuredDecisionHead",
    "UAIATransformerBlock",
    "UAIANanoModel",
]

