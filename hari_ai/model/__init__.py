"""Neural network architecture package for HariAI."""

from hari_ai.model.config import HariNanoConfig, UAIANanoConfig
from hari_ai.model.layers import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_emb,
)
from hari_ai.model.transformer import (
    HariNanoModel,
    StructuredDecisionHead,
    UAIANanoModel,
    UAIATransformerBlock,
)

__all__ = [
    "HariNanoConfig",
    "UAIANanoConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "apply_rotary_emb",
    "StructuredDecisionHead",
    "UAIATransformerBlock",
    "HariNanoModel",
    "UAIANanoModel",
]

