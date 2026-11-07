"""Configuration for the HariNano Foundation Model."""

from dataclasses import dataclass


@dataclass
class HariNanoConfig:
    """Model hyperparameters for HariNano.

    Target parameter budget: ~46.4M - 68.5M parameters
    Target quantized file size: <= 45 MB - 50 MB
    """

    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 2  # Grouped-Query Attention (ratio 4:1)
    d_ff: int = 1376  # SwiGLU intermediate dimension
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    n_actions: int = 32  # Structured decision engine action space
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# Backwards compatibility alias
UAIANanoConfig = HariNanoConfig

