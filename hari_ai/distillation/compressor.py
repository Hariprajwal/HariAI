"""Layer Pruning and Structured Model Compression for HariNano (<45 MB Target).

Implements:
- Structural Layer Pruning: Drops redundant transformer blocks to reduce parameter depth.
- Low-Rank & INT4 Symmetric Quantization: Compresses weights to 0.5 bytes/param.
- Strict <= 45.0 MB Hard Disk Budget Assertion: Guarantees deployment on resource-constrained devices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from hari_ai.model.config import HariNanoConfig, UAIANanoConfig
from hari_ai.model.transformer import HariNanoModel, UAIANanoModel
from hari_ai.quantization.quantizer import quantize_model, save_quantized_checkpoint


@dataclass
class CompressionReport:
    original_layers: int
    pruned_layers: int
    original_params: int
    pruned_params: int
    compressed_file_bytes: int
    compressed_file_mb: float
    under_45mb_target: bool
    checkpoint_path: str


def prune_transformer_layers(
    model: nn.Module, target_layers: int = 14
) -> nn.Module:
    """Structurally prunes transformer blocks down to target layer depth."""
    if hasattr(model, "blocks"):
        layers = model.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unsupported model structure for layer pruning.")

    total_layers = len(layers)
    if target_layers >= total_layers:
        return model

    # Select evenly distributed layers across the depth of the network
    indices_to_keep = sorted(
        [int(i * (total_layers / target_layers)) for i in range(target_layers)]
    )

    if hasattr(model, "blocks"):
        model.blocks = nn.ModuleList([layers[i] for i in indices_to_keep])
        model.config.n_layers = len(model.blocks)
    elif hasattr(model, "model"):
        model.model.layers = nn.ModuleList([layers[i] for i in indices_to_keep])
        model.config.num_hidden_layers = len(model.model.layers)

    return model


def compress_to_target_budget(
    model: nn.Module,
    output_path: str,
    target_mb: float = 45.0,
    target_layers: int = 14,
    precision: str = "int4",
) -> CompressionReport:
    """Applies structural layer pruning and INT4 quantization, verifying size <= target_mb."""
    # Count original parameters
    orig_params = sum(p.numel() for p in model.parameters())
    orig_layers = (
        len(model.blocks) if hasattr(model, "blocks") else len(model.model.layers)
    )

    # 1. Structural Layer Pruning
    pruned_model = prune_transformer_layers(model, target_layers=target_layers)
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    pruned_layers = (
        len(pruned_model.blocks)
        if hasattr(pruned_model, "blocks")
        else len(pruned_model.model.layers)
    )

    # 2. INT4 Quantization
    quantized = quantize_model(pruned_model, precision=precision)

    # 3. Serialize and Assert Budget Constraint
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_info = save_quantized_checkpoint(quantized, output_path)

    file_bytes = save_info["bytes"]
    file_mb = save_info["size_mb"]

    assert file_mb <= target_mb, (
        f"Compression failure: Exported checkpoint is {file_mb:.2f} MB, "
        f"which exceeds the {target_mb:.2f} MB hard constraint!"
    )

    return CompressionReport(
        original_layers=orig_layers,
        pruned_layers=pruned_layers,
        original_params=orig_params,
        pruned_params=pruned_params,
        compressed_file_bytes=file_bytes,
        compressed_file_mb=file_mb,
        under_45mb_target=(file_mb <= target_mb),
        checkpoint_path=output_path,
    )

