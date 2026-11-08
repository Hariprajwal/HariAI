"""Experimental INT4 and ternary weight quantization for HariAI Nano.

Linear weights, including ternary values, are stored as two 4-bit values per
byte; embeddings and other tensors remain floating point. Linear weights are
dequantized on each forward pass. This is not BitNet's 1.58-bit storage format
or its trained architecture and optimized inference kernel.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel


def quantize_tensor_int4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 quantization: maps continuous weights to [-8, 7] integers."""
    # Scale per output channel or per tensor
    scale = w.abs().max(dim=-1, keepdim=True).values / 7.0
    scale = torch.clamp(scale, min=1e-8)
    w_q = torch.clamp(torch.round(w / scale), min=-8, max=7).to(torch.int8)
    return w_q, scale


def quantize_tensor_bitnet(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """BitNet 1.58-bit ternary quantization: maps continuous weights to {-1, 0, +1}."""
    scale = w.abs().mean()
    scale = torch.clamp(scale, min=1e-8)
    w_t = torch.clamp(torch.round(w / scale), min=-1, max=1).to(torch.int8)
    return w_t, scale


def pack_int4(w_int8: torch.Tensor) -> torch.Tensor:
    """Packs two 4-bit signed integers into a single uint8 byte (0.5 bytes/param)."""
    flat = w_int8.flatten()
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=flat.device)])

    low = flat[0::2] & 0x0F
    high = flat[1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed


def unpack_int4(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpacks uint8 bytes back into signed int8 values [-8, 7]."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    # Sign extension for 4-bit signed representation
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    unpacked = torch.stack([low, high], dim=1).flatten()[: original_shape.numel()]
    return unpacked.view(original_shape)


class QuantizedLinear(nn.Module):
    """Linear layer operating over packed INT4 or BitNet ternary weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        precision: str = "int4",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self.shape = torch.Size([out_features, in_features])

        # Storage for packed weights and scale factors
        packed_len = (out_features * in_features + 1) // 2
        self.register_buffer("packed_weight", torch.zeros(packed_len, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls, linear: nn.Linear, precision: str = "int4"
    ) -> "QuantizedLinear":
        """Quantizes an existing full-precision nn.Linear layer."""
        q_layer = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            precision=precision,
        )

        with torch.no_grad():
            if precision == "bitnet":
                w_q, scale = quantize_tensor_bitnet(linear.weight)
            else:
                w_q, scale = quantize_tensor_int4(linear.weight)

            q_layer.packed_weight.copy_(pack_int4(w_q))
            if scale.numel() == 1:
                q_layer.scale.fill_(scale.item())
            else:
                q_layer.scale.copy_(scale.view(linear.out_features, 1))

            if linear.bias is not None:
                q_layer.bias.copy_(linear.bias)

        return q_layer

    def dequantize(self) -> torch.Tensor:
        """Dequantizes packed representation into floating point weight tensor."""
        unpacked_int8 = unpack_int4(self.packed_weight, self.shape)
        return unpacked_int8.to(torch.float32) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_float = self.dequantize()
        return F.linear(x, w_float, self.bias)


def quantize_model(
    model: UAIANanoModel, precision: str = "int4"
) -> UAIANanoModel:
    """Recursively replaces all Linear projection layers with QuantizedLinear layers."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Split parent and child attribute name
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            q_layer = QuantizedLinear.from_float(module, precision=precision)
            setattr(parent, parts[-1], q_layer)

    return model


def save_quantized_checkpoint(
    model: UAIANanoModel, checkpoint_path: str, max_size_mb: float = 50.0
) -> Dict[str, Any]:
    """Serializes quantized model to disk and asserts the <= 50 MB budget constraint."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    checkpoint_data = {
        "state_dict": model.state_dict(),
        "config": model.config,
        "format": "hari_ai_quantized_v1",
    }
    torch.save(checkpoint_data, checkpoint_path)

    file_bytes = os.path.getsize(checkpoint_path)
    file_mb = file_bytes / (1024 * 1024)

    assert file_mb <= max_size_mb, (
        f"Model size violation: Exported checkpoint is {file_mb:.2f} MB, "
        f"which exceeds the {max_size_mb:.2f} MB hard limit!"
    )

    return {
        "path": checkpoint_path,
        "bytes": file_bytes,
        "size_mb": round(file_mb, 2),
        "under_50mb_target": True,
    }


def load_quantized_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> UAIANanoModel:
    """Loads a quantized HariAI Nano model from disk."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = data["config"]

    model = UAIANanoModel(config)
    quantize_model(model, precision="int4")
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    return model

