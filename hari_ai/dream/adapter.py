"""Modular Low-Rank Adaptation (LoRA) and Continual Parameter Updates for HariAI Dream Engine.

Implements:
- Parameter-isolated low-rank delta matrices: W = W_0 + (alpha / r) * (B @ A)
- Freezes base model weights while fitting adapters.
- Requires a caller-supplied score comparison before the in-memory merge.
- Supports rollback to the in-memory base snapshot.

The caller is responsible for meaningful held-out evaluations, independent
checkpoint persistence, and size checks. Frozen base weights do not guarantee
that the adapted model retains all capabilities.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel


class LoRALinear(nn.Module):
    """Linear layer augmented with trainable low-rank decomposition matrices."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        # Freeze base weights permanently
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform for A, zero initialization for B (so initial Delta W is exactly 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        # Compute low-rank adaptation delta
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def inject_lora_adapters(
    model: UAIANanoModel, rank: int = 4, alpha: float = 8.0
) -> UAIANanoModel:
    """Injects trainable LoRA matrices into attention projections while freezing all base weights."""
    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks:
        if not isinstance(block.attn.q_proj, LoRALinear):
            block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=rank, alpha=alpha)
        if not isinstance(block.attn.v_proj, LoRALinear):
            block.attn.v_proj = LoRALinear(block.attn.v_proj, rank=rank, alpha=alpha)
    return model


def remove_lora_adapters(model: UAIANanoModel) -> UAIANanoModel:
    """Removes LoRA adapters, restoring base linear layers without merging delta weights."""
    for block in model.blocks:
        if isinstance(block.attn.q_proj, LoRALinear):
            block.attn.q_proj = block.attn.q_proj.base_linear
        if isinstance(block.attn.v_proj, LoRALinear):
            block.attn.v_proj = block.attn.v_proj.base_linear
    return model


def merge_lora_adapters(model: UAIANanoModel) -> UAIANanoModel:
    """Merges trained LoRA delta weights (B @ A * scaling) into base linear layers."""
    for block in model.blocks:
        if isinstance(block.attn.q_proj, LoRALinear):
            delta = (block.attn.q_proj.lora_B @ block.attn.q_proj.lora_A) * block.attn.q_proj.scaling
            block.attn.q_proj.base_linear.weight.data.add_(delta)
            block.attn.q_proj = block.attn.q_proj.base_linear
        if isinstance(block.attn.v_proj, LoRALinear):
            delta = (block.attn.v_proj.lora_B @ block.attn.v_proj.lora_A) * block.attn.v_proj.scaling
            block.attn.v_proj.base_linear.weight.data.add_(delta)
            block.attn.v_proj = block.attn.v_proj.base_linear
    return model


class DreamLoRAEngine:
    """Orchestrates sleep-time candidate adapter training and regression-gated promotion."""

    def __init__(
        self,
        model: UAIANanoModel,
        rank: int = 4,
        regression_tolerance_pct: float = 1.0,
    ):
        self.model = model
        self.rank = rank
        self.regression_tolerance_pct = regression_tolerance_pct
        self.base_checkpoint_backup: Optional[Dict[str, Any]] = None
        self.original_trainability: Optional[Dict[str, bool]] = None
        self._regression_passed = False

    def prepare_sleep_candidate(self) -> None:
        """Freezes base model and injects low-rank adapter parameters."""
        if self.base_checkpoint_backup is not None:
            raise RuntimeError("A sleep candidate is already active")
        self._regression_passed = False
        # Create in-memory backup for immediate rollback if regression fails
        self.base_checkpoint_backup = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.original_trainability = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        inject_lora_adapters(self.model, rank=self.rank)

    def evaluate_regression_gate(
        self, pre_sleep_score: float, post_sleep_score: float
    ) -> bool:
        """Determines whether candidate adapter passes the regression safety threshold."""
        if self.base_checkpoint_backup is None or not all(math.isfinite(x) and x >= 0 for x in (pre_sleep_score, post_sleep_score)):
            self._regression_passed = False
            return False
        allowed_drop = pre_sleep_score * (self.regression_tolerance_pct / 100.0)
        passed = post_sleep_score >= (pre_sleep_score - allowed_drop)
        self._regression_passed = passed
        return passed

    def rollback(self) -> None:
        """Restores model state from backup, discarding candidate adapter."""
        remove_lora_adapters(self.model)
        if self.base_checkpoint_backup is not None:
            self.model.load_state_dict(self.base_checkpoint_backup)
            if self.original_trainability is not None:
                for name, param in self.model.named_parameters():
                    param.requires_grad = self.original_trainability[name]
            print("[DreamLoRAEngine] Rollback executed: Pre-sleep base model restored.")
        self.base_checkpoint_backup = None
        self.original_trainability = None
        self._regression_passed = False

    def promote(self) -> None:
        """Merges candidate adapter into base model weights."""
        if self.base_checkpoint_backup is None or not self._regression_passed:
            raise RuntimeError("A candidate must pass regression evaluation before promotion")
        merge_lora_adapters(self.model)
        if self.original_trainability is not None:
            for name, param in self.model.named_parameters():
                param.requires_grad = self.original_trainability[name]
        self.base_checkpoint_backup = None
        self.original_trainability = None
        self._regression_passed = False
        print("[DreamLoRAEngine] Candidate adapter successfully merged into base model.")
