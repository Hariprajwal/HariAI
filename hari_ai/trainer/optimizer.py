"""Optimizer and learning rate schedule configuration for HariAI from-scratch training."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def configure_optimizers(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Configures AdamW with decoupled weight decay.

    Applies weight decay strictly to 2D+ weight matrices (projections and embeddings)
    while exempting 1D parameters (RMSNorm scale weights and biases) to prevent
    gradient instability.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # If parameter is 2D or more, apply weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps)


class CosineWarmupScheduler:
    """Cosine Annealing learning rate schedule with linear warmup and min_lr floor."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        """Calculates expected learning rate at given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * float(step + 1) / float(max(1, self.warmup_steps))

        if step > self.max_steps:
            # Min LR floor
            return self.min_lr

        # Cosine decay down to min_lr
        decay_ratio = float(step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, step: Optional[int] = None) -> float:
        """Updates optimizer learning rate and advances step counter."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
