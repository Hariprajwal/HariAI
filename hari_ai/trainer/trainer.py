"""Core training loop and checkpoint management for HariAI Nano from scratch."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hari_ai.model.transformer import UAIANanoModel
from hari_ai.trainer.optimizer import CosineWarmupScheduler, configure_optimizers


@dataclass
class TrainerConfig:
    """Hyperparameters and configuration for model training."""

    output_dir: str = "checkpoints"
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    save_every_steps: int = 250
    eval_every_steps: int = 100
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class UAIATrainer:
    """Trainer orchestrating optimizer updates, gradient clipping, metrics, and checkpoints."""

    def __init__(
        self,
        model: UAIANanoModel,
        config: TrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = configure_optimizers(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
        )

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.global_step = 0
        self.total_tokens_processed = 0
        self.history: List[Dict[str, Any]] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs a single forward, backward, and optimization step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        action_targets = (
            batch["action_targets"].to(self.device)
            if "action_targets" in batch
            else None
        )
        conf_targets = (
            batch["confidence_targets"].to(self.device)
            if "confidence_targets" in batch
            else None
        )
        perm_targets = (
            batch["permission_targets"].to(self.device)
            if "permission_targets" in batch
            else None
        )

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            action_targets=action_targets,
            confidence_targets=conf_targets,
            permission_targets=perm_targets,
        )

        loss = outputs["loss"] / self.config.grad_accum_steps
        loss.backward()

        # Update metrics
        bs, seq_len = input_ids.shape
        self.total_tokens_processed += bs * seq_len

        grad_norm = 0.0
        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss": float(outputs["loss"].item()),
            "lm_loss": (
                float(outputs["lm_loss"].item())
                if outputs["lm_loss"] is not None
                else 0.0
            ),
            "grad_norm": grad_norm,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def train(self) -> List[Dict[str, Any]]:
        """Executes the full training loop up to config.max_steps."""
        print(f"Starting HariAI from-scratch training on device: {self.device}")
        data_iter = iter(self.dataloader)
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            step_metrics = self.train_step(batch)
            self.global_step += 1

            if self.global_step % 20 == 0 or self.global_step == 1:
                elapsed = time.time() - start_time
                tok_per_sec = (
                    self.total_tokens_processed / elapsed if elapsed > 0 else 0.0
                )
                log_entry = {
                    "step": self.global_step,
                    "loss": round(step_metrics["loss"], 4),
                    "lm_loss": round(step_metrics["lm_loss"], 4),
                    "lr": round(step_metrics["lr"], 6),
                    "tok_per_sec": round(tok_per_sec, 1),
                }
                self.history.append(log_entry)
                print(
                    f"Step {self.global_step}/{self.config.max_steps} | "
                    f"Loss: {log_entry['loss']} | LR: {log_entry['lr']} | "
                    f"Speed: {log_entry['tok_per_sec']} tok/s"
                )

            # Checkpoint save
            if self.global_step % self.config.save_every_steps == 0:
                self.save_checkpoint(self.global_step)

        # Final checkpoint save
        final_ckpt = self.save_checkpoint(self.global_step)
        print(f"Training completed successfully. Final checkpoint: {final_ckpt}")
        return self.history

    def save_checkpoint(self, step: int) -> str:
        """Serializes model weights, optimizer state, scheduler, and step count."""
        checkpoint_filename = f"hari_ai_step_{step:06d}.pt"
        checkpoint_path = os.path.join(self.config.output_dir, checkpoint_filename)

        checkpoint_data = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "trainer_config": self.config,
            "total_tokens_processed": self.total_tokens_processed,
            "history": self.history,
        }

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restores training state from an existing checkpoint."""
        checkpoint_data = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.global_step = checkpoint_data["step"]
        self.total_tokens_processed = checkpoint_data.get(
            "total_tokens_processed", 0
        )
        self.history = checkpoint_data.get("history", [])
        self.scheduler.step(self.global_step)
        print(f"Successfully resumed checkpoint from {checkpoint_path} at step {self.global_step}")


HariNanoTrainer = UAIATrainer


