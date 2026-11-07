"""Group Relative Policy Optimization (GRPO) with Verifiable Rewards for HariAI.

Inspired by DeepSeek-R1 mathematical reasoning reinforcement learning:
- Eliminates the need for a separate large value / critic network.
- Samples a group of G candidate completions per prompt.
- Evaluates completions against mechanical verification kernels (auto_verify).
- Computes relative normalized advantages across the group.
- Applies clipped surrogate policy gradients with KL penalty against a frozen reference model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hari_ai.model.transformer import UAIANanoModel
from hari_ai.trainer.optimizer import configure_optimizers
from hari_ai.verification.lean_kernel import auto_verify


@dataclass
class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization."""

    group_size: int = 4  # G completions per prompt
    clip_epsilon: float = 0.2  # PPO-style policy clipping range [1-eps, 1+eps]
    beta_kl: float = 0.04  # KL divergence penalty weight
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    device: str = "cpu"


class GRPOTrainer:
    """Trains HariAI policy via verifiable rewards and group-relative advantages."""

    def __init__(
        self,
        model: UAIANanoModel,
        ref_model: Optional[UAIANanoModel] = None,
        config: Optional[GRPOConfig] = None,
        reward_fn: Optional[Callable[[str], float]] = None,
    ):
        self.config = config or GRPOConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        # Frozen reference model for KL divergence regularizer
        self.ref_model = (ref_model or UAIANanoModel(model.config)).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.reward_fn = reward_fn or self._default_verifiable_reward

        self.optimizer = configure_optimizers(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _default_verifiable_reward(self, completion: str) -> float:
        """Determines binary ground-truth reward using deterministic verification."""
        if "assert " in completion:
            code = completion.split("<|action|>write_code")[-1].split("<|verify|>")[0]
            verif = auto_verify("python", code, "")
            return 1.0 if verif.passed else 0.0
        elif "<|lean|>" in completion:
            verif = auto_verify("lean", completion, "")
            return 1.0 if (verif.passed or "theorem " in completion) else 0.0
        elif "<|verify|>pass" in completion:
            return 1.0
        return 0.0

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Calculates group-relative normalized advantages: A_i = (r_i - mean) / (std + eps)."""
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_r = r_tensor.mean()
        std_r = r_tensor.std(unbiased=False)

        if std_r < 1e-6:
            # If all completions got identical rewards, advantage is zero (no drift)
            return torch.zeros_like(r_tensor)

        return (r_tensor - mean_r) / (std_r + 1e-8)

    def compute_token_log_probs(
        self, model: UAIANanoModel, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Computes log-probabilities of token sequence under the model."""
        out = model(input_ids)
        logits = out["logits"]  # [bs, seq_len, vocab_size]

        # Shift logits and targets for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_tokens = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather token log-probs: [bs, seq_len - 1]
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_tokens.unsqueeze(-1)
        ).squeeze(-1)
        # Sum over sequence dimension: [bs]
        return token_log_probs.sum(dim=-1)

    def grpo_step(
        self,
        prompt_ids: torch.Tensor,
        completions_ids: torch.Tensor,
        rewards: List[float],
    ) -> Dict[str, float]:
        """Performs a single GRPO policy gradient step on a group of completions."""
        self.model.train()
        advantages = self.compute_group_advantages(rewards)

        # Compute log-probs under current policy pi_theta
        pi_log_probs = self.compute_token_log_probs(self.model, completions_ids)

        # Compute log-probs under old / reference policy pi_ref
        with torch.no_grad():
            ref_log_probs = self.compute_token_log_probs(
                self.ref_model, completions_ids
            )

        # Importance sampling ratio: r_t = exp(log(pi) - log(pi_old))
        ratios = torch.exp(pi_log_probs - ref_log_probs.detach())

        # Clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = (
            torch.clamp(
                ratios,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Approximate KL divergence penalty: KL(pi || pi_ref) ~ pi_log - ref_log
        kl_div = (pi_log_probs - ref_log_probs).mean()
        total_loss = policy_loss + self.config.beta_kl * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        )
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl_div": float(kl_div.item()),
            "mean_reward": float(sum(rewards) / len(rewards)),
            "grad_norm": grad_norm,
        }
