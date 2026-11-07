"""Unit tests for Phase 4: Multi-Teacher Distillation and GRPO Policy Optimization."""

import unittest
import torch

from hari_ai.distillation.teachers import (
    DistillationTask,
    MultiTeacherDistiller,
    TeacherModelType,
)
from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.trainer.grpo import GRPOConfig, GRPOTrainer


class TestPhase4Intelligence(unittest.TestCase):
    def setUp(self):
        self.micro_config = UAIANanoConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=32,
            n_actions=4,
            tie_word_embeddings=True,
        )
        self.model = UAIANanoModel(self.micro_config)
        self.distiller = MultiTeacherDistiller(request_timeout=0.1)

    def test_multi_teacher_routing(self):
        code_teacher = self.distiller.route_task_to_teacher(DistillationTask.CODE_SYNTHESIS)
        self.assertEqual(code_teacher, TeacherModelType.QWEN_CODER)

        math_teacher = self.distiller.route_task_to_teacher(DistillationTask.LEAN_MATH_PROOF)
        self.assertEqual(math_teacher, TeacherModelType.PHI3)

        decision_teacher = self.distiller.route_task_to_teacher(DistillationTask.DECISION_TRAJECTORY)
        self.assertEqual(decision_teacher, TeacherModelType.QWEN_CODER)

    def test_distiller_sample_and_verification_gate(self):
        sample = self.distiller.distill_sample(DistillationTask.CODE_SYNTHESIS)
        self.assertIsNotNone(sample.response)
        self.assertFalse(sample.verified)
        self.assertEqual(sample.reward, 0.0)

        # Batch generation
        batch = self.distiller.generate_verified_batch(count=6)
        self.assertEqual(len(batch), 0)

    def test_grpo_advantage_normalization(self):
        grpo = GRPOTrainer(model=self.model)

        # Identical rewards -> advantages should be zero
        equal_adv = grpo.compute_group_advantages([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(equal_adv, torch.zeros_like(equal_adv)))

        # Differentiated rewards -> winners get positive, losers get negative
        diff_adv = grpo.compute_group_advantages([1.0, 1.0, 0.0, 0.0])
        self.assertGreater(diff_adv[0].item(), 0.0)
        self.assertGreater(diff_adv[1].item(), 0.0)
        self.assertLess(diff_adv[2].item(), 0.0)
        self.assertLess(diff_adv[3].item(), 0.0)
        # Sum of normalized advantages should be close to 0
        self.assertAlmostEqual(float(diff_adv.sum().item()), 0.0, places=5)

    def test_grpo_step_and_policy_gradient(self):
        grpo = GRPOTrainer(model=self.model, config=GRPOConfig(clip_epsilon=0.2, beta_kl=0.01))

        bs = 4
        seq_len = 8
        dummy_prompt = torch.randint(0, self.micro_config.vocab_size, (1, 4))
        dummy_completions = torch.randint(0, self.micro_config.vocab_size, (bs, seq_len))
        rewards = [1.0, 1.0, 0.0, 0.0]

        metrics = grpo.grpo_step(
            prompt_ids=dummy_prompt,
            completions_ids=dummy_completions,
            rewards=rewards,
        )

        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("kl_div", metrics)
        self.assertFalse(torch.isnan(torch.tensor(metrics["total_loss"])))


if __name__ == "__main__":
    unittest.main()

