"""Unit tests for the HariAI Modular LoRA Adapter and Continual Learning Engine."""

import unittest
import torch
import torch.nn as nn
from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.dream.adapter import (
    LoRALinear,
    inject_lora_adapters,
    DreamLoRAEngine,
)


class TestDreamAdapter(unittest.TestCase):
    def setUp(self):
        self.config = UAIANanoConfig(
            vocab_size=300,
            d_model=64,
            d_ff=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            n_actions=5,
        )
        self.model = UAIANanoModel(self.config)

    def test_lora_linear_initialization(self):
        base = nn.Linear(32, 64)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        # Base weights must be permanently frozen
        self.assertFalse(lora.base_linear.weight.requires_grad)
        if lora.base_linear.bias is not None:
            self.assertFalse(lora.base_linear.bias.requires_grad)

        # LoRA parameters must be trainable
        self.assertTrue(lora.lora_A.requires_grad)
        self.assertTrue(lora.lora_B.requires_grad)

        # Initial forward pass should match base output exactly because lora_B is initialized to 0
        x = torch.randn(2, 32)
        base_out = base(x)
        lora_out = lora(x)
        self.assertTrue(torch.allclose(base_out, lora_out, atol=1e-6))

    def test_lora_linear_gradient_flow(self):
        base = nn.Linear(32, 64)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(2, 32)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNone(lora.base_linear.weight.grad)
        self.assertIsNotNone(lora.lora_A.grad)
        self.assertIsNotNone(lora.lora_B.grad)

    def test_inject_lora_adapters(self):
        model = inject_lora_adapters(self.model, rank=4, alpha=8.0)
        self.assertTrue(all('lora_' in name for name, param in model.named_parameters() if param.requires_grad))
        for block in model.blocks:
            self.assertIsInstance(block.attn.q_proj, LoRALinear)
            self.assertIsInstance(block.attn.v_proj, LoRALinear)
            self.assertFalse(block.attn.q_proj.base_linear.weight.requires_grad)
            self.assertTrue(block.attn.q_proj.lora_A.requires_grad)

    def test_dream_lora_engine_lifecycle(self):
        engine = DreamLoRAEngine(self.model, rank=4, regression_tolerance_pct=1.0)
        engine.prepare_sleep_candidate()

        self.assertIsNotNone(engine.base_checkpoint_backup)
        self.assertIsInstance(self.model.blocks[0].attn.q_proj, LoRALinear)

        # Test regression gate passes when drop <= 1.0%
        self.assertTrue(engine.evaluate_regression_gate(100.0, 99.5))
        self.assertTrue(engine.evaluate_regression_gate(100.0, 99.0))
        self.assertTrue(engine.evaluate_regression_gate(100.0, 105.0))

        # Test regression gate rejects when drop > 1.0%
        self.assertFalse(engine.evaluate_regression_gate(100.0, 98.9))
        self.assertFalse(engine.evaluate_regression_gate(100.0, 85.0))

    def test_rollback(self):
        engine = DreamLoRAEngine(self.model, rank=4, regression_tolerance_pct=1.0)
        engine.prepare_sleep_candidate()
        self.assertIsInstance(self.model.blocks[0].attn.q_proj, LoRALinear)
        # Rollback restores state dict and removes adapters
        engine.rollback()
        self.assertNotIsInstance(self.model.blocks[0].attn.q_proj, LoRALinear)

    def test_promote(self):
        engine = DreamLoRAEngine(self.model, rank=4, regression_tolerance_pct=1.0)
        engine.prepare_sleep_candidate()
        # Simulate adapter learning by setting lora_B weights
        self.model.blocks[0].attn.q_proj.lora_B.data.fill_(0.01)
        with self.assertRaises(RuntimeError):
            engine.promote()
        self.assertTrue(engine.evaluate_regression_gate(100.0, 100.0))
        engine.promote()
        # Ensure adapters are removed and merged into base linear
        self.assertNotIsInstance(self.model.blocks[0].attn.q_proj, LoRALinear)
        self.assertIsNone(engine.base_checkpoint_backup)


if __name__ == "__main__":
    unittest.main()
