"""Comprehensive unit tests for the UAIA-Nano neural architecture."""

import unittest
import torch

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.layers import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_emb,
)
from hari_ai.model.transformer import (
    StructuredDecisionHead,
    UAIANanoModel,
    UAIATransformerBlock,
)


class TestUAIANanoArchitecture(unittest.TestCase):
    def setUp(self):
        # Micro configuration for rapid unit testing
        self.micro_config = UAIANanoConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=128,
            n_actions=8,
            tie_word_embeddings=True,
        )

    def test_config_dimensions(self):
        config = UAIANanoConfig()
        self.assertEqual(config.vocab_size, 4096)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_layers, 16)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.d_ff, 1376)
        self.assertTrue(config.tie_word_embeddings)

    def test_rms_norm(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        self.assertEqual(out.shape, (2, 8, 64))
        # Variance of normalized dimension should be close to 1
        var = out.pow(2).mean(-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_rotary_embedding(self):
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        x = torch.randn(2, 4, 10, 16)
        cos, sin = rope(x, seq_len=10)
        self.assertEqual(cos.shape, (10, 16))
        self.assertEqual(sin.shape, (10, 16))

        xq = torch.randn(2, 4, 10, 16)
        xk = torch.randn(2, 2, 10, 16)
        xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_grouped_query_attention(self):
        gqa = GroupedQueryAttention(self.micro_config)
        rope = RotaryEmbedding(dim=self.micro_config.head_dim, max_seq_len=32)
        x = torch.randn(2, 8, self.micro_config.d_model)
        cos, sin = rope(x, seq_len=8)

        out, kv = gqa(x, cos=cos, sin=sin)
        self.assertEqual(out.shape, (2, 8, self.micro_config.d_model))
        self.assertEqual(len(kv), 2)
        # kv shapes: [bs, n_kv_heads, seq_len, head_dim]
        self.assertEqual(kv[0].shape, (2, 2, 8, self.micro_config.head_dim))

    def test_swiglu_ffn(self):
        swiglu = SwiGLU(self.micro_config)
        x = torch.randn(2, 8, self.micro_config.d_model)
        out = swiglu(x)
        self.assertEqual(out.shape, (2, 8, self.micro_config.d_model))

    def test_transformer_block(self):
        block = UAIATransformerBlock(self.micro_config)
        rope = RotaryEmbedding(dim=self.micro_config.head_dim, max_seq_len=32)
        x = torch.randn(2, 8, self.micro_config.d_model)
        cos, sin = rope(x, seq_len=8)

        out, kv = block(x, cos=cos, sin=sin)
        self.assertEqual(out.shape, (2, 8, self.micro_config.d_model))

    def test_full_nano_model_forward_and_loss(self):
        model = UAIANanoModel(self.micro_config)
        batch_size = 2
        seq_len = 8

        input_ids = torch.randint(0, self.micro_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        action_targets = torch.randint(0, self.micro_config.n_actions, (batch_size,))
        conf_targets = torch.tensor([1.0, 0.0])

        output = model(
            input_ids=input_ids,
            labels=labels,
            action_targets=action_targets,
            confidence_targets=conf_targets,
        )

        # Shape assertions
        self.assertEqual(
            output["logits"].shape,
            (batch_size, seq_len, self.micro_config.vocab_size),
        )
        self.assertEqual(
            output["action_logits"].shape,
            (batch_size, self.micro_config.n_actions),
        )
        self.assertEqual(output["confidence"].shape, (batch_size, 1))

        # Loss assertions
        self.assertIsNotNone(output["loss"])
        self.assertIsNotNone(output["lm_loss"])
        self.assertIsNotNone(output["action_loss"])
        self.assertIsNotNone(output["confidence_loss"])

        # Backward pass gradient verification
        output["loss"].backward()
        self.assertIsNotNone(model.token_embeddings.weight.grad)
        self.assertIsNotNone(model.decision_head.action_proj.weight.grad)
        self.assertIsNotNone(model.blocks[0].attn.q_proj.weight.grad)

    def test_parameter_count_and_50mb_feasibility(self):
        # Create standard UAIA-Nano model
        standard_config = UAIANanoConfig()
        model = UAIANanoModel(standard_config)

        total_params = model.count_parameters()
        non_embed_params = model.count_parameters(non_embedding=True)
        int4_size_mb = model.calculate_size_mb(precision="int4")
        bitnet_size_mb = model.calculate_size_mb(precision="bitnet")

        # Validate total parameters are within the ~46.4M analytical specification
        self.assertGreater(total_params, 45_000_000)
        self.assertLess(total_params, 50_000_000)

        # Validate INT4 and BitNet sizes strictly respect the <= 50 MB hard budget
        self.assertLess(int4_size_mb, 50.0)
        self.assertLess(bitnet_size_mb, 50.0)

        # Check expected numbers (~26 MB at INT4)
        self.assertAlmostEqual(int4_size_mb, 26.0, delta=2.5)

    def test_autoregressive_generation(self):
        model = UAIANanoModel(self.micro_config)
        prompt_ids = torch.tensor([[10, 20, 30]])
        generated = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)

        self.assertEqual(generated.shape, (1, 8))  # 3 prompt tokens + 5 new tokens
        self.assertTrue(torch.equal(generated[:, :3], prompt_ids))


if __name__ == "__main__":
    unittest.main()
