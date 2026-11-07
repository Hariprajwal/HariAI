"""Unit tests for HariAI Extreme Compression (<50MB Target) and Quantization."""

import os
import tempfile
import unittest
import torch
import torch.nn as nn

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.quantization.quantizer import (
    QuantizedLinear,
    load_quantized_checkpoint,
    pack_int4,
    quantize_model,
    quantize_tensor_int4,
    save_quantized_checkpoint,
    unpack_int4,
)


class TestQuantization(unittest.TestCase):
    def test_pack_and_unpack_int4(self):
        # Sample tensor with diverse positive and negative 4-bit values [-8..7]
        values = torch.tensor([-8, -5, -1, 0, 1, 3, 7, -2], dtype=torch.int8)
        packed = pack_int4(values)

        # 8 values packed into 4 bytes (0.5 bytes per value)
        self.assertEqual(packed.numel(), 4)
        self.assertEqual(packed.dtype, torch.uint8)

        # Unpack and verify exact sign preservation
        unpacked = unpack_int4(packed, values.shape)
        self.assertTrue(torch.equal(values, unpacked))

    def test_quantized_linear_layer(self):
        linear = nn.Linear(32, 16, bias=False)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)

        q_linear = QuantizedLinear.from_float(linear, precision="int4")
        x = torch.randn(2, 8, 32)

        out_float = linear(x)
        out_quant = q_linear(x)

        self.assertEqual(out_quant.shape, (2, 8, 16))
        # Reconstruction error should be bounded and low
        rel_error = (out_float - out_quant).abs().mean() / out_float.abs().mean()
        self.assertLess(float(rel_error.item()), 0.25)

    def test_quantize_model_and_save_load_parity(self):
        config = UAIANanoConfig(
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
        model = UAIANanoModel(config)
        model.eval()

        # Run forward pass on original float model
        input_ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            orig_out = model(input_ids)

        # Quantize model
        q_model = quantize_model(model, precision="int4")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "hari_ai_nano_quantized.pt")
            save_info = save_quantized_checkpoint(q_model, ckpt_path)

            self.assertTrue(os.path.exists(ckpt_path))
            self.assertTrue(save_info["under_50mb_target"])
            self.assertLessEqual(save_info["size_mb"], 50.0)

            # Reload model and test forward pass
            reloaded_model = load_quantized_checkpoint(ckpt_path, device="cpu")
            with torch.no_grad():
                reloaded_out = reloaded_model(input_ids)

            self.assertEqual(
                reloaded_out["logits"].shape,
                (2, 8, config.vocab_size),
            )
            self.assertEqual(
                reloaded_out["action_logits"].shape,
                (2, config.n_actions),
            )


if __name__ == "__main__":
    unittest.main()

