"""Unit tests for the HariNano Layer Pruning and Model Compressor."""

import os
import tempfile
import unittest
import torch
from hari_ai.distillation.compressor import (
    compress_to_target_budget,
    prune_transformer_layers,
)
from hari_ai.model.config import HariNanoConfig
from hari_ai.model.transformer import HariNanoModel


class TestModelCompressor(unittest.TestCase):
    def setUp(self):
        self.config = HariNanoConfig(
            vocab_size=300,
            d_model=64,
            n_layers=6,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=64,
        )
        self.model = HariNanoModel(self.config)

    def test_layer_pruning(self):
        original_layers = len(self.model.blocks)
        self.assertEqual(original_layers, 6)

        pruned = prune_transformer_layers(self.model, target_layers=3)
        self.assertEqual(len(pruned.blocks), 3)
        self.assertEqual(pruned.config.n_layers, 3)

    def test_compress_to_target_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "compressed_nano.pt")
            report = compress_to_target_budget(
                model=self.model,
                output_path=out_path,
                target_mb=45.0,
                target_layers=3,
                precision="int4",
            )
            self.assertTrue(report.under_45mb_target)
            self.assertLessEqual(report.compressed_file_mb, 45.0)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()

