import unittest

from hari_ai.model.budget import ModelBudget
from hari_ai.model.config import HariNanoConfig
from hari_ai.model.transformer import HariNanoModel


class ParameterBudgetTests(unittest.TestCase):
    def test_matches_real_tiny_model_with_tied_and_untied_embeddings(self):
        for tied in (True, False):
            cfg = HariNanoConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4,
                                 n_kv_heads=2, d_ff=128, n_actions=8,
                                 tie_word_embeddings=tied, max_seq_len=32)
            expected = ModelBudget(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads,
                                   cfg.n_kv_heads, cfg.d_ff, cfg.n_actions, tied)
            self.assertEqual(expected.breakdown()['total'], HariNanoModel(cfg).count_parameters())

    def test_size_bound_and_invalid_grouped_attention(self):
        model = ModelBudget(16384, 576, 32, 9, 3, 1600)
        self.assertGreater(model.ideal_weight_bytes(4), 50_000_000)
        with self.assertRaises(ValueError):
            ModelBudget(300, 64, 2, 4, 3, 128)
