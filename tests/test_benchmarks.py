"""Unit tests for the HariAI Benchmarking Suite."""

import unittest
from benchmarks.evaluate_all import BenchmarkSuite
from hari_ai import HariAI
from hari_ai.model.config import UAIANanoConfig


class TestBenchmarks(unittest.TestCase):
    def setUp(self):
        # Micro configuration for instant test run
        micro_cfg = UAIANanoConfig(
            vocab_size=300,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=64,
            n_actions=4,
            tie_word_embeddings=True,
        )
        ai = HariAI("nano", config=micro_cfg)
        self.suite = BenchmarkSuite(ai=ai)

    def test_code_generation_benchmark(self):
        res = self.suite.evaluate_code_generation()
        self.assertIn("pass_at_1_pct", res)
        self.assertIsNone(res["pass_at_1_pct"])
        self.assertEqual(res["status"], "not_evaluated")

    def test_decision_and_calibration_benchmark(self):
        res = self.suite.evaluate_decision_and_calibration()
        self.assertIn("action_accuracy_pct", res)
        self.assertIn("mean_absolute_confidence_error", res)
        self.assertLessEqual(res["mean_absolute_confidence_error"], 1.0)

    def test_system_performance_benchmark(self):
        res = self.suite.evaluate_system_performance()
        self.assertIn("throughput_tokens_per_sec", res)
        self.assertIn("under_50mb_ceiling", res)
        self.assertTrue(res["under_50mb_ceiling"])


if __name__ == "__main__":
    unittest.main()

