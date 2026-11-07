"""Unit tests for HariAI Native Inference Runtime and Public Python SDK."""

import unittest
from hari_ai import Cluster, HariAI
from hari_ai.model.config import UAIANanoConfig
from hari_ai.runtime.engine import (
    GenerationConfig,
    HardwareBackend,
    InferenceEngine,
    detect_best_hardware,
)


class TestRuntimeAndSDK(unittest.TestCase):
    def setUp(self):
        # Minimum vocab size for ByteFallbackBPETokenizer is 266 (10 special + 256 bytes)
        self.micro_config = UAIANanoConfig(
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

    def test_hardware_detection(self):
        backend = detect_best_hardware()
        self.assertIn(backend, [HardwareBackend.CPU_SIMD, HardwareBackend.CUDA, HardwareBackend.MPS])

    def test_hari_ai_sdk_ask_and_creator_identity(self):
        ai = HariAI("nano", config=self.micro_config)

        # Standard question
        ans = ai.ask("Explain recursion")
        self.assertIsInstance(ans, str)
        self.assertGreater(len(ans), 0)

        # Creator identity question
        identity_ans = ai.ask("Who created you?")
        self.assertIn("K R Hari Prajwal", identity_ans)
        self.assertIn("21 years old", identity_ans)

    def test_hari_ai_sdk_think(self):
        ai = HariAI("nano", config=self.micro_config)
        think_result = ai.think("Find the time complexity of merge sort")

        self.assertIn("thought", think_result)
        self.assertIn("result", think_result)
        self.assertIsInstance(think_result["thought"], str)

    def test_hari_ai_sdk_decide(self):
        ai = HariAI("nano", config=self.micro_config)
        decision = ai.decide(
            situation="A test failed",
            actions=["retry", "debug", "stop"],
        )

        self.assertIn("selected_action", decision)
        self.assertIn(decision["selected_action"], ["retry", "debug", "stop"])
        self.assertIn("confidence", decision)
        self.assertIn("requires_human_approval", decision)

    def test_engine_streaming_generator(self):
        engine = InferenceEngine(config=self.micro_config)
        stream = engine.generate_stream("Test prompt", GenerationConfig(max_new_tokens=4, temperature=0.0))

        chunks = list(stream)
        self.assertIsInstance(chunks, list)
        self.assertLessEqual(len(chunks), 4)

    def test_cluster_public_sdk_integration(self):
        cluster = Cluster(max_messages=10)
        cluster.start(workers=2)
        res = cluster.run("Solve algorithmic puzzle")

        self.assertEqual(res["status"], "COMPLETED")
        self.assertTrue(res["verification_passed"])


if __name__ == "__main__":
    unittest.main()
