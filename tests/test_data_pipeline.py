"""Unit tests for the data deduplication, synthetic generation, and shard streaming pipeline."""

import os
import tempfile
import unittest
import torch

from hari_ai.data.deduplication import MinHashLSH
from hari_ai.data.packer import ShardWriter, StreamingBinaryDataset
from hari_ai.data.synthetic import SyntheticCurriculumGenerator


class TestDataPipeline(unittest.TestCase):
    def test_minhash_deduplication(self):
        lsh = MinHashLSH(num_hashes=64, num_bands=16, k_shingle=4)

        doc1 = "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr if x < arr[0]])"
        doc2 = "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr if x < arr[0]])"  # exact duplicate
        doc3 = "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr if x <= arr[0]])"  # near identical variant
        doc4 = "A distinct document about planets, oceans, and weather."  # completely different

        # Index doc1
        lsh.index(doc1)

        # Doc2 should be detected as exact duplicate
        is_dup2, sim2 = lsh.is_duplicate(doc2, threshold=0.8)
        self.assertTrue(is_dup2)
        self.assertAlmostEqual(sim2, 1.0, delta=0.05)

        # Doc3 should be detected as near duplicate
        is_dup3, sim3 = lsh.is_duplicate(doc3, threshold=0.7)
        self.assertTrue(is_dup3)

        # Doc4 should NOT be duplicate
        is_dup4, sim4 = lsh.is_duplicate(doc4, threshold=0.4)
        self.assertFalse(is_dup4)

    def test_synthetic_curriculum_generation(self):
        generator = SyntheticCurriculumGenerator(seed=123)

        code_sample = generator.generate_code_sample()
        self.assertIn("<|thought|>", code_sample)
        self.assertIn("<|action|>write_code", code_sample)
        self.assertIn("assert ", code_sample)
        self.assertIn("<|verify|>pass<|eos|>", code_sample)

        lean_sample = generator.generate_lean_math_sample()
        self.assertIn("<|thought|>", lean_sample)
        self.assertIn("<|lean|>", lean_sample)
        self.assertIn("theorem ", lean_sample)
        self.assertIn("<|verify|>pass<|eos|>", lean_sample)

        decision_sample = generator.generate_decision_sample()
        self.assertIn("Situation:", decision_sample)
        self.assertIn("<|action|>", decision_sample)
        self.assertIn("<|confidence|>", decision_sample)
        self.assertIn("Permission Required:", decision_sample)

        mixture = generator.generate_dataset_mixture(num_samples=20)
        self.assertEqual(len(mixture), 20)

    def test_shard_writer_and_streaming_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ShardWriter(output_dir=tmpdir, shard_size_tokens=256, prefix="test_shard")

            # Generate synthetic token sequence
            dummy_tokens = list(range(1000))
            writer.write_tokens(dummy_tokens)
            shard_paths = writer.close()

            self.assertGreater(len(shard_paths), 1)

            # Test StreamingBinaryDataset
            seq_len = 64
            dataset = StreamingBinaryDataset(shard_paths=shard_paths, seq_len=seq_len)
            self.assertGreater(len(dataset), 0)

            sample = dataset[0]
            self.assertEqual(sample["input_ids"].shape, (seq_len,))
            self.assertEqual(sample["labels"].shape, (seq_len,))

            # The model applies the causal shift internally.
            self.assertTrue(torch.equal(sample["input_ids"], sample["labels"]))

            # Cleanly close memory maps before temporary directory deletion on Windows
            dataset.close()

    def test_sequence_can_cross_multiple_small_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ShardWriter(output_dir=tmpdir, shard_size_tokens=3)
            writer.write_tokens(list(range(20)))
            dataset = StreamingBinaryDataset(writer.close(), seq_len=8)
            try:
                sample = dataset[1]
                self.assertEqual(sample["input_ids"].tolist(), list(range(8, 16)))
                self.assertTrue(torch.equal(sample["input_ids"], sample["labels"]))
            finally:
                dataset.close()


if __name__ == "__main__":
    unittest.main()
