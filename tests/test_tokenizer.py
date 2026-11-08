"""Unit tests for the ByteFallbackBPETokenizer."""

import os
import tempfile
import unittest
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer


class TestByteFallbackBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteFallbackBPETokenizer(target_vocab_size=300)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.special_tokens_map["<|pad|>"], 0)
        self.assertEqual(self.tokenizer.special_tokens_map["<|bos|>"], 1)
        self.assertEqual(self.tokenizer.special_tokens_map["<|eos|>"], 2)
        self.assertEqual(self.tokenizer.special_tokens_map["<|thought|>"], 4)
        self.assertEqual(self.tokenizer.special_tokens_map["<|action|>"], 5)
        self.assertEqual(self.tokenizer.special_tokens_map["<|verify|>"], 7)
        self.assertEqual(self.tokenizer.special_tokens_map["<|lean|>"], 9)

    def test_base_byte_fallback_coverage(self):
        # Even with zero merges trained, any UTF-8 text must encode and decode perfectly
        text_english = "HariAI: Under 50 MB intelligence"
        tokens = self.tokenizer.encode(text_english)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text_english)

        # Unicode / Multilingual test: Hindi (Devanagari)
        text_hindi = "हरि एआई - ओम नमः शिवाय"
        tokens_hindi = self.tokenizer.encode(text_hindi)
        decoded_hindi = self.tokenizer.decode(tokens_hindi)
        self.assertEqual(decoded_hindi, text_hindi)

        # Code and Math symbols
        text_math = "theorem sum_n (n : Nat) : sum n = n*(n+1)/2 := by ..."
        tokens_math = self.tokenizer.encode(text_math)
        decoded_math = self.tokenizer.decode(tokens_math)
        self.assertEqual(decoded_math, text_math)

    def test_special_token_encoding(self):
        text_with_special = "<|thought|>Analyzing problem<|action|>run_code<|verify|>pass<|eos|>"
        tokens = self.tokenizer.encode(text_with_special)
        # Verify special tokens match their direct IDs
        self.assertEqual(tokens[0], self.tokenizer.special_tokens_map["<|thought|>"])
        self.assertEqual(tokens[-1], self.tokenizer.special_tokens_map["<|eos|>"])

        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text_with_special)

    def test_bpe_training_and_compression(self):
        corpus = [
            "def quicksort(arr):",
            "    if len(arr) <= 1: return arr",
            "    pivot = arr[len(arr) // 2]",
            "    left = [x for x in arr if x < pivot]",
            "    middle = [x for x in arr if x == pivot]",
            "    right = [x for x in arr if x > pivot]",
            "    return quicksort(left) + middle + quicksort(right)",
        ] * 10

        initial_len = len(self.tokenizer.encode(corpus[0]))
        self.tokenizer.train(corpus, max_merges=30)
        post_train_len = len(self.tokenizer.encode(corpus[0]))

        # Length in tokens should reduce due to learned subword merges
        self.assertLess(post_train_len, initial_len)

        # Roundtrip must remain lossless
        decoded = self.tokenizer.decode(self.tokenizer.encode(corpus[0]))
        self.assertEqual(decoded, corpus[0])

    def test_save_and_load(self):
        corpus = ["HariAI nano model architecture and reasoning engine"] * 5
        self.tokenizer.train(corpus, max_merges=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded = ByteFallbackBPETokenizer.load(save_path)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.merges, self.tokenizer.merges)

            test_str = "HariAI test string with <|lean|> proof"
            orig_encoded = self.tokenizer.encode(test_str)
            loaded_encoded = loaded.encode(test_str)
            self.assertEqual(orig_encoded, loaded_encoded)
            self.assertEqual(loaded.decode(loaded_encoded), test_str)


if __name__ == "__main__":
    unittest.main()

