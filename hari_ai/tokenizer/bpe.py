"""Byte-Fallback Byte-Pair Encoding (BPE) Tokenizer for HariAI Nano.

Implements:
- 10 Special Control and Decision Tokens (<|pad|>, <|bos|>, <|eos|>, <|thought|>, <|action|>, <|observe|>, <|verify|>, <|confidence|>, <|lean|>)
- 256 Single-Byte Fallback Tokens (IDs 10..265) ensuring 100% UTF-8 coverage with zero OOV errors.
- Learned BPE Merges up to target vocabulary size (default V = 4,096).
- Lossless round-trip encoding and decoding: decode(encode(text)) == text.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple


class ByteFallbackBPETokenizer:
    """Byte-Fallback BPE Tokenizer with structured control token support."""

    SPECIAL_TOKENS = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|unk|>",
        "<|thought|>",
        "<|action|>",
        "<|observe|>",
        "<|verify|>",
        "<|confidence|>",
        "<|lean|>",
    ]

    def __init__(self, target_vocab_size: int = 4096):
        self.target_vocab_size = target_vocab_size
        self.special_tokens_map: Dict[str, int] = {}
        self.inverse_special_tokens_map: Dict[int, str] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.merges: Dict[Tuple[int, int], int] = {}  # (pair0, pair1) -> new_token_id

        self._init_base_vocabulary()

    def _init_base_vocabulary(self) -> None:
        """Initializes special tokens (0..9) and base byte tokens (10..265)."""
        # 1. Special tokens
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.special_tokens_map[tok] = idx
            self.inverse_special_tokens_map[idx] = tok
            self.vocab[idx] = tok.encode("utf-8")

        # 2. 256 single byte tokens
        byte_offset = len(self.SPECIAL_TOKENS)  # 10
        for b in range(256):
            token_id = byte_offset + b
            byte_val = bytes([b])
            self.vocab[token_id] = byte_val
            self.inverse_vocab[byte_val] = token_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_tokens_map["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_tokens_map["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens_map["<|eos|>"]

    def _get_stats(self, token_lists: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Counts frequency of all adjacent token pairs."""
        counts: Dict[Tuple[int, int], int] = {}
        for tokens in token_lists:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(
        self, token_lists: List[List[int]], pair: Tuple[int, int], new_id: int
    ) -> List[List[int]]:
        """Replaces all occurrences of pair in token_lists with new_id."""
        p0, p1 = pair
        new_lists = []
        for tokens in token_lists:
            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_lists.append(new_tokens)
        return new_lists

    def train(
        self,
        texts: Iterable[str],
        max_merges: Optional[int] = None,
        min_frequency: int = 2,
    ) -> None:
        """Trains BPE merges over input corpus up to target_vocab_size."""
        byte_offset = len(self.SPECIAL_TOKENS)

        # Convert texts to initial list of byte token IDs
        token_lists: List[List[int]] = []
        for text in texts:
            if not text:
                continue
            raw_bytes = text.encode("utf-8")
            token_lists.append([byte_offset + b for b in raw_bytes])

        target_merges = (
            max_merges
            if max_merges is not None
            else self.target_vocab_size - len(self.vocab)
        )

        for _ in range(target_merges):
            stats = self._get_stats(token_lists)
            if not stats:
                break

            # Find the most frequent pair
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            if stats[best_pair] < min_frequency:
                break

            new_token_id = len(self.vocab)
            # Combine the underlying byte representations
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_token_id
            self.merges[best_pair] = new_token_id

            token_lists = self._merge_pair(token_lists, best_pair, new_token_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes string into token IDs with special token isolation and byte-fallback BPE."""
        if not text:
            res: List[int] = []
            if add_bos:
                res.insert(0, self.bos_id)
            if add_eos:
                res.append(self.eos_id)
            return res

        # Split text by special tokens if present
        pattern = "(" + "|".join(re.escape(tok) for tok in self.SPECIAL_TOKENS) + ")"
        parts = re.split(pattern, text)

        result: List[int] = []
        byte_offset = len(self.SPECIAL_TOKENS)

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_map:
                result.append(self.special_tokens_map[part])
            else:
                # Convert string part to initial byte token IDs
                tokens = [byte_offset + b for b in part.encode("utf-8")]

                # Iteratively apply learned merges
                while len(tokens) >= 2:
                    # Find all adjacent pairs that exist in self.merges
                    stats = {
                        (tokens[i], tokens[i + 1]): self.merges[(tokens[i], tokens[i + 1])]
                        for i in range(len(tokens) - 1)
                        if (tokens[i], tokens[i + 1]) in self.merges
                    }
                    if not stats:
                        break

                    # Pick the merge with lowest rank / earliest learned
                    best_pair = min(stats, key=lambda p: stats[p])
                    tokens = self._merge_pair([tokens], best_pair, self.merges[best_pair])[0]

                result.extend(tokens)

        if add_bos:
            result.insert(0, self.bos_id)
        if add_eos:
            result.append(self.eos_id)

        return result

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to string."""
        byte_chunks: List[bytes] = []

        for tid in token_ids:
            if tid in self.inverse_special_tokens_map:
                if not skip_special_tokens:
                    tok_str = self.inverse_special_tokens_map[tid]
                    byte_chunks.append(tok_str.encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                # Unknown token fallback
                byte_chunks.append(b"?")

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        """Saves tokenizer state to JSON format."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens_map,
            "merges": [
                {"pair": list(pair), "token_id": token_id}
                for pair, token_id in self.merges.items()
            ],
            "vocab": {
                str(tid): list(raw_bytes) for tid, raw_bytes in self.vocab.items()
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ByteFallbackBPETokenizer":
        """Loads tokenizer state from JSON format."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(target_vocab_size=data["target_vocab_size"])
        tokenizer.special_tokens_map = data["special_tokens"]
        tokenizer.inverse_special_tokens_map = {
            v: k for k, v in tokenizer.special_tokens_map.items()
        }

        tokenizer.vocab = {
            int(tid): bytes(raw_bytes) for tid, raw_bytes in data["vocab"].items()
        }
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        tokenizer.merges = {
            (item["pair"][0], item["pair"][1]): item["token_id"]
            for item in data["merges"]
        }
        return tokenizer
