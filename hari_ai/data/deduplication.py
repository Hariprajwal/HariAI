"""MinHash Locality Sensitive Hashing (LSH) deduplication for HariAI training corpora."""

from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterable, List, Set, Tuple


class MinHashLSH:
    """MinHash LSH deduplicator to filter exact and near-duplicate training sequences."""

    LARGE_PRIME = 2147483647  # 2^31 - 1

    def __init__(
        self,
        num_hashes: int = 64,
        num_bands: int = 16,
        k_shingle: int = 5,
        seed: int = 42,
    ):
        assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.k_shingle = k_shingle

        # Generate deterministic linear hash function parameters (a * x + b) % p
        rng = random.Random(seed)
        self.hash_a = [rng.randint(1, self.LARGE_PRIME - 1) for _ in range(num_hashes)]
        self.hash_b = [rng.randint(0, self.LARGE_PRIME - 1) for _ in range(num_hashes)]

        # LSH Hash Tables: band_idx -> {bucket_hash -> list of doc_ids}
        self.buckets: List[Dict[int, List[int]]] = [{} for _ in range(self.num_bands)]
        self.signatures: Dict[int, List[int]] = {}
        self.documents: Dict[int, str] = {}
        self._next_doc_id = 0

    def _get_shingles(self, text: str) -> Set[int]:
        """Extracts hashed character k-shingles from text."""
        normalized = " ".join(text.lower().split())
        if len(normalized) < self.k_shingle:
            # Fallback for very short strings
            return {int(hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8], 16)}

        shingles: Set[int] = set()
        for i in range(len(normalized) - self.k_shingle + 1):
            shingle_str = normalized[i : i + self.k_shingle]
            shingle_hash = int(
                hashlib.md5(shingle_str.encode("utf-8")).hexdigest()[:8], 16
            )
            shingles.add(shingle_hash)
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        """Computes MinHash signature vector for text."""
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_hashes

        sig = []
        for i in range(self.num_hashes):
            a = self.hash_a[i]
            b = self.hash_b[i]
            min_val = min(((a * s + b) % self.LARGE_PRIME) for s in shingles)
            sig.append(min_val)
        return sig

    def estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimates Jaccard similarity between two MinHash signatures."""
        assert len(sig1) == len(sig2) == self.num_hashes
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / self.num_hashes

    def is_duplicate(self, text: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Checks if text has a near-duplicate already indexed exceeding similarity threshold."""
        sig = self.compute_signature(text)

        # Candidate collection from LSH bands
        candidates: Set[int] = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = hash(tuple(sig[start:end]))

            bucket = self.buckets[b_idx].get(band_key, [])
            candidates.update(bucket)

        max_sim = 0.0
        for cand_id in candidates:
            cand_sig = self.signatures[cand_id]
            sim = self.estimate_jaccard(sig, cand_sig)
            if sim > max_sim:
                max_sim = sim
            if sim >= threshold:
                return True, max_sim

        return False, max_sim

    def index(self, text: str) -> int:
        """Indexes text into MinHash LSH tables and returns assigned doc_id."""
        doc_id = self._next_doc_id
        self._next_doc_id += 1

        sig = self.compute_signature(text)
        self.signatures[doc_id] = sig
        self.documents[doc_id] = text

        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_key = hash(tuple(sig[start:end]))

            if band_key not in self.buckets[b_idx]:
                self.buckets[b_idx][band_key] = []
            self.buckets[b_idx][band_key].append(doc_id)

        return doc_id

    def filter_corpus(
        self, corpus: Iterable[str], threshold: float = 0.8
    ) -> List[str]:
        """Filters a corpus, dropping any sample with similarity >= threshold."""
        clean_corpus: List[str] = []
        for text in corpus:
            is_dup, _ = self.is_duplicate(text, threshold=threshold)
            if not is_dup:
                self.index(text)
                clean_corpus.append(text)
        return clean_corpus

