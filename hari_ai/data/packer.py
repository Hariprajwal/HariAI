"""Memory-mapped binary dataset shard packer and streaming PyTorch dataset for HariAI."""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardWriter:
    """Packs tokenized integer sequences into compact uint16 memory-mapped binary shards."""

    def __init__(self, output_dir: str, shard_size_tokens: int = 1_000_000, prefix: str = "shard"):
        self.output_dir = output_dir
        self.shard_size_tokens = shard_size_tokens
        self.prefix = prefix
        self.current_shard_idx = 0
        self.buffer: List[int] = []
        self.shard_paths: List[str] = []

        os.makedirs(output_dir, exist_ok=True)

    def write_tokens(self, tokens: List[int]) -> None:
        """Appends tokens to buffer and flushes to disk when shard size is reached."""
        self.buffer.extend(tokens)
        while len(self.buffer) >= self.shard_size_tokens:
            self._flush_shard(self.buffer[: self.shard_size_tokens])
            self.buffer = self.buffer[self.shard_size_tokens :]

    def close(self) -> List[str]:
        """Flushes any remaining tokens in buffer to the final shard."""
        if self.buffer:
            self._flush_shard(self.buffer)
            self.buffer.clear()
        return self.shard_paths

    def _flush_shard(self, token_chunk: List[int]) -> None:
        """Writes token chunk as uint16 array to binary shard."""
        shard_filename = f"{self.prefix}_{self.current_shard_idx:04d}.bin"
        shard_path = os.path.join(self.output_dir, shard_filename)

        arr = np.array(token_chunk, dtype=np.uint16)
        with open(shard_path, "wb") as f:
            f.write(arr.tobytes())

        self.shard_paths.append(shard_path)
        self.current_shard_idx += 1


class StreamingBinaryDataset(Dataset):
    """Memory-mapped dataset that streams (input_ids, labels) batches without memory bloat."""

    def __init__(self, shard_paths: List[str], seq_len: int = 2048):
        self.shard_paths = shard_paths
        self.seq_len = seq_len

        self.mmaps: List[np.memmap] = []
        self.shard_token_counts: List[int] = []
        self.total_tokens = 0

        for path in shard_paths:
            file_bytes = os.path.getsize(path)
            num_tokens = file_bytes // 2  # uint16 is 2 bytes
            mmap = np.memmap(path, dtype=np.uint16, mode="r", shape=(num_tokens,))
            self.mmaps.append(mmap)
            self.shard_token_counts.append(num_tokens)
            self.total_tokens += num_tokens

        self.num_samples = max(0, (self.total_tokens - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def close(self) -> None:
        """Closes memory-mapped file handles cleanly (essential for Windows)."""
        for m in self.mmaps:
            if hasattr(m, "_mmap") and m._mmap is not None:
                m._mmap.close()
        self.mmaps.clear()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")

        global_start = idx * self.seq_len
        shard_idx = 0
        local_offset = global_start

        while shard_idx < len(self.shard_token_counts) and local_offset >= self.shard_token_counts[shard_idx]:
            local_offset -= self.shard_token_counts[shard_idx]
            shard_idx += 1

        needed_tokens = self.seq_len + 1
        mmap = self.mmaps[shard_idx]

        if local_offset + needed_tokens <= len(mmap):
            chunk = np.array(mmap[local_offset : local_offset + needed_tokens], dtype=np.int64)
        else:
            chunk = np.array(mmap[local_offset:], dtype=np.int64)
            remaining = needed_tokens - len(chunk)
            if shard_idx + 1 < len(self.mmaps):
                next_chunk = np.array(self.mmaps[shard_idx + 1][:remaining], dtype=np.int64)
                chunk = np.concatenate([chunk, next_chunk])

        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return {"input_ids": input_ids, "labels": labels}
