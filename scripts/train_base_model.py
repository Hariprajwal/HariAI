"""End-to-End Base Model Training Script for HariAI Nano.

Pipeline:
1. Tokenizer initialization (Byte-Fallback BPE V=4,096).
2. Synthetic training curriculum generation (Algorithmic Code, Lean 4 Math, Decision Trajectories).
3. Deduplication with MinHash LSH.
4. Memory-mapped uint16 binary shard packing.
5. Random parameter initialization for UAIA-Nano (46.4M parameters).
6. Execution of training loop with AdamW, Cosine Annealing, and Gradient Clipping.
7. Checkpoint generation and loss validation.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from hari_ai.data.deduplication import MinHashLSH
from hari_ai.data.packer import ShardWriter, StreamingBinaryDataset
from hari_ai.data.synthetic import SyntheticCurriculumGenerator
from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer
from hari_ai.trainer.trainer import TrainerConfig, UAIATrainer


def main():
    parser = argparse.ArgumentParser(description="Train HariAI Nano from scratch.")
    parser.add_argument("--data_dir", type=str, default="data/shards", help="Binary shard output directory.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Model checkpoint directory.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of synthetic samples to generate.")
    parser.add_argument("--max_steps", type=int, default=50, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step.")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence context length for training.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, auto).")
    parser.add_argument("--micro", action="store_true", help="Use micro architecture for rapid verification.")
    args = parser.parse_args()

    print("=" * 80)
    print(" HARIAI NANO — FROM-SCRATCH TRAINING PIPELINE")
    print("=" * 80)

    # 1. Tokenizer
    print("\n[Step 1/6] Initializing ByteFallbackBPETokenizer (V=4,096)...")
    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    os.makedirs(args.data_dir, exist_ok=True)

    generator = SyntheticCurriculumGenerator(seed=42)
    raw_texts = generator.generate_dataset_mixture(num_samples=args.num_samples)

    tokenizer = ByteFallbackBPETokenizer(target_vocab_size=4096)
    tokenizer.train(raw_texts, max_merges=100)
    tokenizer.save(tokenizer_path)
    print(f" -> Tokenizer vocabulary trained: {tokenizer.vocab_size} tokens.")

    # 2. Deduplication
    print("\n[Step 2/6] Running MinHash LSH deduplication on curriculum data...")
    lsh = MinHashLSH(num_hashes=64, num_bands=16, k_shingle=4)
    clean_texts = lsh.filter_corpus(raw_texts, threshold=0.85)
    print(f" -> Filtered {len(raw_texts)} raw samples down to {len(clean_texts)} unique samples.")

    # 3. Binary Shard Packing
    print("\n[Step 3/6] Packing tokens into uint16 memory-mapped binary shards...")
    shard_writer = ShardWriter(output_dir=args.data_dir, shard_size_tokens=10000, prefix="hari_train")

    total_tokens = 0
    for text in clean_texts:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        shard_writer.write_tokens(tokens)
        total_tokens += len(tokens)

    shard_paths = shard_writer.close()
    print(f" -> Packed {total_tokens} tokens into {len(shard_paths)} binary shard(s).")

    # 4. Dataset Loader
    print("\n[Step 4/6] Creating StreamingBinaryDataset (Zero-RAM memmap)...")
    dataset = StreamingBinaryDataset(shard_paths=shard_paths, seq_len=args.seq_len)
    print(f" -> Dataset initialized with {len(dataset)} training sequences of length {args.seq_len}.")

    # 5. Architecture Initialization
    print("\n[Step 5/6] Initializing UAIA-Nano neural network from random initialization...")
    if args.micro:
        model_config = UAIANanoConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=args.seq_len,
            n_actions=8,
        )
    else:
        model_config = UAIANanoConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=max(args.seq_len, 256),
        )

    model = UAIANanoModel(model_config)
    total_params = model.count_parameters()
    int4_mb = model.calculate_size_mb("int4")
    print(f" -> Model initialized: {total_params:,} parameters ({int4_mb:.2f} MB at INT4).")

    # 6. Training Execution
    print("\n[Step 6/6] Launching UAIATrainer optimization loop...")
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=2,
        learning_rate=1e-3 if args.micro else 3e-4,
        max_steps=args.max_steps,
        warmup_steps=10,
        save_every_steps=args.max_steps,
        device=args.device,
    )

    trainer = UAIATrainer(
        model=model,
        config=trainer_config,
        train_dataset=dataset,
    )

    history = trainer.train()

    # Cleanly close memory maps
    dataset.close()

    print("\n" + "=" * 80)
    print(f" TRAINING RUN COMPLETE: Final Step {history[-1]['step']}, Final Loss: {history[-1]['loss']}")
    print(f" Checkpoints saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
