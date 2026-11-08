"""From-scratch, document-held-out language-model pilot for HariAI."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.quantization.quantizer import quantize_model, save_quantized_checkpoint
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/fineweb_pilot.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/language_pilot_v1'))
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 16 <= args.seq_len <= 512:
        parser.error('steps and batch size must be positive; seq-len must be 16..512')
    rows = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line]
    seen = set()
    for row in rows:
        text, claimed, split = row.get('text'), row.get('sha256'), row.get('research_split')
        if not isinstance(text, str) or not text or split not in ('train', 'eval'):
            parser.error('Every document must contain nonempty text and a train/eval split')
        actual = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if claimed != actual:
            parser.error('A document hash does not match its text')
        if actual in seen:
            parser.error('Duplicate documents are not allowed across or within splits')
        seen.add(actual)
    splits = {name: [row for row in rows if row.get('research_split') == name] for name in ('train', 'eval')}
    if not all(splits.values()):
        parser.error('Both train and eval document splits are required')
    train_hashes = {r['sha256'] for r in splits['train']}
    if train_hashes.intersection(r['sha256'] for r in splits['eval']):
        parser.error('Train and eval documents overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = ByteFallbackBPETokenizer(target_vocab_size=266)
    tokenizer.save(str(args.output / 'tokenizer.json'))

    def token_stream(documents):
        ids = []
        for row in documents:
            # Byte fallback IDs follow the ten special token IDs.
            ids.extend(byte + 10 for byte in row['text'].encode('utf-8'))
            ids.append(tokenizer.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    train_ids, eval_ids = (token_stream(splits[name]) for name in ('train', 'eval'))
    if min(len(train_ids), len(eval_ids)) <= args.seq_len:
        parser.error('Not enough tokens for the configured context length')
    config = UAIANanoConfig(vocab_size=tokenizer.vocab_size, d_model=192, n_layers=6,
        n_heads=6, n_kv_heads=2, d_ff=512, max_seq_len=max(256, args.seq_len), n_actions=8)
    model = UAIANanoModel(config)
    parameter_count = model.count_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    eval_rng = random.Random(args.seed + 1)
    eval_starts = [eval_rng.randrange(len(eval_ids) - args.seq_len) for _ in range(32)]

    def batch(stream, starts):
        return torch.stack([stream[start:start + args.seq_len] for start in starts])

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for offset in range(0, len(eval_starts), args.batch_size):
            inputs = batch(eval_ids, eval_starts[offset:offset + args.batch_size])
            losses.append((model(inputs, labels=inputs)['loss'].item(), len(inputs)))
        return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)

    before = evaluate()
    print(f'Initial held-out loss: {before:.4f}', flush=True)
    training_start = time.perf_counter()
    for step in range(args.steps):
        model.train()
        starts = [rng.randrange(len(train_ids) - args.seq_len) for _ in range(args.batch_size)]
        inputs = batch(train_ids, starts)
        loss = model(inputs, labels=inputs)['loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f'{step + 1}/{args.steps} train_loss={loss.item():.4f}', flush=True)

    training_seconds = time.perf_counter()-training_start
    after = evaluate()
    print(f'Trained held-out loss: {after:.4f}', flush=True)
    torch.save({'config': config, 'model_state_dict': model.state_dict(), 'step': args.steps},
               args.output / 'hari_language_fp32.pt')
    quantize_model(model, precision='int4')
    quantized_loss = evaluate()
    export = save_quantized_checkpoint(model, str(args.output / 'hari_language_int4.pt'),
        max_size_mb=35_000_000 / (1024 * 1024))
    sources = sorted({str(row.get('source') or 'unspecified') for row in rows})
    report = {'purpose': 'language pretraining pilot; not an instruction-tuned assistant',
        'data_path': str(args.data), 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'source': sources[0] if len(sources) == 1 else 'multiple sources', 'data_sources': sources,
        'initialization': 'random', 'model_parameters':parameter_count, 'model_config':asdict(config),
        'train_documents': len(splits['train']), 'eval_documents': len(splits['eval']),
        'train_tokens': len(train_ids), 'eval_tokens': len(eval_ids), 'seed': args.seed,
        'steps': args.steps, 'batch_size': args.batch_size, 'sequence_length': args.seq_len,
        'training_tokens_processed': args.steps * args.batch_size * args.seq_len,
        'training_seconds':training_seconds,
        'training_tokens_per_second':args.steps*args.batch_size*args.seq_len/training_seconds,
        'last_training_loss_nats':float(loss.item()), 'initial_eval_loss_nats': before, 'trained_eval_loss_nats': after,
        'quantized_eval_loss_nats': quantized_loss, 'export': export}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
