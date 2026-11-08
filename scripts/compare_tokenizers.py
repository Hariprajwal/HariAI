"""Measure byte versus learned BPE tokenization on held-out documents."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/fineweb_pilot.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/tokenizer_pilot_v1'))
    parser.add_argument('--merges', type=int, default=128)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line]
    train = [r['text'][:1024] for r in rows if r['research_split'] == 'train'][:20]
    evaluation = [r['text'][:1024] for r in rows if r['research_split'] == 'eval'][:20]
    if not train or not evaluation:
        parser.error('Requires training and evaluation documents')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = ByteFallbackBPETokenizer(target_vocab_size=266 + args.merges)
    start = time.perf_counter()
    tokenizer.train(train, max_merges=args.merges)
    tokenizer.save(str(args.output / 'tokenizer.json'))
    base = ByteFallbackBPETokenizer(target_vocab_size=266)
    byte_tokens = sum(len(base.encode(text)) for text in evaluation)
    learned_tokens = sum(len(tokenizer.encode(text)) for text in evaluation)
    report = {'training_documents': len(train), 'evaluation_documents': len(evaluation),
              'vocab_size': tokenizer.vocab_size, 'learned_merges': len(tokenizer.merges),
              'byte_tokens': byte_tokens, 'bpe_tokens': learned_tokens,
              'token_reduction_fraction': 1.0 - learned_tokens / byte_tokens,
              'extra_fp32_embedding_bytes_at_width_192': (tokenizer.vocab_size - 266) * 192 * 4,
              'tokenizer_file_bytes': (args.output / 'tokenizer.json').stat().st_size,
              'elapsed_seconds': round(time.perf_counter() - start, 2),
              'scope': '20 short held-out document prefixes; not an assistant quality benchmark.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
