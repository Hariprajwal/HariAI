"""Small, transparent smoke probe for a local HariAI checkpoint.

Scores are exact matches on a hand-written probe, not established benchmarks.
Generated code is never executed. Treat the probe as public and avoid training on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from hari_ai.runtime.engine import GenerationConfig, InferenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path, help='Matching tokenizer, if not beside checkpoint')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).with_name('general_probe.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path.')
    torch.set_num_threads(4)
    candidates = [args.checkpoint.parent / 'tokenizer.json', Path('data/shards/tokenizer.json')]
    tokenizer_path = args.tokenizer or next((path for path in candidates if path.is_file()), None)
    engine = InferenceEngine(str(args.checkpoint), tokenizer_path=str(tokenizer_path) if tokenizer_path else None)
    cases = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    results = []
    start = time.perf_counter()
    for case in cases:
        prompt = 'User: ' + case['prompt'] + '\nHariAI:'
        answer = engine.generate(prompt, GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0)).strip()
        accepted = [value.strip().casefold() for value in case['answers']]
        results.append({'domain': case['domain'], 'prompt': case['prompt'], 'generated': answer,
                        'answers': case['answers'], 'correct': answer.casefold() in accepted})
    groups = {}
    for result in results:
        group = groups.setdefault(result['domain'], {'correct': 0, 'total': 0})
        group['total'] += 1
        group['correct'] += int(result['correct'])
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = {'dataset': str(args.dataset), 'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest,
              'tokenizer': str(tokenizer_path) if tokenizer_path else 'built-in byte tokenizer',
              'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() if tokenizer_path else None,
              'checkpoint_bytes': args.checkpoint.stat().st_size, 'max_new_tokens': args.max_new_tokens,
              'elapsed_seconds': round(time.perf_counter() - start, 3),
              'scope': 'Small public smoke probe; exact matches, not a general assistant benchmark.',
              'correct': sum(item['correct'] for item in results), 'total': len(results),
              'domains': groups, 'cases': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('correct', 'total', 'domains', 'checkpoint_bytes')}, indent=2))


if __name__ == '__main__':
    main()
