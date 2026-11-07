"""Collect independently checked arithmetic answers from a local Ollama teacher.

Each example includes the exact expression, expected answer, teacher identity,
and whether its final answer matches. Nothing is silently labeled verified.
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='http://127.0.0.1:11526')
    parser.add_argument('--model', default='hari-teacher-qwen')
    parser.add_argument('--output', default='data/teacher_arithmetic.jsonl')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    expressions = [(a, b, op) for a in range(2, 30) for b in range(2, 20) for op in ('+', '-')]
    rng.shuffle(expressions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous collection.
    with output.open('x', encoding='utf-8') as sink:
        for index, (a, b, op) in enumerate(expressions[:args.count]):
            prompt = f'What is {a} {op} {b}? Answer with only the integer.'
            expected = a + b if op == '+' else a - b
            payload = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
                       'stream': False, 'options': {'temperature': 0.1, 'num_predict': 512, 'num_ctx': 1024}}
            request = urllib.request.Request(args.endpoint.rstrip('/') + '/api/chat',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=180) as resp:
                    result = json.load(resp)
                answer = result.get('message', {}).get('content', '').strip()
                verified = bool(re.fullmatch(r'-?\d+', answer)) and int(answer) == expected
                record = {'prompt': prompt, 'answer': answer, 'expected': expected,
                          'verified': verified, 'teacher': args.model, 'source': 'ollama',
                          'split': 'eval' if index % 5 == 0 else 'train',
                          'done_reason': result.get('done_reason')}
            except (OSError, ValueError) as exc:
                record = {'prompt': prompt, 'expected': expected, 'verified': False,
                          'teacher': args.model, 'source': 'ollama', 'error': str(exc)}
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            print(f"{index+1}/{args.count}: verified={record['verified']}", flush=True)


if __name__ == '__main__':
    main()
