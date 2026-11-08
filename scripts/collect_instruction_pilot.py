"""Collect a bounded sample from the Apache-licensed Smol-Magpie subset."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/instruction_pilot.jsonl'))
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--max-example-bytes', type=int, default=460)
    parser.add_argument('--max-prompt-bytes', type=int, default=160)
    args = parser.parse_args()
    if not 1 <= args.rows <= 2000:
        parser.error('rows must be between 1 and 2000')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, seen = 0, set()
    with args.output.open('x', encoding='utf-8') as sink:
        for offset in range(0, args.rows, 100):
            query = urllib.parse.urlencode({'dataset': 'HuggingFaceTB/smol-smoltalk',
                'config': 'default', 'split': 'train', 'offset': offset,
                'length': min(100, args.rows - offset)})
            with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?' + query, timeout=60) as response:
                batch = json.load(response)
            for item in batch.get('rows', []):
                row = item['row']
                if row.get('source') != 'smol-magpie-ultra-short':
                    continue
                messages = row.get('messages', [])
                if len(messages) < 2 or [m.get('role') for m in messages[:2]] != ['user', 'assistant']:
                    continue
                prompt, answer = (m.get('content', '').strip() for m in messages[:2])
                if (not prompt or not answer or len(prompt.encode()) > args.max_prompt_bytes or
                    len(prompt.encode()) + len(answer.encode()) > args.max_example_bytes):
                    continue
                key = hashlib.sha256(json.dumps([prompt, answer], ensure_ascii=False).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                sink.write(json.dumps({'prompt': prompt, 'answer': answer, 'sha256': key,
                    'source': 'HuggingFaceTB/smol-smoltalk:smol-magpie-ultra-short',
                    'row_index': item.get('row_idx'), 'research_split': 'eval' if int(key[:8], 16) % 10 == 0 else 'train'}, ensure_ascii=False) + '\n')
                count += 1
            print(f'Inspected {min(offset + 100, args.rows)} rows; kept {count}', flush=True)


if __name__ == '__main__':
    main()
