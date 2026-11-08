"""Download a bounded, attributed FineWeb-Edu sample for a research pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/fineweb_pilot.jsonl'))
    parser.add_argument('--documents', type=int, default=400)
    args = parser.parse_args()
    if not 1 <= args.documents <= 1000:
        parser.error('documents must be between 1 and 1000')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://datasets-server.huggingface.co/rows?'
    count = 0
    seen = set()
    with args.output.open('x', encoding='utf-8') as sink:
        for offset in range(0, args.documents, 100):
            query = urllib.parse.urlencode({'dataset': 'HuggingFaceFW/fineweb-edu',
                'config': 'sample-10BT', 'split': 'train', 'offset': offset,
                'length': min(100, args.documents - offset)})
            with urllib.request.urlopen(url + query, timeout=60) as response:
                batch = json.load(response)
            for item in batch.get('rows', []):
                row = item['row']
                text = row.get('text', '').strip()
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if len(text) < 200 or digest in seen:
                    continue
                seen.add(digest)
                record = {'text': text, 'sha256': digest, 'source': 'HuggingFaceFW/fineweb-edu',
                    'config': 'sample-10BT', 'split': 'train', 'row_index': item.get('row_idx'),
                    'id': row.get('id'), 'url': row.get('url'), 'file_path': row.get('file_path'),
                    'dataset_license': 'ODC-By-1.0',
                    'research_split': 'eval' if int(digest[:8], 16) % 10 == 0 else 'train'}
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
            print(f'Collected {count} documents', flush=True)
    print(f'Saved {count} documents to {args.output}')


if __name__ == '__main__':
    main()
