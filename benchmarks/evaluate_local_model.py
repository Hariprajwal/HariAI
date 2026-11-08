"""Fresh small smoke set for standalone GGUF files; not a general benchmark.

Run after adaptation. These prompts are not used by the training script. Exact
answer checks deliberately fail extra prose. Qualitative outputs are retained
for human review and are not converted into a made-up aggregate quality score.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from llama_cpp import Llama

EXACT = [
    ('Reply with only the number: 9 + 6.', '15'),
    ('Reply with only the number: 18 - 7.', '11'),
    ('Reply with only the number: 6 times 4.', '24'),
    ('Reply with only the number of minutes in one hour.', '60'),
    ('Reply with only the capital of Italy.', 'rome'),
    ('Reply with only the capital of Germany.', 'berlin'),
    ('Reply with only the opposite of slow.', 'fast'),
    ('Reply with only the plural of child.', 'children'),
    ('Reply with only the next number: 2, 4, 6, 8, ...', '10'),
    ('Reply with only the chemical symbol of oxygen.', 'o'),
    ('Reply with only the number of sides of a rectangle.', '4'),
    ('Reply with only the first item: pear, orange, lemon.', 'pear'),
]
QUALITATIVE = [
    'Hello, how are you?',
    'Describe yourself and name your creator.',
    'Explain what a Python tuple is in two sentences.',
    'Write a Python function that returns the largest of three numbers.',
    'Give two practical tips for keeping a desk organized.',
    'Summarize this: Mira watered her garden every morning. After a week, the seeds sprouted.',
    'Which language model did HariAI start from?',
    'Can you guarantee that all your answers are correct?',
]


def evaluate(path):
    model = Llama(model_path=str(path), n_ctx=2048, n_threads=4, verbose=False, chat_format='chatml', seed=42)
    start = time.monotonic()
    results = []
    for question, expected in EXACT + [(q, None) for q in QUALITATIVE]:
        answer = model.create_chat_completion(messages=[{'role': 'system', 'content':
            'You are HariAI, a helpful assistant created by K R Hari Prajwal. Answer clearly and briefly. Be honest when you do not know.'},
            {'role': 'user', 'content': question}], max_tokens=160, temperature=0)['choices'][0]['message']['content']
        cleaned = answer.lower().strip().strip('.!').strip()
        results.append({'prompt': question, 'answer': answer, 'expected': expected,
                        'exact_pass': cleaned == expected if expected is not None else None})
    model.close()
    return {'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'model_bytes': path.stat().st_size,
        'exact_pass': sum(r['exact_pass'] is True for r in results), 'exact_total': len(EXACT),
        'qualitative_count': len(QUALITATIVE), 'seconds': round(time.monotonic()-start, 2), 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'candidate': evaluate(args.model), 'limitations': 'Small, original smoke set; exact matches do not establish broad assistant quality.'}
    if args.baseline:
        report['baseline'] = evaluate(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: {k:v for k,v in value.items() if k != 'results'} if isinstance(value, dict) else value
                      for key,value in report.items()}))


if __name__ == '__main__':
    main()
