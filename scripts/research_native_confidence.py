"""Probe model-native token likelihood on independently keyed ARC questions.

Research only. This protocol uses specially formatted multiple-choice prompts;
it cannot establish confidence calibration for unrestricted chat.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings

DATA_SHA = '4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177'
MODEL_SHA = 'eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_choice(answer):
    match = re.fullmatch(r'\s*(?:(?:the\s+)?answer\s+is\s+)?\(?([A-D])\)?[.!]?\s*', answer, re.I)
    return match.group(1).upper() if match else None


def question_groups(path):
    if digest(path) != DATA_SHA:
        raise ValueError('Unexpected ARC dataset version.')
    questions = []
    seen = set()
    for row in pq.read_table(path).to_pylist():
        options, original = row['choices']['text'], row['choices']['label']
        key = row['answerKey']
        group = ' '.join(row['question'].casefold().split())
        if len(options) != 4 or len(original) != 4 or key not in original or group in seen:
            continue
        seen.add(group)
        prompt = row['question'].strip()+'\n'+ '\n'.join(
            f'{label}. {option}' for label, option in zip('ABCD', options))+'\nAnswer with only the letter.'
        questions.append({'prompt':prompt, 'question':row['question'].strip(), 'options':options,
                          'expected':'ABCD'[original.index(key)],
                          'group_id':hashlib.sha256(group.encode()).hexdigest()})
    random.Random(401).shuffle(questions)
    # Every split is assigned before model generation or score inspection.
    for i, question in enumerate(questions):
        question['split'] = 'development' if i < 300 else 'calibration' if i < 600 else 'test'
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/arc_easy_210d026_test.parquet'))
    parser.add_argument('--model', type=Path, default=Path('hari_ai/assets/hariai-chat.gguf'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/native_confidence_arc_pilot/answers.jsonl'))
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    if digest(args.model) != MODEL_SHA:
        raise ValueError('Unexpected model; calibration depends on exact weights.')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output file.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    questions = question_groups(args.data)[:args.limit]
    system = ChatEngine(settings=ChatSettings()).system_prompt
    model = Llama(model_path=str(args.model), n_ctx=2048, n_threads=4, verbose=False,
                  chat_format='chatml', logits_all=True, seed=42)
    counts = Counter()
    with args.output.open('x', encoding='utf-8') as out:
        for i, task in enumerate(questions):
            response, logps, finish = [], [], None
            stream = model.create_chat_completion(messages=[{'role':'system', 'content':system},
                       {'role':'user', 'content':task['prompt']}], max_tokens=192, temperature=.2,
                       top_p=.9, repeat_penalty=1.1, logprobs=True, top_logprobs=1, stream=True)
            for chunk in stream:
                choice = chunk['choices'][0]
                response.append(choice.get('delta', {}).get('content') or '')
                logps.extend(v['logprob'] for v in ((choice.get('logprobs') or {}).get('content') or []))
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
            answer = ''.join(response).strip()
            selected = parse_choice(answer) if finish == 'stop' else None
            label = int(selected == task['expected']) if selected else None
            score = math.exp(statistics.mean(logps)) if logps else None
            counts['unscorable' if label is None else 'correct' if label else 'incorrect'] += 1
            out.write(json.dumps({**task, 'answer':answer, 'finish_reason':finish, 'label':label,
                                  'token_likelihood':score}, ensure_ascii=False)+'\n')
            out.flush()
            if (i+1) % 20 == 0:
                print(f'Generated {i+1}/{len(questions)}: {dict(counts)}', flush=True)
    model.close()
    print(json.dumps({'count':len(questions), **counts}, indent=2), flush=True)


if __name__ == '__main__':
    main()
