"""Test confidence for a constrained four-choice decision made by HariAI.

Research only: the decoder restricts output to one letter, unlike normal
chat. The benchmark is locally stored under its own licence; no questions or
scores from it are distributed with HariAI.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.calibration import calibration_metrics
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings
from research_native_confidence import DATA_SHA, MODEL_SHA, digest, question_groups


def choice_weights(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Exactly four finite option logits are required.')
    weights = np.exp(values-values.max())
    return weights/weights.sum()


def restore_order(scores, offset):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (4,) or offset not in range(4) or not np.isfinite(values).all():
        raise ValueError('Expected four finite option scores and a valid rotation.')
    restored = np.zeros(4, dtype=np.float64)
    for new, original in enumerate((i+offset)%4 for i in range(4)):
        restored[original] = values[new]
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('development', 'calibration', 'test'), required=True)
    parser.add_argument('--data', type=Path, default=Path('data/arc_easy_210d026_test.parquet'))
    parser.add_argument('--model', type=Path, default=Path('hari_ai/assets/hariai-chat.gguf'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rotate', action='store_true', help='Average over four cyclic option positions')
    parser.add_argument('--limit', type=int, default=300)
    args = parser.parse_args()
    if digest(args.model) != MODEL_SHA or digest(args.data) != DATA_SHA:
        raise ValueError('Unexpected language-model weights or benchmark revision.')
    if args.output.exists():
        raise FileExistsError('Refusing to replace previous results.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.limit <= 300 or (args.split != 'development' and args.limit != 300):
        raise ValueError('Partial runs are supported only for development.')
    data = [r for r in question_groups(args.data) if r['split'] == args.split][:args.limit]
    if len(data) != args.limit:
        raise ValueError('Predeclared split contains fewer question groups than requested.')
    system = ChatEngine(settings=ChatSettings()).system_prompt
    model = Llama(model_path=str(args.model), n_ctx=512, n_batch=512, n_threads=4, verbose=False)
    token_ids = [model.tokenize(s.encode(), add_bos=False) for s in 'ABCD']
    if any(len(t) != 1 for t in token_ids) or len({t[0] for t in token_ids}) != 4:
        raise ValueError('Decision labels must be distinct single tokens.')
    token_ids = [t[0] for t in token_ids]
    rows = []
    for index, row in enumerate(data):
        probabilities = np.zeros(4, dtype=np.float64)
        offsets = range(4) if args.rotate else range(1)
        for offset in offsets:
            order = [(i+offset)%4 for i in range(4)]
            choices = '\n'.join(f'{label}. {row["options"][original]}'
                                for label, original in zip('ABCD', order))
            question = f'{row["question"]}\n{choices}\nAnswer with only the letter.'
            prompt = (f'<|im_start|>system\n{system}<|im_end|>\n'
                      f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n')
            tokens = model.tokenize(prompt.encode('utf-8'), special=True)
            if len(tokens) >= 512:
                raise ValueError('A benchmark prompt exceeded the predeclared context.')
            model.reset()
            model.eval(tokens)
            # llama-cpp-python 0.3.8 exposes the final decoded logits here even
            # without the large per-token logits_all buffer.
            logits = np.ctypeslib.as_array(model._ctx.get_logits(), shape=(model.n_vocab(),))
            selected_logits = np.array([float(logits[t]) for t in token_ids], dtype=np.float64)
            scores = choice_weights(selected_logits)
            probabilities += restore_order(scores, offset)/len(offsets)
        choice = int(np.argmax(probabilities))
        rows.append({'group_id':row['group_id'], 'split':row['split'],
                     'label':int('ABCD'[choice] == row['expected']),
                     'probability':float(probabilities[choice]),
                     'choice':'ABCD'[choice], 'probabilities':probabilities.tolist()})
        if (index+1) % 50 == 0:
            print(f'Scored {index+1}/{len(data)} {args.split} questions', flush=True)
    model.close()
    args.output.write_text(json.dumps({'protocol':'native_rotated_choice_v1' if args.rotate else 'native_constrained_choice_v1',
                           'source':'arc_easy_author_answer_keys', 'model_sha256':MODEL_SHA,
                           'dataset_sha256':DATA_SHA, 'rows':rows}, indent=2), encoding='utf-8')
    result = calibration_metrics(rows)
    print(json.dumps({k:v for k,v in result.items() if k != 'bins'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
