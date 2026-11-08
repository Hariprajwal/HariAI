"""Measure token-likelihood confidence against independently checked arithmetic.

Research only: no confidence value is installed or used by the chat runtime.
Scores measure an exact-answer task, not correctness on general conversations.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import torch
from llama_cpp import Llama

SYSTEM = ('You are HariAI, a helpful assistant created by K R Hari Prajwal. '
          'Answer clearly and briefly. Be honest when you do not know.')


def metrics(rows):
    if not rows:
        return None
    n = len(rows)
    brier = sum((r['estimate'] - r['correct']) ** 2 for r in rows) / n
    ece = 0.0
    bins = []
    for low in range(10):
        group = [r for r in rows if min(9, int(r['estimate'] * 10)) == low]
        if group:
            accuracy = sum(r['correct'] for r in group) / len(group)
            confidence = sum(r['estimate'] for r in group) / len(group)
            ece += len(group) / n * abs(confidence - accuracy)
            bins.append({'bin': low, 'count': len(group), 'accuracy': accuracy, 'confidence': confidence})
    return {'count': n, 'correct': sum(r['correct'] for r in rows), 'brier': brier, 'ece_10': ece, 'bins': bins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('hari_ai/assets/hariai-chat.gguf'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/confidence_evaluation/report.json'))
    parser.add_argument('--per-operation', type=int, default=60)
    args = parser.parse_args()
    rng = random.Random(42)
    # Unique operand pairs are split before inference to avoid train/test overlap.
    pairs = rng.sample([(a, b) for a in range(2, 41) for b in range(1, 21)], args.per_operation * 2)
    tasks = []
    for i, (a, b) in enumerate(pairs):
        operation = '+' if i < args.per_operation else '-'
        expected = a + b if operation == '+' else a - b
        tasks.append({'prompt': f'What is {a} {operation} {b}? Answer with only the number.',
                      'expected': expected, 'operation': operation,
                      'split': 'calibration' if (i % args.per_operation) < int(args.per_operation * .7) else 'test'})
    model = Llama(model_path=str(args.model), n_ctx=2048, n_threads=4, verbose=False,
                  chat_format='chatml', logits_all=True, seed=42)
    rows = []
    for i, task in enumerate(tasks):
        stream = model.create_chat_completion(messages=[{'role':'system', 'content': SYSTEM},
            {'role':'user', 'content':task['prompt']}], max_tokens=32, temperature=.2,
            top_p=.9, repeat_penalty=1.1, logprobs=True, top_logprobs=1, stream=True)
        pieces, logps = [], []
        for chunk in stream:
            choice = chunk['choices'][0]
            pieces.append(choice.get('delta', {}).get('content') or '')
            logps.extend(v['logprob'] for v in ((choice.get('logprobs') or {}).get('content') or []))
        answer = ''.join(pieces).strip()
        # Exact-format success means a single integer, possibly followed by a period.
        match = re.fullmatch(r'([+-]?\d+)\.?', answer)
        correct = bool(match and int(match.group(1)) == task['expected'])
        if not logps:
            raise RuntimeError('Runtime did not return token log probabilities; no score can be evaluated.')
        mean_logprob = statistics.mean(logps)
        rows.append({**task, 'answer':answer, 'mean_logprob':mean_logprob,
                     'estimate':math.exp(mean_logprob), 'correct':int(correct)})
        if (i + 1) % 20 == 0:
            print(json.dumps({'completed': i+1, 'total':len(tasks), 'exact_correct':sum(r['correct'] for r in rows)}), flush=True)
    model.close()
    train = [r for r in rows if r['split'] == 'calibration']
    test = [r for r in rows if r['split'] == 'test']
    counts = Counter(r['correct'] for r in train)
    # Fit a one-dimensional Platt calibrator only with both observed classes.
    calibrated = None
    if min(counts[0], counts[1]) >= 5:
        x = torch.tensor([[r['mean_logprob']] for r in train], dtype=torch.float64)
        y = torch.tensor([[r['correct']] for r in train], dtype=torch.float64)
        layer = torch.nn.Linear(1, 1, dtype=torch.float64)
        optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=100, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(layer(x), y)
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            z = torch.tensor([[r['mean_logprob']] for r in test], dtype=torch.float64)
            estimates = torch.sigmoid(layer(z)).flatten().tolist()
        calibrated_rows = [{**row, 'estimate': probability} for row, probability in zip(test, estimates)]
        calibrated = {'method':'Platt scaling of mean token log probability',
                      'slope':float(layer.weight.item()), 'intercept':float(layer.bias.item()),
                      'heldout':metrics(calibrated_rows)}
    baseline_rate = sum(r['correct'] for r in train) / len(train)
    baseline = metrics([{**r, 'estimate':baseline_rate} for r in test])
    report = {'scope':'single-turn exact-format addition/subtraction; 32 generated token limit',
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(), 'examples':len(rows),
        'calibration_counts':dict(counts), 'raw_likelihood_heldout':metrics(test),
        'constant_baseline_heldout':baseline, 'fitted_calibration':calibrated,
        'limitations':'Token likelihood is not correctness. Correct means output is exactly the expected integer; other text is scored as failure. Synthetic in-domain test, not general chat calibration.',
        'rows':rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
