"""Train a tiny native-model decision probe on generated choice outcomes.

Research only: uses a constrained multiple-choice protocol, not free chat.
Train/validation questions belong to the development partition; calibration
and final test partitions must never be used for selecting probe parameters.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_native_confidence import MODEL_SHA, digest, question_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/arc_easy_210d026_test.parquet'))
    p.add_argument('--model', type=Path, default=Path('hari_ai/assets/hariai-chat.gguf'))
    p.add_argument('--decisions', type=Path, default=Path('checkpoints/native_choice_v1/development.json'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/native_choice_v1/probe_development.npz'))
    args = p.parse_args()
    if digest(args.model) != MODEL_SHA or args.output.exists():
        raise ValueError('Unexpected model or existing output.')
    decision = json.loads(args.decisions.read_text(encoding='utf-8'))
    if decision['protocol'] != 'native_constrained_choice_v1' or decision['model_sha256'] != MODEL_SHA:
        raise ValueError('The decision protocol or model weights do not match.')
    rows = decision['rows']
    groups = {x['group_id']:x for x in question_groups(args.data) if x['split']=='development'}
    if len(rows) != 300 or len({r['group_id'] for r in rows}) != 300 or any(r['group_id'] not in groups for r in rows):
        raise ValueError('Exactly 300 unique development decisions are required.')
    model = Llama(model_path=str(args.model), embedding=True, pooling_type=LLAMA_POOLING_TYPE_NONE,
                  n_ctx=512, n_batch=512, n_threads=4, verbose=False)
    features, labels = [], []
    for i, row in enumerate(rows):
        group = groups[row['group_id']]
        choice = row['choice']
        if choice not in ('A', 'B', 'C', 'D') or row['label'] != int(choice == group['expected']):
            raise ValueError('Decision label does not match the independent answer key.')
        answer = group['options']['ABCD'.index(choice)]
        text = f"{group['prompt']}\nSelected answer: {choice}. {answer}"
        tokens = model.tokenize(text.encode('utf-8'), special=True)
        if len(tokens) >= 512:
            raise ValueError('Probe input exceeds the predeclared context window.')
        hidden = np.asarray(model.embed(text, normalize=False, truncate=False), dtype=np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != 576:
            raise ValueError('Unexpected model embedding dimensions.')
        features.append(np.concatenate([hidden.mean(0), hidden[-1], np.asarray(row['probabilities'], dtype=np.float32)]))
        labels.append(row['label'])
        if (i+1) % 50 == 0:
            print(f'Extracted {i+1}/300 decision representations', flush=True)
    model.close()
    x = np.stack(features)
    y = np.asarray(labels)
    mean = x[:200].mean(0)
    scale = np.maximum(x[:200].std(0), 1e-5)
    x = (x-mean)/scale
    results = []
    best_classifier = None
    best_brier = float('inf')
    for strength in (0.001, 0.01, 0.1, 1.0):
        classifier = LogisticRegression(C=strength, max_iter=1000, random_state=401)
        classifier.fit(x[:200], y[:200])
        pred = classifier.predict_proba(x[200:])[:,1]
        brier = float(brier_score_loss(y[200:],pred))
        results.append({'C':strength, 'auc':float(roc_auc_score(y[200:],pred)), 'brier':brier})
        if brier < best_brier:
            best_classifier = classifier
            best_brier = brier
    np.savez_compressed(args.output, features=np.stack(features), labels=y,
                        groups=np.array([r['group_id'] for r in rows]),
                        results=json.dumps(results), probe_weights=best_classifier.coef_,
                        probe_bias=best_classifier.intercept_, feature_mean=mean, feature_scale=scale,
                        deployment_allowed=np.array(False))
    print(json.dumps({'training_examples':200, 'validation_examples':100,
                      'training_correct':int(y[:200].sum()), 'validation_correct':int(y[200:].sum()),
                      'candidates':results, 'deployment_allowed':False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
