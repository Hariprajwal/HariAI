"""Research probe, not a deployable chat-answer confidence estimator.

Uses labelled TruthfulQA candidate answers, not answers produced by HariAI.
Question groups are split before feature extraction and the held-out results
must not be generalized to user chat. Dataset and generated files stay local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.calibration import calibration_metrics, evaluate_calibrator
from hari_ai.vendor.open_jev.main import Jev, JevConfig, Noul, StateCache

SOURCE_HASH = 'b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c'
MODEL_HASH = 'eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data_path):
    if sha(data_path) != SOURCE_HASH:
        raise ValueError('Dataset differs from the predeclared pinned source.')
    with data_path.open(encoding='utf-8-sig', newline='') as f:
        items = list(csv.DictReader(f))
    unique = {}
    for item in items:
        q, yes, no = (item[k].strip() for k in ('Question', 'Best Answer', 'Best Incorrect Answer'))
        if q and yes and no and yes.casefold() != no.casefold():
            unique.setdefault(q.casefold(), (q, yes, no))
    values = list(unique.values())
    random.Random(901).shuffle(values)
    if len(values) < 700:
        raise ValueError('Unexpectedly few unique labelled questions.')
    # Fixed before model development; every question belongs to exactly one split.
    borders = (int(len(values)*.57), int(len(values)*.70), int(len(values)*.85))
    return [(('train' if i < borders[0] else 'validation' if i < borders[1] else
              'calibration' if i < borders[2] else 'test'), item)
            for i, item in enumerate(values)]


def extract(args):
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
    if sha(args.model) != MODEL_HASH:
        raise ValueError('This research protocol requires the pinned HariAI model.')
    if args.features.exists():
        raise FileExistsError('Refusing to overwrite extracted features.')
    args.features.parent.mkdir(parents=True, exist_ok=True)
    llm = Llama(model_path=str(args.model), embedding=True, pooling_type=LLAMA_POOLING_TYPE_NONE,
                n_ctx=512, n_batch=512, n_threads=4, verbose=False)
    vectors, rows = [], []
    for i, (split, (q, yes, no)) in enumerate(prepare(args.data)):
        # Balanced authored examples are useful to test representation quality,
        # but their prevalence is not the accuracy of real model-generated answers.
        for label, answer in ((1, yes), (0, no)):
            text = f'Question: {q}\nAnswer: {answer}'
            value = np.asarray(llm.embed(text, normalize=False, truncate=True), dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != 576:
                raise ValueError('Unexpected embedding shape.')
            vectors.append(np.stack((value.mean(axis=0), value[-1])))
            rows.append({'split':split, 'label':label, 'group_id':hashlib.sha256(q.encode()).hexdigest()})
        if (i+1) % 100 == 0:
            print(f'Extracted {i+1} question groups', flush=True)
    llm.close()
    np.savez_compressed(args.features, vectors=np.stack(vectors), metadata=json.dumps(rows))
    print(f'Saved {len(rows)} labeled candidate representations.', flush=True)


class SemanticJev(nn.Module):
    """SmolLM state features + the open reference implementation's Jev heads."""
    def __init__(self):
        super().__init__()
        cfg = JevConfig(vocab_size=512, d_model=48, n_heads=4, d_ff=96,
                        n_state_layers=1, n_question_layers=1, n_slots=2,
                        n_readout_layers=1, max_state_len=8, max_question_len=16,
                        path_hash_buckets=32, dropout=0.05)
        self.projection = nn.Linear(576, cfg.d_model)
        self.jev = Jev(cfg)

    def forward(self, x):
        cache = StateCache(self.projection(x), torch.zeros(x.size(0), x.size(1), dtype=torch.bool))
        pooled = self.jev._readout(cache, [Noul('Is the answer correct?')])
        decision = self.jev.noul_head(pooled)
        # The reference head estimates confidence in the selected decision.
        # Detach the shared representation for this auxiliary objective.
        confidence, _ = self.jev.confidence_head(pooled.detach(), 2)
        return decision, confidence


def validate_feature_rows(rows, vectors):
    if len(vectors) != len(rows) or len(rows) % 2:
        raise ValueError('Feature metadata is incomplete.')
    seen = set()
    for i in range(0, len(rows), 2):
        first, second = rows[i:i+2]
        if (first['group_id'] != second['group_id'] or first['group_id'] in seen
                or first['split'] != second['split'] or first['label'] != 1 or second['label'] != 0
                or first['split'] not in ('train', 'validation', 'calibration', 'test')):
            raise ValueError('Invalid question grouping, split, or candidate labels.')
        seen.add(first['group_id'])


def balanced_rows(rows, probabilities, split):
    # One predetermined authored candidate per question. This distribution is
    # deliberately balanced and is not the prevalence of correct chat answers.
    result = []
    pairs = [(rows[i:i+2], probabilities[i:i+2]) for i in range(0, len(rows), 2)
             if rows[i]['split'] == split]
    for idx, (pair, p) in enumerate(pairs):
        selected = idx % 2
        result.append({**pair[selected], 'probability':float(p[selected])})
    return result


def train(args):
    torch.set_num_threads(3)
    torch.manual_seed(901)
    rng = np.random.default_rng(901)
    packed = np.load(args.features, allow_pickle=False)
    raw = packed['vectors']
    rows = json.loads(str(packed['metadata']))
    validate_feature_rows(rows, raw)
    # No rows from calibration or test enter normalization or model selection.
    training = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    valid = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    means = raw[training].mean(axis=(0, 1), keepdims=True)
    stds = raw[training].std(axis=(0, 1), keepdims=True).clip(min=1e-5)
    x = torch.from_numpy((raw-means)/stds)
    y = torch.tensor([r['label'] for r in rows], dtype=torch.float32)
    model = SemanticJev()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.02)
    best = None
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, 41):
        model.train()
        for batch in np.array_split(rng.permutation(training), max(1, len(training)//32)):
            optimizer.zero_grad()
            decision, confidence = model(x[batch])
            selected_correct = ((decision.detach() >= 0) == (y[batch] > .5)).float()
            loss = (F.binary_cross_entropy_with_logits(decision, y[batch]) +
                    F.binary_cross_entropy(confidence.clamp(1e-6, 1-1e-6), selected_correct))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            decision, _ = model(x[valid])
            val_loss = F.binary_cross_entropy_with_logits(decision, y[valid]).item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch >= 7:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        decision, confidence = model(x)
        probabilities = torch.where(decision >= 0, confidence, 1-confidence).numpy()
    cal, test = (balanced_rows(rows, probabilities, split) for split in ('calibration', 'test'))
    result = evaluate_calibrator(cal, test, predeclared=True)
    output = {'description':'TruthfulQA authored answer research only; NOT validated on HariAI chat responses',
              'model_sha256':MODEL_HASH, 'dataset_sha256':SOURCE_HASH,
              'split_question_counts':{s:sum(r['split']==s for r in rows)//2 for s in ('train','validation','calibration','test')},
              'best_validation_epoch':best_epoch, 'best_validation_loss':best_loss,
              'validation_raw_both_answers':calibration_metrics([{**rows[i], 'probability':float(probabilities[i])} for i in valid]),
              'evaluation':result, 'deployable':False,
              'reason':'Held-out candidate answers were written by dataset authors, not generated by the chat model.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists() or args.checkpoint.exists():
        raise FileExistsError('Refusing to replace an existing evaluation.')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state':best, 'feature_mean':torch.from_numpy(means),
                'feature_std':torch.from_numpy(stds), 'model_sha256':MODEL_HASH,
                'dataset_sha256':SOURCE_HASH, 'deployable':False}, args.checkpoint)
    args.report.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'train'])
    parser.add_argument('--data', type=Path, default=Path('data/truthfulqa_d71c110.csv'))
    parser.add_argument('--model', type=Path, default=Path('hari_ai/assets/hariai-chat.gguf'))
    parser.add_argument('--features', type=Path, default=Path('checkpoints/truthfulqa_jev_v1/features.npz'))
    parser.add_argument('--report', type=Path, default=Path('checkpoints/truthfulqa_jev_v1/report.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/truthfulqa_jev_v1/verifier.pt'))
    args = parser.parse_args()
    (extract if args.action == 'extract' else train)(args)


if __name__ == '__main__':
    main()
