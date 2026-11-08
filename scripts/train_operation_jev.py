"""Train Open-JEV on an explicitly bounded synthetic operation-review task.

The deterministic labeling policy is the teacher. This is a research example,
not a replacement for application permissions or a general decision model.
All exact semantic records are split before training; no evaluation records
are used to update weights or select the fixed 400-step checkpoint.
"""
import itertools
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hari_ai.vendor.open_jev.main import Jev, JevConfig, Choice
from hari_ai.runtime.openjev import canonical_state

OPTIONS = ['proceed', 'review', 'stop']
QUESTION = 'Classify the operation under the training policy.'
OPERATIONS = ['read', 'list', 'create', 'update', 'delete', 'format', 'overwrite', 'export']


def label(record):
    if record['authorized'] == 'no':
        return 'stop'
    if record['operation'] in ('read', 'list'):
        return 'proceed'
    if record['scope'] != 'single':
        return 'review'
    if record['operation'] in ('delete', 'format', 'overwrite') and record['backup'] == 'no':
        return 'review'
    return 'proceed'


def main():
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(2)
    folder = Path('checkpoints/operation_jev')
    folder.mkdir(parents=True, exist_ok=True)
    records = [dict(zip(['operation', 'scope', 'authorized', 'backup'], values)) for values in
        itertools.product(OPERATIONS, ['single', 'batch', 'all'], ['yes', 'no'], ['yes', 'no'])]
    # Stratification prevents a rare choice from vanishing from a small split.
    train, test = [], []
    for option in OPTIONS:
        group = [r for r in records if label(r) == option]
        random.shuffle(group)
        cut = max(1, round(len(group) * .25))
        test.extend(group[:cut])
        train.extend(group[cut:])
    cfg = JevConfig(vocab_size=512, d_model=48, n_heads=4, d_ff=96, dropout=0.0,
        n_state_layers=1, n_question_layers=1, n_readout_layers=2, n_slots=2,
        max_state_len=32, max_question_len=24, path_hash_buckets=128, max_sibling_index=8)
    model = Jev(cfg)
    question = Choice(QUESTION, options=OPTIONS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.time()
    for step in range(400):
        rows = [canonical_state(row) for row in random.sample(train, 24)]
        target = torch.full((len(rows), len(OPTIONS)), .02)
        for i, row in enumerate(rows):
            target[i, OPTIONS.index(label(row))] = .96
        probs, _ = model.logits(rows, [question])[0]
        # Only choice outputs are trained. The reference confidence head remains
        # untrained and is deliberately absent from the production result.
        loss = -(target * probs.log()).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(json.dumps({'step': step+1, 'loss': float(loss), 'seconds': round(time.time()-start)}), flush=True)
    model.eval()
    with torch.no_grad():
        probs, _ = model.logits([canonical_state(row) for row in test], [question])[0]
        reverse_probs, _ = model.logits([canonical_state(dict(reversed(list(row.items())))) for row in test], [question])[0]
        reordered, _ = model.logits([canonical_state(row) for row in test], [Choice(QUESTION, options=OPTIONS[::-1])])[0]
    results = [{'state': row, 'expected': label(row), 'predicted': OPTIONS[int(probs[i].argmax())],
        'probabilities': dict(zip(OPTIONS, probs[i].tolist()))} for i, row in enumerate(test)]
    accuracy = sum(r['expected'] == r['predicted'] for r in results) / len(test)
    invariant = bool(torch.allclose(probs, reverse_probs, atol=1e-5)
                     and torch.allclose(probs, reordered.flip(-1), atol=1e-5))
    report = {'creator': 'K R Hari Prajwal', 'scope': 'operation_policy_v1',
        'training_records': len(train), 'test_records': len(test), 'seed': 17, 'steps': 400,
        'test_accuracy': accuracy, 'majority_baseline': max(sum(label(r) == o for r in test) for o in OPTIONS)/len(test),
        'ordering_invariant': invariant, 'passed': accuracy >= .85 and invariant,
        'confidence_trained': False, 'calibrated': False,
        'limitations': 'Synthetic finite schema only. No general language decisions or security enforcement.',
        'results': results}
    checkpoint = folder / 'operation_jev.pt'
    torch.save({'config': asdict(cfg), 'model_state_dict': model.state_dict(), 'training_steps': 400,
                'scope': 'operation_policy_v1', 'test_accuracy': accuracy, 'passed': report['passed']}, checkpoint)
    report['checkpoint_bytes'] = checkpoint.stat().st_size
    (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (folder/'split.json').write_text(json.dumps({'train':train,'test':test}, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
