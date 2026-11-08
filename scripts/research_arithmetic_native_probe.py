"""Development-only native probe on real, independently checked math replies.

Both original splits have already been inspected. These results cannot serve
as fresh calibration or final-test evidence, and this probe is not deployable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.answer_confidence import sha256_file
from hari_ai.runtime.calibration import brier_improvement_interval
from hari_ai.runtime.verified_arithmetic import verify_arithmetic

MODEL_SHA = 'eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('checkpoints/native_arithmetic_probe_v1/report.json'))
    args = parser.parse_args()
    model_path = Path('hari_ai/assets/hariai-chat.gguf')
    source = Path('checkpoints/jev_calibration_fresh_v2')
    output = args.output
    if output.exists():
        raise FileExistsError('Choose a new experiment directory.')
    meta = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if sha256_file(model_path) != MODEL_SHA or meta['model_sha256'] != MODEL_SHA:
        raise ValueError('Model or collected answers do not match the pinned weights.')
    rows = [json.loads(line) for line in (source/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['label'] in (0, 1)]
    if len({r['group_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate question groups.')
    model = Llama(model_path=str(model_path), embedding=True, pooling_type=LLAMA_POOLING_TYPE_NONE,
                  n_ctx=512, n_batch=512, n_threads=4, verbose=False)
    features = []
    for i, row in enumerate(rows):
        result = verify_arithmetic(row['question'], row['answer'])
        if row['finish_reason'] != 'stop' or result is None or int(result) != row['label']:
            raise ValueError('A stored label does not match the independent arithmetic check.')
        prompt = f"Question: {row['question']}\nAnswer: {row['answer']}"
        hidden = np.asarray(model.embed(prompt, normalize=False, truncate=False), dtype=np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != 576:
            raise ValueError('Unexpected model hidden representation.')
        features.append(np.concatenate([hidden.mean(0), hidden[-1]]))
        if (i+1) % 50 == 0:
            print(f'Extracted {i+1}/{len(rows)} actual response representations', flush=True)
    model.close()
    x = np.stack(features)
    y = np.asarray([r['label'] for r in rows])
    train = np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
    valid = np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-5)
    x = (x-mean)/std
    base_rate = float(y[train].mean())
    candidates = []
    for strength in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=strength, max_iter=1000, random_state=1108)
        clf.fit(x[train], y[train])
        probs = clf.predict_proba(x[valid])[:,1]
        evaluation = [{'group_id':rows[index]['group_id'], 'label':int(y[index]), 'probability':float(prob)}
                      for index,prob in zip(valid, probs)]
        interval = brier_improvement_interval(evaluation, baseline=base_rate, slope=1.0, intercept=0.0)
        candidates.append({'C':strength, 'auc':float(roc_auc_score(y[valid],probs)),
                           'brier':float(brier_score_loss(y[valid],probs)),
                           'brier_gain_interval_95':[interval['lower_95'], interval['upper_95']]})
    report = {'previously_explored':True, 'source':'model_generated_exactly_checked_arithmetic',
              'training_examples':len(train), 'training_correct':int(y[train].sum()),
              'validation_examples':len(valid), 'validation_correct':int(y[valid].sum()),
              'constant_baseline_brier':float(np.mean((base_rate-y[valid])**2)),
              'candidates':candidates, 'deployable':False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
