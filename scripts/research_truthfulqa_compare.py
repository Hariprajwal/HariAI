"""Explore scores on judge-reviewed outputs, without claiming calibration."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE

from research_truthfulqa_jev import SemanticJev
from hari_ai.runtime.calibration import platt_probability


def main():
    root = Path('checkpoints/truthfulqa_jev_v1')
    rows = [json.loads(x) for x in (root/'generated.jsonl').read_text(encoding='utf-8').splitlines()]
    verdicts = {r['group_id']:r['judgment'] for r in
                (json.loads(x) for x in (root/'judge.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = [r for r in rows if hashlib.sha256(r['question'].encode()).hexdigest() in verdicts]
    checkpoint = torch.load(root/'verifier.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('deployable') is not False:
        raise ValueError('Unexpected research checkpoint.')
    model = SemanticJev()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    params = json.loads((root/'report.json').read_text(encoding='utf-8'))['evaluation']
    llm = Llama(model_path='hari_ai/assets/hariai-chat.gguf', embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_NONE, n_ctx=512, n_batch=512,
                n_threads=4, verbose=False)
    observations = []
    for row in rows:
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        hidden = np.asarray(llm.embed(text, normalize=False, truncate=True), dtype=np.float32)
        feature = torch.from_numpy(np.stack((hidden.mean(axis=0), hidden[-1]))[None])
        feature = (feature-checkpoint['feature_mean'])/checkpoint['feature_std']
        with torch.no_grad():
            decision, confidence = model(feature)
        raw = float(torch.where(decision >= 0, confidence, 1-confidence)[0])
        probability = platt_probability(raw, params['slope'], params['intercept'])
        key = hashlib.sha256(row['question'].encode()).hexdigest()
        observations.append({'group_id':key, 'split':row['split'], 'verdict':verdicts[key],
                             'experimental_score':probability})
    llm.close()
    judged = [r for r in observations if r['verdict'] in ('CORRECT', 'INCORRECT')]
    output = {'warning':'Exploratory; fallible LLM judgments are not ground truth and questions were previously explored.',
              'counts':dict(Counter(r['verdict'] for r in observations)),
              'mean_score_judged':sum(r['experimental_score'] for r in judged)/len(judged) if judged else None,
              'judge_correct_fraction':sum(r['verdict']=='CORRECT' for r in judged)/len(judged) if judged else None,
              'rows':observations}
    path = root/'judge_comparison.json'
    if path.exists():
        raise FileExistsError('Refusing to replace a prior comparison.')
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
