"""Measure native P(Yes) self-assessment on prior independently graded answers.

This is an explored arithmetic dataset, so no results can be promoted as a
fresh held-out test. The normalized Yes/No token score is not a probability of
answer correctness without suitable calibration and held-out evidence.
"""
import json
import math
import sys
from pathlib import Path

from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.calibration import evaluate_calibrator
from hari_ai.runtime.answer_confidence import sha256_file
from hari_ai.runtime.verified_arithmetic import verify_arithmetic

MODEL_SHA = 'eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94'


def main():
    base = Path('checkpoints/jev_calibration_fresh_v2')
    outpath = Path('checkpoints/native_self_assessment_arithmetic/report.json')
    if outpath.exists():
        raise FileExistsError('Refusing to replace a previous evaluation.')
    metadata = json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    model_path = Path('hari_ai/assets/hariai-chat.gguf')
    if metadata.get('model_sha256') != MODEL_SHA or sha256_file(model_path) != MODEL_SHA:
        raise ValueError('The model or collection does not match the pinned weights.')
    rows = [json.loads(line) for line in (base/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['label'] in (0, 1)]
    for row in rows:
        checked = verify_arithmetic(row['question'], row['answer'])
        if row['finish_reason'] != 'stop' or checked is None or row['label'] != int(checked):
            raise ValueError('A scored answer does not match its independent arithmetic label.')
    model = Llama(model_path=str(model_path), n_ctx=512, n_batch=512,
                  n_threads=4, logits_all=True, verbose=False)
    yes = model.tokenize(b'Yes', add_bos=False)
    no = model.tokenize(b'No', add_bos=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError('The tested model does not encode each answer as one token.')
    scored = []
    for i, row in enumerate(rows):
        text = ('<|im_start|>system\nJudge whether the proposed answer is correct. '
                'Answer Yes or No.<|im_end|>\n<|im_start|>user\n'
                f"Question: {row['question']}\nProposed answer: {row['answer']}\n"
                'Is the proposed answer correct?<|im_end|>\n<|im_start|>assistant\n')
        tokens = model.tokenize(text.encode('utf-8'), special=True)
        if len(tokens) >= 512:
            raise ValueError('Self-assessment prompt exceeds context length.')
        model.reset()
        model.eval(tokens)
        logits = model.scores[model.n_tokens-1]
        difference = max(-60, min(60, float(logits[yes[0]] - logits[no[0]])))
        score = 1/(1+math.exp(-difference))
        scored.append({'group_id':row['group_id'], 'split':row['split'],
                       'label':row['label'], 'probability':score})
        if (i+1) % 50 == 0:
            print(f'Scored {i+1}/{len(rows)} previous model answers', flush=True)
    model.close()
    cal = [r for r in scored if r['split']=='calibration']
    test = [r for r in scored if r['split']=='test']
    result = evaluate_calibrator(cal, test, predeclared=False)
    output = {'method':'native_model_yes_no_logits', 'test_already_explored':True,
              'data':'actual completed model outputs with exact independent arithmetic checks',
              'supported_as_general_chat_confidence':False, 'evaluation':result}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
