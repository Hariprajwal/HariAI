"""Train a Jev answer verifier, fit calibration, evaluate a pre-split holdout.

Requires locally collected synthetic-question/model-answer records. Synthetic
training operands exclude every problem in the fresh calibration/test set.
Artifacts are experimental unless the runtime independently accepts the gate.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hari_ai.runtime.answer_confidence import (DOMAIN, ENCODING, SCORE_SEMANTICS, OPTIONS, QUESTION, sha256_file,
    stated_integer, answer_probability_from_head, arithmetic_state)
from hari_ai.runtime.calibration import evaluate_calibrator
from hari_ai.runtime.openjev import canonical_state
from hari_ai.vendor.open_jev.main import Choice, Jev, JevConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('checkpoints/jev_calibration_fresh_v1'))
    parser.add_argument('--prior', type=Path, default=Path('checkpoints/confidence_evaluation/report.json'))
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--output', type=Path, help='Fresh directory for the verifier and calibration artifact')
    args = parser.parse_args()
    output_dir = args.output or args.data
    output_dir.mkdir(parents=True, exist_ok=True)
    # A second architecture evaluated on a previously inspected test set is
    # development research, never an independent promotion test.
    independent_test = not (args.data/'answer_confidence.json').exists()
    checkpoint = output_dir/'verifier.pt'
    if checkpoint.exists() or (output_dir/'answer_confidence.json').exists():
        raise FileExistsError('Refusing to replace a verifier or calibration artifact; choose a fresh output directory.')
    metadata = json.loads((args.data/'metadata.json').read_text(encoding='utf-8'))
    fresh = [json.loads(line) for line in (args.data/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(fresh) != metadata['requested'] or not (args.data/'summary.json').is_file():
        raise ValueError('Fresh collection is incomplete; no evaluation was run.')
    if metadata.get('problem_grouping') != 'commutative_addition_v1':
        raise ValueError('Fresh collection does not prevent commutative split leakage.')
    if metadata.get('source') != 'model_generated_independently_graded' or metadata.get('predeclared_split') is not True:
        raise ValueError('Fresh data needs the model-generated, predeclared collection protocol.')
    generator = random.Random(metadata['seed'])
    population = [(a,op,b) for a in range(30) for b in range(30) for op in ('+','-') if op != '+' or a <= b]
    original_questions = generator.sample(population,metadata['requested'])
    original_assignment = list(range(len(original_questions)))
    generator.shuffle(original_assignment)
    original_calibration = set(original_assignment[:len(original_assignment)//2])
    all_ids = {r['group_id'] for r in fresh}
    if len(all_ids) != len(fresh):
        raise ValueError('Fresh problem groups must be unique.')
    holdout_pairs = set()
    for index,row in enumerate(fresh):
        m = re.fullmatch(r'What is (\d+) ([+-]) (\d+)\? Answer with only the number\.', row['question'])
        if not m:
            raise ValueError('Fresh question does not match the supported task.')
        a,op,b = int(m[1]),m[2],int(m[3])
        if ((a,op,b) != original_questions[index]
                or row['split'] != ('calibration' if index in original_calibration else 'test')
                or row['group_id'] != hashlib.sha256(row['question'].encode('utf-8')).hexdigest()):
            raise ValueError('Fresh questions or the predeclared split changed.')
        expected = a+b if op == '+' else a-b
        value = stated_integer(row['question'],row['answer'])
        if row['finish_reason'] == 'length':
            value = None
        label = None if value is None else int(value == expected)
        if row['expected'] != expected or row['label'] != label:
            raise ValueError('Fresh arithmetic label does not match the independently computed result.')
        holdout_pairs.add((int(m[1]),m[2],int(m[3])))
        if m[2] == '+':
            holdout_pairs.add((int(m[3]),m[2],int(m[1])))
    rng = random.Random(901)
    torch.manual_seed(901)
    torch.set_num_threads(2)
    possibilities = [(a,op,b) for a in range(0,45) for b in range(0,25) for op in ('+','-')
                     if (a,op,b) not in holdout_pairs]
    synthetic = []
    for _ in range(3200):
        a,op,b = rng.choice(possibilities)
        expected = a+b if op == '+' else a-b
        correct = bool(rng.randrange(2))
        number = expected if correct else expected+rng.choice([-9,-5,-2,-1,1,2,5,9])
        question = f'What is {a} {op} {b}? Answer with only the number.'
        answer = rng.choice([str(number), f'The answer is {number}.', f'{a} {op} {b} = {number}.'])
        synthetic.append((canonical_state(arithmetic_state(question,answer)),correct))
    prior = json.loads(args.prior.read_text(encoding='utf-8'))
    if prior['model_sha256'] != metadata['model_sha256']:
        raise ValueError('Prior data came from different language-model weights.')
    real_train = []
    for row in prior['rows']:
        m = re.fullmatch(r'What is (\d+) ([+-]) (\d+)\? Answer with only the number\.', row['prompt'])
        if not m or (int(m[1]),m[2],int(m[3])) in holdout_pairs:
            continue
        value = stated_integer(row['prompt'],row['answer'])
        if value is not None:
            real_train.append((canonical_state(arithmetic_state(row['prompt'],row['answer'])),
                               value == row['expected']))
    positives = [item for item in real_train if item[1]]
    negatives = [item for item in real_train if not item[1]]
    cfg = JevConfig(vocab_size=512,d_model=48,n_heads=4,d_ff=96,dropout=0.0,
        n_state_layers=1,n_question_layers=1,n_readout_layers=2,n_slots=2,
        max_state_len=96,max_question_len=24,path_hash_buckets=128,max_sibling_index=8)
    model = Jev(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    question = Choice(QUESTION,options=OPTIONS)
    for step in range(args.steps):
        batch = rng.sample(synthetic, 20)
        if positives and negatives:
            batch += rng.choices(negatives,k=2) + rng.choices(positives,k=2)
        targets = torch.tensor([[.98,.02] if correct else [.02,.98] for _,correct in batch])
        probabilities, confidence = model.logits([s for s,_ in batch],[question])[0]
        agreement = (probabilities.detach()*targets).sum(-1)
        loss = (-(targets*probabilities.log()).sum(-1).mean()
                + .25*torch.nn.functional.mse_loss(confidence,agreement))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%200 == 0:
            print(json.dumps({'step':step+1,'loss':float(loss)}),flush=True)
    model.eval()
    torch.save({'config':asdict(cfg),'model_state_dict':model.state_dict(),'training_steps':args.steps,
                'scope':DOMAIN,'state_encoding':ENCODING},checkpoint)
    selected = [row for row in fresh if row['label'] in (0,1)]
    output = []
    with torch.no_grad():
        for start in range(0,len(selected),32):
            batch = selected[start:start+32]
            states = [canonical_state(arithmetic_state(r['question'],r['answer'])) for r in batch]
            probability,confidence = model.logits(states,[question])[0]
            for row,p,c in zip(batch,probability[:,0].tolist(),confidence.tolist()):
                raw = answer_probability_from_head('correct' if p >= .5 else 'incorrect',c)
                output.append({'group_id':row['group_id'],'split':row['split'],
                               'label':row['label'],'probability':raw})
    cal = [row for row in output if row['split'] == 'calibration']
    test = [row for row in output if row['split'] == 'test']
    try:
        result = evaluate_calibrator(cal,test,predeclared=metadata['predeclared_split'] and independent_test)
    except ValueError as exc:
        result = {'eligible':False, 'issues':[str(exc)], 'slope':None, 'intercept':None}
    bundle = {'format':'hariai_openjev_answer_calibration_v1', 'domain':DOMAIN, 'state_encoding':ENCODING,
        'score_semantics':SCORE_SEMANTICS,
        'source':metadata['source'],'predeclared_split':metadata['predeclared_split'] and independent_test,
        'problem_grouping':metadata['problem_grouping'],
        'model_sha256':metadata['model_sha256'],'system_prompt_sha256':metadata['system_prompt_sha256'],
        'protocol':metadata['protocol'],'verifier_file':checkpoint.name,
        'verifier_sha256':sha256_file(checkpoint),'slope':result['slope'], 'intercept':result['intercept'], 'rows':output,
        'result':result,'synthetic_training_examples':len(synthetic), 'prior_model_training_examples':len(real_train),
        'fresh_generated_examples':len(fresh), 'fresh_scorable_examples':len(output),
        'limitations':'Only complete single-turn integer answers to this fixed task; synthetic training. Not general chat correctness.'}
    (output_dir/'answer_confidence.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    print(json.dumps({'result':result,'fresh_generated':len(fresh),'fresh_scorable':len(output),
                      'prior_training_rows':len(real_train),'artifact':str(output_dir/'answer_confidence.json')}),flush=True)


if __name__ == '__main__':
    main()
