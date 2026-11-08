"""Train an experimental Open-JEV arithmetic answer verifier and test it.

It is never installed into chat. Synthetic operands exclude all observed
evaluation operands; real model answers are split before training. Accuracy,
class recall and calibration are compared with an all-incorrect baseline.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from hari_ai.runtime.openjev import canonical_state
from hari_ai.vendor.open_jev.main import Choice, Jev, JevConfig
from analyze_confidence_results import extract_result

QUESTION = 'Is the stated numerical answer correct?'
OPTIONS = ['correct', 'incorrect']


def state(question, answer):
    return canonical_state({'question': question, 'answer': answer})


def stats(rows, probs):
    correct = [int(row[1]) for row in rows]
    values = probs.tolist()
    n = len(rows)
    hits = sum((p >= .5) == bool(y) for p, y in zip(values, correct))
    positives = sum(correct)
    tp = sum(p >= .5 and y for p, y in zip(values, correct))
    bins = []
    ece = 0.0
    for bucket in range(10):
        pairs = [(p, y) for p, y in zip(values, correct) if min(9, int(p*10)) == bucket]
        if pairs:
            mean_p = sum(p for p, y in pairs) / len(pairs)
            mean_y = sum(y for p, y in pairs) / len(pairs)
            ece += len(pairs)/n * abs(mean_p-mean_y)
            bins.append({'bucket':bucket,'n':len(pairs),'predicted':mean_p,'observed':mean_y})
    return {'count':n, 'correct_labels':positives, 'accuracy':hits/n,
        'correct_answer_recall':tp/positives if positives else None,
        'brier':sum((p-y)**2 for p,y in zip(values,correct))/n, 'ece_10':ece, 'bins':bins}


def main():
    random.seed(91)
    torch.manual_seed(91)
    torch.set_num_threads(2)
    source = json.loads(Path('checkpoints/confidence_evaluation/report.json').read_text(encoding='utf-8'))
    all_rows = source['rows']
    used = set()
    observed = []
    for row in all_rows:
        match = re.search(r'What is (\d+) ([+-]) (\d+)', row['prompt'])
        if match:
            used.add((int(match[1]),int(match[3])))
        value = extract_result(row['answer'])
        if value is not None:
            observed.append((state(row['prompt'], row['answer']), value == row['expected'], row['split']))
    train_real = [(s,y) for s,y,split in observed if split == 'calibration']
    heldout = [(s,y) for s,y,split in observed if split == 'test']
    pairs = [(a,b) for a in range(2,41) for b in range(1,21) if (a,b) not in used]
    synthetic = []
    for _ in range(1600):
        a,b = random.choice(pairs)
        op = random.choice(['+', '-'])
        result = a+b if op == '+' else a-b
        is_correct = random.choice([True,False])
        output = result if is_correct else result + random.choice([-9,-5,-3,-1,1,3,5,9])
        synthetic.append((state(f'What is {a} {op} {b}? Answer with only the number.',
                                f'{a} {op} {b} = {output}'), is_correct))
    cfg = JevConfig(vocab_size=512,d_model=48,n_heads=4,d_ff=96,dropout=0.0,
        n_state_layers=1,n_question_layers=1,n_readout_layers=2,n_slots=2,
        max_state_len=96,max_question_len=24,path_hash_buckets=128,max_sibling_index=8)
    model = Jev(cfg)
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    question = Choice(QUESTION,options=OPTIONS)
    positive = [x for x in train_real if x[1]]
    negative = [x for x in train_real if not x[1]]
    for step in range(400):
        rows = random.sample(synthetic, 16) + random.choices(negative,k=6) + random.choices(positive,k=2)
        targets = torch.tensor([[.98,.02] if yes else [.02,.98] for _,yes in rows])
        probs,confidence = model.logits([s for s,_ in rows],[question])[0]
        agreement = (probs.detach()*targets).sum(-1)
        loss = (-(targets*probs.log()).sum(-1).mean()
                + .25 * torch.nn.functional.mse_loss(confidence, agreement))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if (step+1)%100 == 0:
            print(json.dumps({'step':step+1,'loss':float(loss)}),flush=True)
    model.eval()
    with torch.no_grad():
        probability,confidence = model.logits([s for s,_ in heldout],[question])[0]
    probability = probability[:,0]
    score = stats(heldout,probability)
    classification_agreement = [(s, bool((float(p)>=.5) == bool(y)))
                                for (s,y),p in zip(heldout,probability)]
    head_score = stats(classification_agreement,confidence)
    reference_rate = sum(y for _,y in train_real)/len(train_real)
    base = stats(heldout,torch.full_like(probability,reference_rate))
    report = {'scope':'arithmetic explicit-integer answer classification, research only',
        'source_model_sha256':source['model_sha256'], 'synthetic_train_rows':len(synthetic),
        'real_train_counts':dict(Counter(int(y) for _,y in train_real)),
        'heldout_counts':dict(Counter(int(y) for _,y in heldout)),
        'training_steps':400,'verifier':score,'constant_prior_baseline':base,
        'confidence_head_trained':True, 'confidence_head_for_verifier_choice':head_score,
        'predictions':[{'state':s,'answer_correct':bool(y),'answer_correct_probability_raw':float(p),
                        'verifier_choice_confidence_raw':float(c)}
                       for (s,y),p,c in zip(heldout,probability,confidence)],
        'limitations':'Small synthetic task; only answers with one unambiguous integer are scored. The trained head targets confidence in the verifier choice, not the probability that arbitrary chat answers are correct. Neither output is calibrated. Too few independent positive labels for generalization.'}
    folder=Path('checkpoints/confidence_evaluation')
    (folder/'jev_verifier_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save({'config':cfg.__dict__, 'model_state_dict':model.state_dict(),
                'training_steps':400,'scope':'research_arithmetic_verifier'},folder/'answer_jev_research.pt')
    print(json.dumps({k:v for k,v in report.items() if k != 'predictions'}),flush=True)


if __name__ == '__main__':
    main()
