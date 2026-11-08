"""Collect an independently scored, pre-split model-output calibration set.

All prompts are synthetic; no chat history or user memory is accessed. Raw
outputs stay in the ignored output directory. Run the exact production prompt,
sampling options and token limit. Only completely stated integer answers are
eligible for scoring.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.runtime.answer_confidence import stated_integer, sha256_file
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.local_model import LocalModel, bundled_model_path
from hari_ai.runtime.settings import ChatSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=400)
    parser.add_argument('--model', type=Path, default=bundled_model_path())
    parser.add_argument('--output', type=Path, default=Path('checkpoints/jev_calibration_fresh'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output/'raw.jsonl'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose another output directory.')
    rng = random.Random(247)
    # Equivalent commutative additions must not cross the held-out split.
    population = [(a, op, b) for a in range(0, 30) for b in range(0, 30) for op in ('+', '-')
                  if op != '+' or a <= b]
    if args.count < 1 or args.count > len(population):
        raise ValueError(f'count must be between 1 and {len(population)} unique problem groups')
    questions = rng.sample(population, args.count)
    # Assign problem groups without looking at their answers or labels.
    assignment = list(range(len(questions)))
    rng.shuffle(assignment)
    calibration_indices = set(assignment[:len(assignment)//2])
    model = LocalModel(str(args.model))
    system = ChatEngine(settings=ChatSettings()).system_prompt
    metadata = {'format':'hariai_answer_examples_v1', 'model_sha256':sha256_file(args.model),
        'system_prompt_sha256':hashlib.sha256(system.encode('utf-8')).hexdigest(),
        'protocol':{'backend':'local','max_tokens':192,'temperature':0.2,'top_p':0.9,
                    'repeat_penalty':1.1,'history':'single_turn'},
        'predeclared_split':True,'source':'model_generated_independently_graded',
        'seed':247,'requested':args.count,'problem_grouping':'commutative_addition_v1'}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    counts = {'generated':0,'scorable':0,'correct':0,'incorrect':0}
    with output.open('x',encoding='utf-8') as file:
        for i,(a,op,b) in enumerate(questions):
            question = f'What is {a} {op} {b}? Answer with only the number.'
            answer = ''.join(model.chat_stream([{'role':'system','content':system},
                                               {'role':'user','content':question}]))
            result = stated_integer(question,answer)
            # A response cut at the generation limit is not a complete answer.
            if model.last_finish_reason == 'length':
                result = None
            expected = a+b if op == '+' else a-b
            label = None if result is None else int(result == expected)
            row = {'question':question,'answer':answer,'expected':expected,'label':label,
                   'split':'calibration' if i in calibration_indices else 'test',
                   'group_id':hashlib.sha256(question.encode('utf-8')).hexdigest(),
                   'finish_reason':model.last_finish_reason}
            file.write(json.dumps(row,ensure_ascii=False)+'\n')
            file.flush()
            counts['generated'] += 1
            if label is not None:
                counts['scorable'] += 1
                counts['correct' if label else 'incorrect'] += 1
            if (i+1)%25 == 0:
                print(json.dumps(counts),flush=True)
    model.close()
    (args.output/'summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts),flush=True)


if __name__ == '__main__':
    main()
