"""Train and evaluate a tiny, from-scratch arithmetic student.

This bounded experiment measures exact-match generalization and cannot establish
general assistant ability. Teacher examples are independently checked before use.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.quantization.quantizer import quantize_model, save_quantized_checkpoint
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer


def example(a, b, op):
    return f'What is {a} {op} {b}? Answer with only the integer.', str(a + b if op == '+' else a - b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='data/qwen_teacher_arithmetic_v1.jsonl')
    parser.add_argument('--output', default='checkpoints/arithmetic_student_v1')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = ByteFallbackBPETokenizer(target_vocab_size=266)
    tokenizer.save(str(output / 'tokenizer.json'))
    rows = [json.loads(line) for line in Path(args.teacher).read_text(encoding='utf-8').splitlines() if line]
    def is_correct(record):
        match = re.fullmatch(r'What is (\d+) ([+-]) (\d+)\? Answer with only the integer\.', record.get('prompt', ''))
        if not match:
            return False
        a, op, b = match.groups()
        answer = int(a) + int(b) if op == '+' else int(a) - int(b)
        return record.get('verified') is True and record.get('answer') == str(answer)

    verified = [r for r in rows if is_correct(r)]
    teacher_eval = [(r['prompt'], r['answer']) for r in verified if r.get('split') == 'eval']
    held_out = {p for p, _ in teacher_eval}
    examples = [example(a, b, op) for a in range(2, 30) for b in range(2, 20) for op in ('+', '-')]
    random.shuffle(examples)
    eval_pairs = [(p, a) for p, a in examples if p not in held_out][:40] + teacher_eval
    held_out.update(p for p, _ in eval_pairs)
    train = [(p, a) for p, a in examples if p not in held_out]
    teacher_train = [(r['prompt'], r['answer']) for r in verified if r.get('split') == 'train' and r['prompt'] not in held_out]
    train.extend(teacher_train * 4)
    config = UAIANanoConfig(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=4,
                            n_heads=4, n_kv_heads=2, d_ff=384, max_seq_len=128, n_actions=8)
    model = UAIANanoModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.01)

    def encode(pair):
        prompt, answer = pair
        prefix = tokenizer.encode('User: ' + prompt + '\nHariAI:', add_bos=True)
        response = tokenizer.encode(' ' + answer, add_eos=True)
        ids = prefix + response
        labels = [-100] * len(prefix) + response
        return ids, labels

    encoded = [encode(pair) for pair in train]

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = 0
        samples = []
        for prompt, answer in eval_pairs:
            ids = tokenizer.encode('User: ' + prompt + '\nHariAI:', add_bos=True)
            output_ids = model.generate(torch.tensor([ids]), max_new_tokens=6, temperature=0)
            generated = output_ids[0, len(ids):].tolist()
            if tokenizer.eos_id in generated:
                generated = generated[:generated.index(tokenizer.eos_id)]
            result = tokenizer.decode(generated).strip()
            correct += int(result == answer)
            if len(samples) < 5:
                samples.append({'prompt': prompt, 'expected': answer, 'generated': result})
        return {'correct': correct, 'total': len(eval_pairs), 'samples': samples}

    initial = evaluate()
    print('Before:', initial, flush=True)
    for step in range(args.steps):
        model.train()
        batch = random.choices(encoded, k=16)
        width = max(len(ids) for ids, _ in batch)
        xs = [ids + [tokenizer.pad_id] * (width - len(ids)) for ids, _ in batch]
        ys = [labels + [-100] * (width - len(labels)) for _, labels in batch]
        loss = model(torch.tensor(xs), labels=torch.tensor(ys))['loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f'{step+1}/{args.steps} loss={loss.item():.4f}', flush=True)
    final = evaluate()
    print('After:', final, flush=True)
    torch.save({'config': config, 'model_state_dict': model.state_dict(), 'step': args.steps}, output / 'hari_student.pt')
    quantize_model(model, precision='int4')
    quantized_result = evaluate()
    print('After quantization:', quantized_result, flush=True)
    export = save_quantized_checkpoint(model, str(output / 'hari_student_int4.pt'), max_size_mb=35_000_000 / (1024 * 1024))
    (output / 'report.json').write_text(json.dumps({'seed': args.seed, 'steps': args.steps,
        'teacher_source': args.teacher, 'teacher_verified_train': len(teacher_train),
        'teacher_verified_eval': len(teacher_eval), 'oracle_train_examples': len(train)-len(teacher_train)*4,
        'baseline': initial, 'final': final, 'quantized': quantized_result, 'export': export}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
