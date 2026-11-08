"""Offline LoRA candidate trained on checked teacher answers and replay.

This local experiment never promotes or modifies an existing checkpoint.
Only independently checked arithmetic records are admitted; private chat is not used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from hari_ai.dream.adapter import inject_lora_adapters, merge_lora_adapters
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.quantization.quantizer import quantize_model, save_quantized_checkpoint
from hari_ai.tokenizer.bpe import ByteFallbackBPETokenizer


def example(a, b, op):
    return f'What is {a} {op} {b}? Answer with only the integer.', str(a + b if op == '+' else a - b)


def checked(record):
    match = re.fullmatch(r'What is (\d+) ([+-]) (\d+)\? Answer with only the integer\.', record.get('prompt', ''))
    if not match or not record.get('verified'):
        return False
    a, op, b = match.groups()
    expected = int(a) + int(b) if op == '+' else int(a) - int(b)
    return record.get('answer') == str(expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('checkpoints/arithmetic_student_long_v1/hari_student.pt'))
    p.add_argument('--teachers', type=Path, nargs='+', default=[Path('data/qwen_teacher_arithmetic_v1.jsonl'), Path('data/smollm_dream_arithmetic_v1.jsonl')])
    p.add_argument('--language-data', type=Path, default=Path('data/fineweb_pilot.jsonl'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/dream_candidate_v1'))
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    rows = [json.loads(line) for path in args.teachers for line in path.read_text(encoding='utf-8').splitlines() if line]
    # Reconstruct the exact held-out split of the existing arithmetic student.
    baseline_rows = [json.loads(line) for line in args.teachers[0].read_text(encoding='utf-8').splitlines() if line]
    teacher_eval = [(r['prompt'], r['answer']) for r in baseline_rows if checked(r) and r.get('split') == 'eval']
    held = {q for q, _ in teacher_eval}
    examples = [example(a, b, op) for a in range(2, 30) for b in range(2, 20) for op in ('+', '-')]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    eval_pairs = [(q, a) for q, a in examples if q not in held][:40] + teacher_eval
    held.update(q for q, _ in eval_pairs)
    replay = [(q, a) for q, a in examples if q not in held]
    admitted = [(r['prompt'], r['answer']) for r in rows if checked(r) and r.get('split') == 'train' and r['prompt'] not in held]
    if not admitted:
        p.error('No independently checked, non-evaluation teacher examples')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    saved = torch.load(args.base, map_location='cpu', weights_only=False)
    model = UAIANanoModel(saved['config'])
    model.load_state_dict(saved['model_state_dict'])
    tokenizer = ByteFallbackBPETokenizer.load(str(args.base.parent / 'tokenizer.json'))
    tokenizer.save(str(args.output / 'tokenizer.json'))
    language = [json.loads(line)['text'] for line in args.language_data.read_text(encoding='utf-8').splitlines()
                if line and json.loads(line).get('research_split') == 'eval'][:16]
    if not language:
        p.error('No held-out language documents')
    language_ids = [torch.tensor([tokenizer.encode(text[:128])[:128]]) for text in language if len(text) >= 128]
    if not language_ids:
        p.error('No sufficiently long held-out language documents')

    @torch.no_grad()
    def evaluate():
        model.eval()
        cases = []
        for prompt, answer in eval_pairs:
            ids = tokenizer.encode('User: ' + prompt + '\nHariAI:', add_bos=True)
            result = model.generate(torch.tensor([ids]), max_new_tokens=6, temperature=0)[0, len(ids):].tolist()
            if tokenizer.eos_id in result:
                result = result[:result.index(tokenizer.eos_id)]
            text = tokenizer.decode(result).strip()
            cases.append({'prompt': prompt, 'expected': answer, 'generated': text, 'correct': text == answer})
        losses = [model(ids, labels=ids)['loss'].item() for ids in language_ids]
        return {'arithmetic_correct': sum(c['correct'] for c in cases), 'arithmetic_total': len(cases),
                'language_loss_nats': sum(losses) / len(losses), 'cases': cases}

    before = evaluate()
    fp32_model = model
    model = quantize_model(copy.deepcopy(model), precision='int4')
    quantized_before = evaluate()
    model = fp32_model
    inject_lora_adapters(model, rank=8, alpha=8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0005)
    def encode(pair):
        prompt, answer = pair
        prefix = tokenizer.encode('User: ' + prompt + '\nHariAI:', add_bos=True)
        suffix = tokenizer.encode(' ' + answer, add_eos=True)
        return prefix + suffix, [-100] * len(prefix) + suffix
    examples_encoded = [encode(x) for x in admitted]
    replay_encoded = [encode(x) for x in replay]
    for step in range(args.steps):
        model.train()
        entries = rng.choices(examples_encoded, k=8) + rng.choices(replay_encoded, k=8)
        width = max(len(x) for x, _ in entries)
        inputs = torch.tensor([x + [tokenizer.pad_id] * (width - len(x)) for x, _ in entries])
        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in entries])
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, labels=labels)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f'{step+1}/{args.steps} loss={loss.item():.4f}', flush=True)
    merge_lora_adapters(model)
    after = evaluate()
    torch.save({'config': model.config, 'model_state_dict': model.state_dict(), 'step': args.steps}, args.output / 'candidate_fp32.pt')
    quantize_model(model, precision='int4')
    quantized = evaluate()
    info = save_quantized_checkpoint(model, str(args.output / 'candidate_int4.pt'), max_size_mb=35_000_000 / (1024 * 1024))
    # Compare deployment candidates at the same precision, with no automatic promotion.
    passes = (quantized['arithmetic_correct'] > quantized_before['arithmetic_correct'] and
              quantized['language_loss_nats'] <= quantized_before['language_loss_nats'] + 0.05 and
              info['bytes'] < 35_000_000)
    report = {'base': str(args.base), 'base_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
        'teacher_files': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.teachers],
        'language_data_sha256': hashlib.sha256(args.language_data.read_bytes()).hexdigest(),
        'teacher_records': len(rows),
        'admitted_teacher_examples': len(admitted), 'replay_examples': len(replay),
        'trainable': 'LoRA q/v projections; base frozen', 'steps': args.steps,
        'before': before, 'quantized_before': quantized_before, 'after': after, 'quantized': quantized, 'export': info,
        'gate': {'passes': passes, 'requirements': 'held-out arithmetic improves, language loss increases <=0.05 nats, weights <35M bytes',
                 'promoted': False, 'note': 'Research gate on narrow data; never automatically promotes.'}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'admitted': len(admitted), 'before': before['arithmetic_correct'],
                      'quantized_before': quantized_before['arithmetic_correct'],
                      'after': after['arithmetic_correct'], 'quantized': quantized['arithmetic_correct'],
                      'language_before': before['language_loss_nats'], 'quantized_language_before': quantized_before['language_loss_nats'],
                      'language_quantized': quantized['language_loss_nats'],
                      'passes': passes, 'promoted': False, 'bytes': info['bytes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
