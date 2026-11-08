"""Train, merge and evaluate an independent HariAI GGUF candidate offline.

This is a small identity adaptation, NOT a claim of better general intelligence.
The original quantized base is never overwritten. The report decides whether
the candidate passes the narrow release gates; promotion is a separate step.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import gguf
import torch
from torch import nn
from llama_cpp import Llama
from gguf_training import load_trainable, export_merged, permute
from adaptation_data import IDENTITY, REPLAY, VALIDATION, IDENTITY_PROBE, GENERAL_PROBE

SYSTEM = 'You are a helpful assistant.'


class LoRA(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged(self):
        return (self.base.weight + self.b @ self.a * self.scale).detach().cpu().numpy()


def prompt_text(question):
    return f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'


def encode(llm, pairs):
    out = []
    for question, answer in pairs:
        prompt = llm.tokenize(prompt_text(question).encode(), add_bos=False, special=True)
        whole = llm.tokenize((prompt_text(question) + answer + '<|im_end|>\n').encode(), add_bos=False, special=True)
        out.append((whole, [-100] * len(prompt) + whole[len(prompt):]))
    return out


def batch(items):
    size = max(len(x[0]) for x in items)
    ids = torch.zeros((len(items), size), dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (x, y) in enumerate(items):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return dict(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)


@torch.no_grad()
def loss_on(model, data):
    model.eval()
    losses = [float(model(**batch(data[i:i + 4])).loss) for i in range(0, len(data), 4)]
    return sum(losses) / len(losses)


def probe(path):
    llm = Llama(model_path=str(path), n_ctx=2048, n_threads=4, verbose=False, chat_format='chatml', seed=42)
    results = []
    identity_targets = [('hariai',), ('k r hari prajwal',), ('k r hari prajwal',), ('hariai',)]
    for question, wanted in list(zip(IDENTITY_PROBE, identity_targets)) + GENERAL_PROBE:
        output = llm.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': question}], max_tokens=120, temperature=0)['choices'][0]['message']['content']
        results.append({'question': question, 'answer': output,
                        'pass': all(w in output.lower() for w in wanted)})
    llm.close()
    return {'identity_pass': sum(r['pass'] for r in results[:4]),
            'general_pass': sum(r['pass'] for r in results[4:]), 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/hariai_adaptation'))
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--changed-precision', choices=['original', 'q8_0', 'f16'], default='q8_0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    start = time.time()
    print('Loading quantized pretrained weights for local adaptation...', flush=True)
    model, reader = load_trainable(args.base)
    tokenizer = Llama(model_path=str(args.base), n_ctx=2048, n_threads=4, verbose=False)
    train = encode(tokenizer, IDENTITY * 2 + REPLAY)
    validation = encode(tokenizer, VALIDATION)
    tokenizer.close()
    baseline_loss = loss_on(model, validation)
    baseline_probe = probe(args.base)
    print(json.dumps({'baseline_loss': baseline_loss, 'baseline_probe': baseline_probe}), flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = {}
    for index in range(model.config.num_hidden_layers - args.layers, model.config.num_hidden_layers):
        attention = model.model.layers[index].self_attn
        for projection, gguf_part in [('q_proj', 'attn_q'), ('v_proj', 'attn_v')]:
            adapter = LoRA(getattr(attention, projection), rank=args.rank)
            setattr(attention, projection, adapter)
            adapters[f'blk.{index}.{gguf_part}.weight'] = adapter
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(random.sample(train, 4))).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        if (step + 1) % 20 == 0:
            print(json.dumps({'step': step + 1, 'loss': float(loss), 'seconds': round(time.time()-start)}), flush=True)
    candidate_loss = loss_on(model, validation)
    torch.save({name: {'a': adapter.a.detach().cpu(), 'b': adapter.b.detach().cpu(), 'scale': adapter.scale}
                for name, adapter in adapters.items()}, args.output/'adapter.pt')
    weights = {}
    for name, adapter in adapters.items():
        weights[name] = adapter.merged()
        if '.attn_q.' in name:
            weights[name] = permute(weights[name], model.config.num_attention_heads)
    candidate = args.output / 'hariai-chat.gguf'
    # Quantizing small merged updates again can erase learned behavior. Keep
    # only adapted tensors at the requested precision; all others are untouched.
    precision = {'original': None, 'q8_0': gguf.GGMLQuantizationType.Q8_0,
                 'f16': gguf.GGMLQuantizationType.F16}[args.changed_precision]
    export_merged(reader, candidate, weights, changed_precision=precision)
    candidate_probe = probe(candidate)
    passed = (candidate_probe['identity_pass'] >= 3
              and candidate_probe['general_pass'] >= baseline_probe['general_pass']
              and candidate_loss <= baseline_loss + 0.15)
    report = {'creator': 'K R Hari Prajwal', 'base': 'HuggingFaceTB/SmolLM-135M-Instruct',
        'base_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'candidate_bytes': candidate.stat().st_size, 'optimizer_steps': args.steps,
        'layers_adapted': args.layers, 'rank': args.rank, 'learning_rate': args.learning_rate,
        'changed_tensor_precision': args.changed_precision,
        'trainable_parameters': sum(p.numel() for p in parameters),
        'training_examples_unique': len(IDENTITY) + len(REPLAY),
        'validation_examples': len(VALIDATION), 'baseline_validation_loss': baseline_loss,
        'candidate_validation_loss': candidate_loss, 'baseline_probe': baseline_probe,
        'candidate_probe': candidate_probe, 'narrow_gates_passed': passed,
        'limitations': 'Identity adaptation only; 8 general substring probes and 8 validation examples are not broad quality evidence.',
        'elapsed_seconds': round(time.time()-start)}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
