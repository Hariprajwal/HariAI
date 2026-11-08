"""Local SmolLM/Llama GGUF import and merged export for HariAI research.

Uses the installed gguf library; does not download or execute remote code.
Only the Llama tensor layout is supported. Quantized weights are dequantized
for training; untouched tensors remain byte-identical during export.
"""
from pathlib import Path
import re
import gguf
import numpy as np
import torch


class Reader(gguf.GGUFReader):
    # Compatibility with gguf 0.9 on NumPy 2, without changing site-packages.
    def _get(self, offset, dtype, count=1, override_order=None):
        dtype = np.dtype(dtype).newbyteorder(override_order or self.byte_order)
        return self.data[offset:offset + dtype.itemsize * int(count)].view(dtype)[:int(count)]


def field_value(field):
    if field.types[-1] == gguf.GGUFValueType.STRING:
        values = [bytes(field.parts[i]).decode('utf-8') for i in field.data]
    else:
        values = [v.item() for i in field.data for v in field.parts[i]]
    return values if field.types[0] == gguf.GGUFValueType.ARRAY else values[0]


def hf_name(name):
    names = {'token_embd.weight': 'model.embed_tokens.weight', 'output_norm.weight': 'model.norm.weight',
             'output.weight': 'lm_head.weight'}
    if name in names:
        return names[name]
    match = re.fullmatch(r'blk\.(\d+)\.(.+)\.weight', name)
    if not match:
        raise ValueError(f'Unsupported tensor: {name}')
    parts = {'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm',
             'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
             'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj',
             'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj'}
    return f'model.layers.{match[1]}.{parts[match[2]]}.weight'


def unpermute(weight, heads):
    return weight.reshape(heads, weight.shape[0] // heads // 2, 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def permute(weight, heads):
    return weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def load_trainable(path):
    # This text-only trainer does not require the host's optional audio stack.
    from unittest.mock import patch
    with patch('transformers.utils.is_librosa_available', return_value=False):
        from transformers import LlamaConfig, LlamaForCausalLM
    reader = Reader(str(path))
    meta = {key: field_value(value) for key, value in reader.fields.items()}
    if meta['general.architecture'] != 'llama':
        raise ValueError('Only Llama/SmolLM GGUF files are supported.')
    config = LlamaConfig(vocab_size=meta['llama.vocab_size'], hidden_size=meta['llama.embedding_length'],
        intermediate_size=meta['llama.feed_forward_length'], num_hidden_layers=meta['llama.block_count'],
        num_attention_heads=meta['llama.attention.head_count'], num_key_value_heads=meta['llama.attention.head_count_kv'],
        max_position_embeddings=meta['llama.context_length'], rms_norm_eps=meta['llama.attention.layer_norm_rms_epsilon'],
        rope_theta=meta['llama.rope.freq_base'], attention_bias=False, mlp_bias=False,
        tie_word_embeddings=not any(t.name == 'output.weight' for t in reader.tensors))
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config)
    state = {}
    for tensor in reader.tensors:
        values = gguf.dequantize(tensor.data, tensor.tensor_type)
        if '.attn_q.' in tensor.name:
            values = unpermute(values, config.num_attention_heads)
        elif '.attn_k.' in tensor.name:
            values = unpermute(values, config.num_key_value_heads)
        state[hf_name(tensor.name)] = torch.from_numpy(np.array(values, copy=True))
    if config.tie_word_embeddings:
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, strict=True)
    return model, reader


def export_merged(reader, destination, changed_weights, name='HariAI-Smollm-135M', changed_precision=None):
    """Requantize changed tensors; preserve tokenizer, architecture and upstream credit."""
    writer = gguf.GGUFWriter(str(destination), 'llama')
    for key, field in reader.fields.items():
        if key.startswith('GGUF.') or key == 'general.architecture':
            continue
        value = field_value(field)
        if key == 'general.name':
            value = name
        elif key == 'general.license':
            value = 'LicenseRef-HariAI-Noncommercial AND Apache-2.0'
        writer.add_key_value(key, value, field.types[0])
    writer.add_string('general.finetuned_by', 'K R Hari Prajwal')
    writer.add_string('general.description', 'HariAI identity adaptation of SmolLM-135M-Instruct; not trained from scratch.')
    for tensor in reader.tensors:
        data = tensor.data
        dtype = tensor.tensor_type
        if tensor.name in changed_weights:
            dtype = changed_precision or dtype
            data = gguf.quantize(changed_weights[tensor.name].astype(np.float32), dtype)
        writer.add_tensor(tensor.name, data, raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
