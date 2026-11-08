"""Report actual file size and stored tensor bytes for a trusted checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Only load trusted checkpoints: PyTorch deserialization can execute code.')
    parser.add_argument('--budget-bytes', type=int, default=35_000_000)
    args = parser.parse_args()
    # These project checkpoints store config objects and require weights_only=False.
    data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = data.get('state_dict', data.get('model_state_dict'))
    if not isinstance(state, dict):
        parser.error('No recognized model state dictionary in checkpoint.')
    categories = {'embeddings': 0, 'packed_linear_weights': 0, 'scales': 0, 'other_tensors': 0}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        size = value.numel() * value.element_size()
        if name.startswith('token_embeddings.'):
            category = 'embeddings'
        elif name.endswith('.packed_weight'):
            category = 'packed_linear_weights'
        elif name.endswith('.scale'):
            category = 'scales'
        else:
            category = 'other_tensors'
        categories[category] += size
    size = args.checkpoint.stat().st_size
    print(json.dumps({'checkpoint': str(args.checkpoint),
                      'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      'file_bytes': size, 'budget_bytes': args.budget_bytes,
                      'under_budget': size < args.budget_bytes,
                      'logical_tensor_bytes': categories,
                      'note': 'Tensor byte totals are logical sizes; serialization overhead and shared storage can change file size.'}, indent=2))


if __name__ == '__main__':
    main()
