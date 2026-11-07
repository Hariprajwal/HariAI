"""Optional adapter to the independently installed Open-JEV reference model.

Open-JEV produces typed decisions, not generated chat or general-answer
confidence. Its upstream repository currently ships random weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch


def _reference_module():
    root = Path(__file__).resolve().parents[2] / 'third_party' / 'open-jev'
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from open_jev.main import Choice, Jev, JevConfig
    except ImportError as exc:
        raise RuntimeError('Open-JEV reference is not installed; clone it into third_party/open-jev.') from exc
    return Choice, Jev, JevConfig


class OpenJevAdapter:
    """Load a user-supplied trained reference checkpoint for typed choices."""

    def __init__(self, checkpoint: str):
        if not checkpoint:
            raise ValueError('A trained Open-JEV checkpoint is required; upstream ships random weights.')
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Open-JEV checkpoint not found: {path}')
        Choice, Jev, JevConfig = _reference_module()
        data = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(data, dict) or not isinstance(data.get('config'), dict) or 'model_state_dict' not in data:
            raise ValueError('Expected an Open-JEV checkpoint with config and model_state_dict.')
        if not isinstance(data.get('training_steps'), int) or data['training_steps'] <= 0:
            raise ValueError('Open-JEV checkpoint must record a positive training_steps value.')
        self.choice_type = Choice
        self.model = Jev(JevConfig(**data['config']))
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()
        self.checkpoint_bytes = path.stat().st_size

    def decide(self, state: Any, question: str, options: Sequence[str]) -> Dict[str, Any]:
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError('Provide at least two distinct choices.')
        with torch.no_grad():
            answer = self.model([state], [self.choice_type(question, options=list(options))])[0][0]
        return {'choice': answer.choice, 'probabilities': answer.probabilities,
                'confidence_raw': answer.confidence, 'calibrated': False,
                'checkpoint_bytes': self.checkpoint_bytes}
