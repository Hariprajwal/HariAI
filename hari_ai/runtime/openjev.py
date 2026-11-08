"""Adapters for the attributed, bundled Open-JEV reference architecture.

Open-JEV produces typed decisions, not generated chat or general-answer
confidence. Its upstream repository currently ships random weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch


def canonical_state(value):
    """Keep equivalent JSON objects invariant to insertion order at the boundary."""
    if isinstance(value, dict):
        return {key: canonical_state(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_state(item) for item in value]
    return value


def _reference_module():
    from hari_ai.vendor.open_jev.main import Choice, Jev, JevConfig
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
        self.checkpoint_scope = data.get('scope')
        self.state_encoding = data.get('state_encoding')
        self.model = Jev(JevConfig(**data['config']))
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()
        self.checkpoint_bytes = path.stat().st_size

    def decide(self, state: Any, question: str, options: Sequence[str]) -> Dict[str, Any]:
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError('Provide at least two distinct choices.')
        with torch.no_grad():
            answer = self.model([canonical_state(state)], [self.choice_type(question, options=list(options))])[0][0]
        return {'choice': answer.choice, 'probabilities': answer.probabilities,
                'confidence_raw': answer.confidence, 'calibrated': False,
                'checkpoint_bytes': self.checkpoint_bytes}


class OperationDecisionModel:
    """Bounded synthetic policy classifier. Advisory only; never executes an action."""
    OPTIONS = ['proceed', 'review', 'stop']
    QUESTION = 'Classify the operation under the training policy.'
    OPERATIONS = {'read', 'list', 'create', 'update', 'delete', 'format', 'overwrite', 'export'}

    def __init__(self, checkpoint: str = ''):
        self.path = Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[1] / 'assets' / 'operation_jev.pt'
        self._adapter = None

    def decide(self, state: Dict[str, str]):
        if not isinstance(state, dict) or set(state) != {'operation', 'scope', 'authorized', 'backup'}:
            raise ValueError('Expected exactly operation, scope, authorized, backup.')
        if state['operation'] not in self.OPERATIONS or state['scope'] not in ('single', 'batch', 'all'):
            raise ValueError('Operation or scope is outside the trained schema.')
        if state['authorized'] not in ('yes', 'no') or state['backup'] not in ('yes', 'no'):
            raise ValueError('authorized and backup must be yes or no.')
        if self._adapter is None:
            metadata = torch.load(self.path, map_location='cpu', weights_only=True)
            if metadata.get('scope') != 'operation_policy_v1' or metadata.get('passed') is not True:
                raise ValueError('Checkpoint did not pass the operation-policy development gates.')
            self._adapter = OpenJevAdapter(str(self.path))
        result = self._adapter.decide(state, self.QUESTION, self.OPTIONS)
        result.pop('confidence_raw')
        return {**result, 'scope': 'operation_policy_v1', 'advisory_only': True}
