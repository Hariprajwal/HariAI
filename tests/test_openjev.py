"""Open-JEV adapter loads an explicit trained reference checkpoint."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from hari_ai.runtime.openjev import OpenJevAdapter, OperationDecisionModel, canonical_state


class FakeChoice:
    def __init__(self, question, options):
        self.options = options


class FakeConfig:
    def __init__(self, **kwargs):
        pass


class FakeModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, states, questions):
        class Result:
            choice = questions[0].options[0]
            probabilities = {choice: 1.0}
            confidence = 0.2
        return [[Result()]]


class OpenJevTests(unittest.TestCase):
    def test_json_field_order_is_canonicalized(self):
        left = {'b': {'y': 1, 'a': 2}, 'a': ['z', 'a']}
        right = {'a': ['z', 'a'], 'b': {'a': 2, 'y': 1}}
        self.assertEqual(repr(canonical_state(left)), repr(canonical_state(right)))

    def test_bounded_operation_model_rejects_unsupported_tasks(self):
        model = OperationDecisionModel()
        with self.assertRaises(ValueError):
            model.decide({'question': 'What is the capital of Japan?'})
        with self.assertRaises(ValueError):
            model.decide({'operation': 'launch', 'scope': 'single', 'authorized': 'yes', 'backup': 'yes'})

    def test_requires_trained_checkpoint(self):
        with self.assertRaises(ValueError):
            OpenJevAdapter('')
        with tempfile.TemporaryDirectory() as tmp, patch('hari_ai.runtime.openjev._reference_module',
                return_value=(FakeChoice, FakeModel, FakeConfig)):
            path = Path(tmp) / 'reference.pt'
            torch.save({'config': {}, 'model_state_dict': FakeModel(None).state_dict(), 'training_steps': 0}, path)
            with self.assertRaises(ValueError):
                OpenJevAdapter(str(path))
            torch.save({'config': {}, 'model_state_dict': FakeModel(None).state_dict(), 'training_steps': 1}, path)
            result = OpenJevAdapter(str(path)).decide({'task': 'demo'}, 'Which?', ['a', 'b'])
            self.assertEqual(result['choice'], 'a')
            self.assertFalse(result['calibrated'])
            self.assertGreater(result['checkpoint_bytes'], 0)

    @unittest.skipUnless((Path(__file__).resolve().parent.parent / 'third_party/open-jev/open_jev/main.py').exists(), 'Open-JEV clone not present')
    def test_real_reference_checkpoint_and_typed_choice(self):
        from dataclasses import asdict
        from hari_ai.runtime.openjev import _reference_module
        choice, model_type, config_type = _reference_module()
        cfg = config_type(vocab_size=128, d_model=32, n_heads=4, d_ff=64,
                          n_state_layers=1, n_question_layers=1, n_readout_layers=1,
                          n_slots=2, max_state_len=32, max_question_len=16,
                          path_hash_buckets=32, max_sibling_index=8)
        model = model_type(cfg)
        question = choice('Pick an action', options=['review', 'stop'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        probabilities, _ = model.logits([{'task': 'review'}], [question])[0]
        loss = -probabilities[0, 0].log()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'real_reference.pt'
            torch.save({'config': asdict(cfg), 'model_state_dict': model.state_dict(), 'training_steps': 1}, path)
            result = OpenJevAdapter(str(path)).decide({'task': 'review'}, 'Pick an action', ['review', 'stop'])
            self.assertIn(result['choice'], ('review', 'stop'))
            self.assertFalse(result['calibrated'])
