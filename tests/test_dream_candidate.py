"""Verify teacher examples cannot self-certify their arithmetic answers."""
import unittest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'dream_arithmetic_candidate', Path(__file__).resolve().parent.parent / 'scripts' / 'dream_arithmetic_candidate.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checked = module.checked


class DreamCandidateTests(unittest.TestCase):
    def test_independent_arithmetic_check(self):
        prompt = 'What is 4 - 9? Answer with only the integer.'
        self.assertTrue(checked({'prompt': prompt, 'answer': '-5', 'verified': True}))
        self.assertFalse(checked({'prompt': prompt, 'answer': '5', 'expected': 5, 'verified': True}))
        self.assertFalse(checked({'prompt': prompt, 'answer': '-5', 'verified': False}))
        self.assertFalse(checked({'prompt': 'arbitrary code', 'answer': '-5', 'verified': True}))


if __name__ == '__main__':
    unittest.main()
