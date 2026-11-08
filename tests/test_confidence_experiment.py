"""Tests for the arithmetic evaluation's conservative extraction rules."""
import unittest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('analyze_confidence_results',
    Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_confidence_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analyze, extract_result = module.analyze, module.extract_result


class ConfidenceExperimentTests(unittest.TestCase):
    def test_extracts_a_unique_integer_but_rejects_ambiguous_or_fractional_values(self):
        self.assertEqual(extract_result('7 + 5 = 12. The answer is 12.'), 12)
        self.assertEqual(extract_result('-3'), -3)
        self.assertIsNone(extract_result('The answer is 2.5'))
        self.assertIsNone(extract_result('The answer is 33 + 14.'))
        self.assertIsNone(extract_result('The answer is 1,200.'))
        self.assertIsNone(extract_result('2 + 3 = 5, but the answer is 8.'))
        self.assertIsNone(extract_result('I do not know.'))

    def test_unscorable_answers_are_not_counted_as_incorrect(self):
        rows = [
            {'answer':'2 + 3 = 5', 'split':'test', 'expected':5, 'estimate':0.6},
            {'answer':'unclear', 'split':'test', 'expected':5, 'estimate':0.5},
        ]
        result = analyze(rows)
        self.assertEqual(result['correct_among_extracted'], 1)
        self.assertEqual(result['not_extracted'], 1)
        self.assertFalse(result['enough_labels_to_fit_minimum_5_per_class'])
