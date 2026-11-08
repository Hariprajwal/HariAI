"""Research split integrity: candidate pairs must never leak across groups."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_truthfulqa_jev import balanced_rows, validate_feature_rows
from research_truthfulqa_generated import grade


class ResearchSplitTests(unittest.TestCase):
    def test_balanced_subset_is_one_candidate_per_question(self):
        rows = [dict(group_id=str(i), split='test', label=label)
                for i in range(6) for label in (1, 0)]
        validate_feature_rows(rows, list(range(12)))
        selected = balanced_rows(rows, [0.2] * 12, 'test')
        self.assertEqual(len(selected), 6)
        self.assertEqual([row['label'] for row in selected], [1, 0, 1, 0, 1, 0])
        self.assertEqual(len({row['group_id'] for row in selected}), 6)

    def test_question_cannot_occur_in_multiple_splits(self):
        rows = [dict(group_id='same', split=split, label=label)
                for split in ('train', 'test') for label in (1, 0)]
        with self.assertRaises(ValueError):
            validate_feature_rows(rows, [0] * 4)

    def test_question_pair_cannot_be_split(self):
        rows = [dict(group_id='a', split='train', label=1),
                dict(group_id='a', split='test', label=0)]
        with self.assertRaises(ValueError):
            validate_feature_rows(rows, [0] * 2)

    def test_generation_grading_abstains_on_unmatched_or_truncated_answer(self):
        references = {'Correct Answers':'Blue; The sky is blue.',
                      'Incorrect Answers':'Red; The sky is red.',
                      'Best Answer':'Blue', 'Best Incorrect Answer':'Red'}
        self.assertEqual(grade('  BLUE! ', references, 'stop'), 1)
        self.assertEqual(grade('Red.', references, 'stop'), 0)
        self.assertIsNone(grade('Blue, because it is red.', references, 'stop'))
        self.assertIsNone(grade('Blue', references, 'length'))


if __name__ == '__main__':
    unittest.main()
