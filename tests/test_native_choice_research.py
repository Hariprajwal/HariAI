"""Validate the experimental decision-score transformations."""
import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if importlib.util.find_spec('pyarrow') is not None:
    from research_native_choice import choice_weights, restore_order


@unittest.skipUnless(importlib.util.find_spec('pyarrow') is not None,
                     'pyarrow is an optional research dependency')
class NativeChoiceResearchTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self):
        values = choice_weights([10000.0, 9999.0, 9998.0, 9997.0])
        self.assertAlmostEqual(float(values.sum()), 1.0)
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[0], values[1])

    def test_rotation_returns_scores_to_original_option(self):
        self.assertEqual(restore_order([.1, .2, .3, .4], 1).tolist(), [.4, .1, .2, .3])
        self.assertEqual(restore_order([.1, .2, .3, .4], 3).tolist(), [.2, .3, .4, .1])

    def test_invalid_scores_cannot_reach_calibration(self):
        with self.assertRaises(ValueError):
            choice_weights([1, 2, float('nan'), 4])
        with self.assertRaises(ValueError):
            restore_order([.5, .5], 0)
        with self.assertRaises(ValueError):
            restore_order([.1, .2, .3, .4], 4)


if __name__ == '__main__':
    unittest.main()
