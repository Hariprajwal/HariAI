"""Check training/export rotation inverses without loading real weights."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('gguf'), 'optional gguf training dependency is absent')
class GgufTrainingTests(unittest.TestCase):
    def test_query_key_rotation_roundtrip(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from gguf_training import permute, unpermute
        for heads, width in [(9, 576), (3, 192)]:
            values = np.arange(width * 32, dtype=np.float32).reshape(width, 32)
            np.testing.assert_array_equal(permute(unpermute(values, heads), heads), values)

