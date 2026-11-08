"""Checks that the smoke probe scores actual generated responses."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import evaluate_checkpoint


class CheckpointProbeTests(unittest.TestCase):
    def test_scores_outputs_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model.pt'
            model.write_bytes(b'example model')
            tokenizer = root / 'tokenizer.json'
            tokenizer.write_text('{}', encoding='utf-8')
            dataset = root / 'cases.jsonl'
            dataset.write_text('\n'.join(json.dumps(row) for row in [
                {'domain': 'instruction', 'prompt': 'Return blue', 'answers': ['blue']},
                {'domain': 'arithmetic', 'prompt': 'Return four', 'answers': ['4']},
            ]), encoding='utf-8')
            report_path = root / 'report.json'

            class FakeEngine:
                def __init__(self, *args, **kwargs):
                    pass

                def generate(self, prompt, config):
                    return 'blue' if 'Return blue' in prompt else 'wrong'

            argv = ['probe', str(model), '--dataset', str(dataset), '--output', str(report_path)]
            with patch.object(sys, 'argv', argv), patch.object(evaluate_checkpoint, 'InferenceEngine', FakeEngine), contextlib.redirect_stdout(io.StringIO()):
                evaluate_checkpoint.main()
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual((report['correct'], report['total']), (1, 2))
            self.assertEqual(report['cases'][1]['generated'], 'wrong')
            self.assertEqual(report['checkpoint_bytes'], len(b'example model'))
            self.assertEqual(len(report['dataset_sha256']), 64)
            self.assertEqual(len(report['tokenizer_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
