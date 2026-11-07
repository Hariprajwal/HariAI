"""Integration checks for the from-scratch language pilot."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LanguagePilotTests(unittest.TestCase):
    def test_small_run_writes_evaluation_and_budgeted_checkpoint(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / 'data.jsonl'
            records = []
            for split, text in [('train', 'A small training document. ' * 12),
                                ('eval', 'A separate validation document. ' * 12)]:
                records.append({'text': text, 'research_split': split,
                                'sha256': hashlib.sha256(text.encode()).hexdigest()})
            data.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
            output = folder / 'result'
            result = subprocess.run([sys.executable, str(root / 'scripts/train_language_pilot.py'),
                '--data', str(data), '--output', str(output), '--steps', '1', '--batch-size', '2',
                '--seq-len', '32'], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['train_documents'], 1)
            self.assertEqual(report['eval_documents'], 1)
            self.assertEqual(report['training_tokens_processed'], 64)
            self.assertLess((output / 'hari_language_int4.pt').stat().st_size, 35_000_000)
            self.assertGreater(report['quantized_eval_loss_nats'], 0)


if __name__ == '__main__':
    unittest.main()
