"""Normal local chat should not initialize the training framework at startup."""
import subprocess
import sys
import unittest


class LazyImportTests(unittest.TestCase):
    def test_chat_import_does_not_load_torch(self):
        script = ("import sys; from hari_ai import ChatEngine; "
                  "print(ChatEngine.__name__, 'torch' in sys.modules)")
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ChatEngine False')

    def test_public_research_exports_remain_available(self):
        from hari_ai import HariNanoConfig, HariNanoModel
        from hari_ai.dream import DreamEngineController, LoRALinear
        from hari_ai.runtime import ChatEngine, InferenceEngine
        self.assertTrue(all((HariNanoConfig, HariNanoModel, DreamEngineController,
                             LoRALinear, ChatEngine, InferenceEngine)))
