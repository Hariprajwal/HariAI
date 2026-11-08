"""Backend configuration defaults and persistence."""
import tempfile
import json
import unittest
from pathlib import Path
from hari_ai.runtime.settings import ChatSettings, load_settings, save_settings


class SettingsTests(unittest.TestCase):
    def test_legacy_research_default_migrates_but_custom_choices_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            for original, expected in [({'backend': 'nano'}, 'local'),
                    ({'backend': 'nano', 'nano_checkpoint': 'custom.pt'}, 'nano'),
                    ({'backend': 'ollama'}, 'ollama')]:
                path.write_text(json.dumps(original), encoding='utf-8')
                self.assertEqual(load_settings(path).backend, expected)
            save_settings(ChatSettings(backend='nano'), path)
            self.assertEqual(load_settings(path).backend, 'nano')
    def test_default_is_local_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            self.assertEqual(load_settings(path).backend, 'local')
            save_settings(ChatSettings(backend='ollama', ollama_model='smollm:135m'), path)
            settings = load_settings(path)
            self.assertEqual((settings.backend, settings.ollama_model), ('ollama', 'smollm:135m'))

    def test_invalid_backend_is_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            with self.assertRaises(ValueError):
                save_settings(ChatSettings(backend='other'), path)
            self.assertFalse(path.exists())
