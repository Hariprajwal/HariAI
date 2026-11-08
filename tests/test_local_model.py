"""Context retention and missing-model failures, independent of downloaded weights."""
import tempfile
import unittest
from pathlib import Path
from hari_ai.runtime.local_model import LocalModel


class TokenCounter:
    def tokenize(self, text, **kwargs):
        return text.split()


class LocalModelTests(unittest.TestCase):
    def test_drops_oldest_complete_turn(self):
        model = LocalModel(context_size=45)
        model._model = TokenCounter()
        messages = [{'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'old ' * 20}, {'role': 'assistant', 'content': 'old reply'},
            {'role': 'user', 'content': 'newest'}]
        fitted = model._fit_messages(messages, max_tokens=15)
        self.assertEqual(fitted, [messages[0], messages[-1]])
        self.assertEqual(len(messages), 4)

    def test_oversized_latest_message_is_not_silently_truncated(self):
        model = LocalModel(context_size=30)
        model._model = TokenCounter()
        with self.assertRaisesRegex(ValueError, 'too long'):
            model._fit_messages([{'role': 'system', 'content': 'system'},
                {'role': 'user', 'content': 'word ' * 100}], 10)

    def test_missing_file_never_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            model = LocalModel(str(Path(folder) / 'absent.gguf'))
            with self.assertRaises(FileNotFoundError):
                list(model.chat_stream([{'role': 'user', 'content': 'Hello'}]))
