"""Context retention and missing-model failures, independent of downloaded weights."""
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from hari_ai.runtime.local_model import LocalModel


class TokenCounter:
    def tokenize(self, text, **kwargs):
        return text.split()


class LocalModelTests(unittest.TestCase):
    def test_terminal_load_preserves_console_handles_and_quiets_generation(self):
        seen = []
        class StubLlama:
            def __init__(self, **kwargs):
                seen.append(kwargs['verbose'])
                self.verbose = kwargs['verbose']
            def close(self):
                pass
        fake_module = types.SimpleNamespace(Llama=StubLlama, __version__='0.3.8')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            path.write_bytes(b'placeholder')
            with patch.dict(sys.modules, {'llama_cpp': fake_module}):
                for terminal in (False, True):
                    model = LocalModel(str(path))
                    model.preserve_terminal = terminal
                    self.assertFalse(model._load().verbose)
                    model.close()
        # False uses upstream descriptor suppression, True bypasses it so a
        # running Windows terminal retains valid input/output handles.
        self.assertEqual(seen, [False, True])

    def test_overlapping_streams_are_rejected_and_close_releases_lock(self):
        class StreamModel(TokenCounter):
            def create_chat_completion(self, **kwargs):
                yield {'choices': [{'delta': {'content': 'first'}, 'finish_reason': None}]}
                yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        model = LocalModel(context_size=100)
        model._model = StreamModel()
        messages = [{'role':'user', 'content':'hi'}]
        first = model.chat_stream(messages, max_tokens=10)
        self.assertEqual(next(first), 'first')
        with self.assertRaisesRegex(RuntimeError, 'already generating'):
            next(model.chat_stream(messages, max_tokens=10))
        first.close()
        self.assertEqual(''.join(model.chat_stream(messages, max_tokens=10)), 'first')

    def test_length_finish_reason_is_recorded_for_the_current_turn(self):
        class StreamModel(TokenCounter):
            def create_chat_completion(self, **kwargs):
                yield {'choices': [{'delta': {'content': 'partial'}, 'finish_reason': None}]}
                yield {'choices': [{'delta': {}, 'finish_reason': 'length'}]}
        model = LocalModel(context_size=100)
        model._model = StreamModel()
        result = ''.join(model.chat_stream([{'role': 'user', 'content': 'write code'}], max_tokens=10))
        self.assertEqual(result, 'partial')
        self.assertEqual(model.last_finish_reason, 'length')

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

    def test_unverified_llama_version_keeps_chat_but_disables_token_score(self):
        class StubLlama(TokenCounter):
            def __init__(self, **kwargs):
                pass
            def create_chat_completion(self, **kwargs):
                yield {'choices':[{'delta':{'content':'answer'}, 'finish_reason':None}]}
                yield {'choices':[{'delta':{}, 'finish_reason':'stop'}]}
            def close(self):
                pass
        fake_module = types.SimpleNamespace(Llama=StubLlama, __version__='99.0')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'
            path.write_bytes(b'placeholder')
            model = LocalModel(str(path), context_size=100)
            with patch.dict(sys.modules, {'llama_cpp':fake_module}):
                self.assertEqual(''.join(model.chat_stream([{'role':'user','content':'hello'}], 10)), 'answer')
            self.assertFalse(model._token_likelihood_compatible)
            self.assertIsNone(model.last_token_likelihood)
            model.close()

    def test_anti_repetition_sampling_defaults_and_overrides(self):
        seen_kwargs = {}
        class MockSamplingLlama(TokenCounter):
            def __init__(self, **kwargs):
                pass
            def create_chat_completion(self, **kwargs):
                seen_kwargs.update(kwargs)
                yield {'choices': [{'delta': {'content': 'detailed response'}, 'finish_reason': 'stop'}]}
            def close(self):
                pass

        model = LocalModel(context_size=1024)
        self.assertEqual(model.temperature, 0.7)
        self.assertEqual(model.repeat_penalty, 1.25)
        self.assertEqual(model.default_max_tokens, 512)

        model._model = MockSamplingLlama()
        messages = [{'role': 'user', 'content': 'explain recursion'}]

        # Default run inherits anti-repetition defaults and 512 tokens
        ''.join(model.chat_stream(messages))
        self.assertEqual(seen_kwargs['temperature'], 0.7)
        self.assertEqual(seen_kwargs['repeat_penalty'], 1.25)
        self.assertEqual(seen_kwargs['max_tokens'], 512)

        # Explicit overrides work as expected
        ''.join(model.chat_stream(messages, max_tokens=768, temperature=0.85, repeat_penalty=1.35))
        self.assertEqual(seen_kwargs['temperature'], 0.85)
        self.assertEqual(seen_kwargs['repeat_penalty'], 1.35)
        self.assertEqual(seen_kwargs['max_tokens'], 768)
