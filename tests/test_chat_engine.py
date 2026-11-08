"""Local model streaming and explicitly selected research/optional backends."""
import unittest
from hari_ai.memory.hierarchy import HierarchicalMemoryManager
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings


class FakeNano:
    checkpoint_loaded = True
    def __init__(self, output='Nano response'):
        self.output = output
        self.prompts = []
    def generate(self, prompt, config):
        self.prompts.append(prompt)
        return self.output


class FakeOllama:
    def __init__(self):
        self.calls = []
    def chat(self, messages):
        self.calls.append(messages)
        return 'Ollama response'


class FakeLocal:
    def __init__(self):
        self.calls = []
    def chat_stream(self, messages):
        self.calls.append(messages)
        yield 'Hello '
        yield 'from HariAI.'


class TestChatEngine(unittest.TestCase):
    def setUp(self):
        self.nano, self.ollama = FakeNano(), FakeOllama()
        self.chat = ChatEngine(memory_manager=HierarchicalMemoryManager(),
            inference_engine=self.nano, ollama_client=self.ollama, settings=ChatSettings(backend='nano'))

    def test_explicit_research_nano_generates_response(self):
        result = ''.join(self.chat.chat_stream('Explain recursion'))
        self.assertEqual(result, 'Nano response')
        self.assertEqual(self.nano.prompts, ['User: Explain recursion\nHariAI:'])
        self.assertEqual(self.ollama.calls, [])
        self.assertNotIn('confidence', result.lower())

    def test_default_local_streams_and_retains_history_without_ollama(self):
        local = FakeLocal()
        chat = ChatEngine(settings=ChatSettings(), local_model=local, ollama_client=self.ollama)
        self.assertEqual(list(chat.chat_stream('Hello')), ['Hello ', 'from HariAI.'])
        self.assertEqual(''.join(chat.chat_stream('Tell me more')), 'Hello from HariAI.')
        self.assertEqual(local.calls[-1][-2], {'role': 'assistant', 'content': 'Hello from HariAI.'})
        self.assertEqual(self.ollama.calls, [])
        self.assertIn('K R Hari Prajwal', chat.system_prompt)

    def test_local_failure_does_not_pollute_history(self):
        class Missing:
            def chat_stream(self, messages):
                raise FileNotFoundError('missing model')
        chat = ChatEngine(settings=ChatSettings(), local_model=Missing())
        self.assertIn('missing model', ''.join(chat.chat_stream('Hello')))
        self.assertEqual(len(chat.history), 1)

    def test_ollama_only_when_selected_and_keeps_context(self):
        self.chat.settings.backend = 'ollama'
        self.assertEqual(''.join(self.chat.chat_stream('Hello')), 'Ollama response')
        self.assertEqual(''.join(self.chat.chat_stream('Continue')), 'Ollama response')
        self.assertEqual(self.ollama.calls[-1][-2]['content'], 'Ollama response')
        self.assertEqual(self.ollama.calls[-1][-1]['content'], 'Continue')
        self.assertEqual(self.nano.prompts, [])

    def test_unusable_nano_output_is_not_presented_as_answer(self):
        self.nano.output = '     '
        result = ''.join(self.chat.chat_stream('Explain recursion'))
        self.assertIn('could not produce a usable answer', result)
        self.assertEqual(self.ollama.calls, [])

    def test_missing_checkpoint_is_explained(self):
        self.nano.checkpoint_loaded = False
        self.assertIn('checkpoint not found', ''.join(self.chat.chat_stream('Hello')))

    def test_creator_and_clear(self):
        self.assertIn('K R Hari Prajwal', ''.join(self.chat.chat_stream('Who created you?')))
        self.chat.clear_history()
        self.assertEqual(len(self.chat.history), 1)
        self.assertEqual(self.chat.get_user_turns(), [])


if __name__ == '__main__':
    unittest.main()
