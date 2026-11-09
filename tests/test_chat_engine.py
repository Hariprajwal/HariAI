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
        self.assertFalse(chat.last_reply_succeeded)
        self.assertEqual(chat.memory.working.scratchpad, [])

    def test_local_replies_record_session_experiences_only_on_completion(self):
        chat = ChatEngine(settings=ChatSettings(), local_model=FakeLocal())
        stream = chat.chat_stream('hello')
        next(stream)
        stream.close()
        self.assertEqual(len(chat.history), 1)
        self.assertEqual(chat.memory.working.scratchpad, [])
        self.assertEqual(chat.memory.episodic.entries, [])
        self.assertFalse(chat.last_reply_succeeded)
        ''.join(chat.chat_stream('hello again'))
        self.assertEqual(len(chat.memory.episodic.entries), 1)
        self.assertEqual(chat.memory.episodic.entries[0].assistant_output, 'Hello from HariAI.')
        self.assertFalse(chat.memory.episodic.entries[0].was_successful)

    def test_identity_shorthand_does_not_use_model(self):
        local = FakeLocal()
        chat = ChatEngine(settings=ChatSettings(), local_model=local)
        answer = ''.join(chat.chat_stream('who are u?'))
        self.assertIn('HariAI', answer)
        self.assertIn('K R Hari Prajwal', answer)
        self.assertTrue(chat.last_reply_succeeded)
        self.assertEqual(local.calls, [])

    def test_token_likelihood_is_not_reused_for_a_non_model_reply(self):
        local = FakeLocal()
        local.last_token_likelihood = .62
        local.last_finish_reason = 'stop'
        chat = ChatEngine(settings=ChatSettings(), local_model=local)
        ''.join(chat.chat_stream('Hello'))
        self.assertEqual(chat.last_token_likelihood, .62)
        ''.join(chat.chat_stream('who are u?'))
        self.assertIsNone(chat.last_token_likelihood)

    def test_vague_c_program_request_asks_for_the_missing_task(self):
        local = FakeLocal()
        chat = ChatEngine(settings=ChatSettings(), local_model=local)
        answer = ''.join(chat.chat_stream('ok give me the c program !?'))
        self.assertIn('What should the C program do?', answer)
        self.assertEqual(local.calls, [])
        ''.join(chat.chat_stream('write a C program that implements binary search over a sorted list'))
        self.assertEqual(len(local.calls), 1)
        ''.join(chat.chat_stream('generate the full c program'))
        self.assertEqual(len(local.calls), 2)

    def test_direct_reply_clears_previous_token_limit_status(self):
        local = FakeLocal()
        local.last_finish_reason = 'length'
        chat = ChatEngine(settings=ChatSettings(), local_model=local)
        ''.join(chat.chat_stream('who are u?'))
        self.assertIsNone(local.last_finish_reason)

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

    def test_clean_repetitive_history_deduplicates_consecutive_assistant_replies(self):
        chat = ChatEngine(settings=ChatSettings())
        raw = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'What a great question!'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'What a great question!'},
            {'role': 'assistant', 'content': 'what a great question!'},
            {'role': 'user', 'content': 'q3'},
            {'role': 'assistant', 'content': 'A different distinct answer.'},
        ]
        cleaned = chat._clean_repetitive_history(raw)
        assistant_contents = [m['content'] for m in cleaned if m['role'] == 'assistant']
        # Repeated loop outputs deduplicated, distinct answer preserved
        self.assertEqual(assistant_contents, ['What a great question!', 'A different distinct answer.'])

    def test_loop_detection_filter_resamples_with_higher_temperature(self):
        from hari_ai.runtime.local_model import LocalModel

        class MockRepeatingLocalModel(LocalModel):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.call_count = 0

            def chat_stream(self, messages, max_tokens=None, temperature=None, repeat_penalty=None):
                self.calls.append({
                    'messages': messages,
                    'temperature': temperature,
                    'repeat_penalty': repeat_penalty,
                })
                self.call_count += 1
                if self.call_count == 1:
                    # Duplicate of the previous turn
                    yield "What a great question!"
                else:
                    # Fresh resampled output
                    yield "Here is the newly formulated answer."

        mock_local = MockRepeatingLocalModel()
        chat = ChatEngine(settings=ChatSettings(backend='local'), local_model=mock_local)
        # Seed history with a previous assistant reply
        from hari_ai.runtime.chat import ChatMessage
        chat.history.append(ChatMessage(role='user', content='what can you do?'))
        chat.history.append(ChatMessage(role='assistant', content='What a great question!'))

        # Now ask a new question
        result = ''.join(chat.chat_stream('tell me your capabilities'))

        # Loop detection should reject 'What a great question!' and return the resampled output
        self.assertEqual(result, 'Here is the newly formulated answer.')
        self.assertEqual(mock_local.call_count, 2)
        # Second call used elevated temperature and repeat penalty
        self.assertEqual(mock_local.calls[1]['temperature'], 0.85)
        self.assertEqual(mock_local.calls[1]['repeat_penalty'], 1.35)
        # Purged repetitive assistant messages from resample call
        resample_assistant_turns = [
            m for m in mock_local.calls[1]['messages'] if m['role'] == 'assistant'
        ]
        self.assertEqual(resample_assistant_turns, [])


if __name__ == '__main__':
    unittest.main()
