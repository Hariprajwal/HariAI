"""Unit tests for the HariAI Conversational Multi-Turn Chat Engine."""

import unittest
from hari_ai.memory.hierarchy import HierarchicalMemoryManager
from hari_ai.runtime.chat import ChatEngine, ChatMessage


class TestChatEngine(unittest.TestCase):
    def setUp(self):
        self.mem = HierarchicalMemoryManager(session_id="test_chat_session")
        self.chat = ChatEngine(memory_manager=self.mem)
        self.chat.ollama.chat = lambda messages: None

    def test_initial_state(self):
        self.assertEqual(len(self.chat.history), 1)
        self.assertEqual(self.chat.history[0].role, "system")

    def test_identity_question(self):
        tokens = list(self.chat.chat_stream("Who created you?"))
        response = "".join(tokens)
        self.assertIn("K R Hari Prajwal", response)
        self.assertIn("21 years old", response)
        self.assertEqual(len(self.chat.history), 3)  # system + user + assistant

    def test_casual_greeting(self):
        tokens = list(self.chat.chat_stream("Hello!"))
        response = "".join(tokens)
        self.assertIn("HariAI", response)

    def test_code_generation(self):
        tokens = list(self.chat.chat_stream("Write a python function to reverse a list"))
        response = "".join(tokens)
        self.assertIn("couldn't reach an installed Ollama", response)
        self.assertNotIn("<|verify|>pass", response)

    def test_math_reasoning(self):
        tokens = list(self.chat.chat_stream("Calculate the derivative of x^2"))
        response = "".join(tokens)
        self.assertIn("couldn't reach an installed Ollama", response)

    def test_decision_evaluation(self):
        tokens = list(self.chat.chat_stream("Should I delete the production database?"))
        response = "".join(tokens)
        self.assertIn("couldn't reach an installed Ollama", response)

    def test_multi_turn_context_retention(self):
        list(self.chat.chat_stream("What is quantum computing?"))
        list(self.chat.chat_stream("Can you elaborate?"))
        user_turns = self.chat.get_user_turns()
        self.assertEqual(len(user_turns), 2)
        self.assertEqual(user_turns[0], "What is quantum computing?")
        self.assertEqual(user_turns[1], "Can you elaborate?")

        # Check asking about earlier question
        tokens = list(self.chat.chat_stream("What was my first question?"))
        response = "".join(tokens)
        self.assertIn("What is quantum computing?", response)

    def test_clear_history(self):
        list(self.chat.chat_stream("Hello"))
        self.chat.clear_history()
        self.assertEqual(len(self.chat.history), 1)
        self.assertEqual(self.chat.history[0].role, "system")


if __name__ == "__main__":
    unittest.main()

