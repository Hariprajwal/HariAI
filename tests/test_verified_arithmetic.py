import unittest

from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings
from hari_ai.runtime.verified_arithmetic import verify_arithmetic


class ArithmeticVerificationTests(unittest.TestCase):
    def test_correct_and_incorrect_results_across_signs_and_operations(self):
        operations = {'+': lambda a,b:a+b, '-':lambda a,b:a-b, '*':lambda a,b:a*b}
        for a in range(-12, 13):
            for b in range(-12, 13):
                for symbol, operation in operations.items():
                    prompt = f'What is {a} {symbol} {b}?'
                    result = operation(a, b)
                    self.assertIs(verify_arithmetic(prompt, str(result)), True)
                    self.assertIs(verify_arithmetic(prompt, str(result+1)), False)

    def test_rejects_extra_claims_ambiguity_and_unsupported_math(self):
        cases = [
            ('What is 2 + 2?', '4, and the Earth is flat.'),
            ('What is 2 + 2 and 3 + 4?', '4'),
            ('What is 2 + 2?', '2 + 3 = 4'),
            ('What is 2 + 2?', '4 or 5'),
            ('What is 2 + 2?', '2 + 2 = 4\nThe answer is 5.'),
            ('What is 2 + 2?', '2 + 3 = 4\nThe answer is 4.'),
            ('What is 2 + 2?', 'Here is the answer:\n4\nWater is red.'),
            ('What is 2 / 2?', '1'),
            ('What is 1.2 + 2?', '3'),
            ('hello', '4'),
        ]
        for prompt, answer in cases:
            self.assertIsNone(verify_arithmetic(prompt, answer), (prompt, answer))

    def test_repeated_identical_arithmetic_answers_are_verified(self):
        prompt = 'What is 32 * 1? Answer with only the number.'
        self.assertIs(verify_arithmetic(prompt, '32 * 1 = 32\nThe answer is 32.'), True)
        self.assertIs(verify_arithmetic(prompt, 'Here is the answer:\n32 * 1 = 33'), False)

    def test_chat_marks_only_complete_single_turn_answers(self):
        class Local:
            last_finish_reason = None
            def __init__(self, answer, finish):
                self.answer, self.finish = answer, finish
            def chat_stream(self, messages):
                yield self.answer
                self.last_finish_reason = self.finish
        chat = ChatEngine(settings=ChatSettings(), local_model=Local('5', 'stop'))
        self.assertEqual(''.join(chat.chat_stream('What is 2 + 2?')), '5')
        self.assertEqual(chat.last_answer_confidence, 0.0)
        self.assertEqual(chat.last_confidence_source, 'verified arithmetic')
        ''.join(chat.chat_stream('What is 2 + 2?'))
        self.assertIsNone(chat.last_answer_confidence)
        chat = ChatEngine(settings=ChatSettings(), local_model=Local('4', 'length'))
        ''.join(chat.chat_stream('What is 2 + 2?'))
        self.assertIsNone(chat.last_answer_confidence)
        chat = ChatEngine(settings=ChatSettings(), local_model=Local('4, plus another claim.', 'stop'))
        ''.join(chat.chat_stream('What is 2 + 2?'))
        self.assertIsNone(chat.last_answer_confidence)


if __name__ == '__main__':
    unittest.main()
