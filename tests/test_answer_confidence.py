"""Calibration gates, Open-JEV head semantics, and chat integration."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hari_ai.runtime.answer_confidence import (OpenJevAnswerConfidence, answer_probability_from_head, arithmetic_state,
    sha256_file, stated_integer)
from hari_ai.runtime.calibration import (brier_improvement_interval, evaluate_calibrator,
    platt_probability, wilson_accuracy_interval)
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def rows(prefix, count=120):
        return [{'group_id':f'{prefix}-{i}', 'split':prefix, 'label':i%2,
                 'probability':.8 if i%2 else .2} for i in range(count)]

    def test_fit_only_promotes_disjoint_sufficient_heldout_evidence(self):
        calibration, test = self.rows('calibration'), self.rows('test')
        result = evaluate_calibrator(calibration, test, predeclared=True)
        self.assertTrue(result['eligible'])
        self.assertGreater(result['brier_improvement_interval']['lower_95'], 0)
        self.assertEqual(result['test']['auroc'], 1.0)
        self.assertGreater(platt_probability(.8, result['slope'], result['intercept']), .8)
        self.assertFalse(evaluate_calibrator(calibration[:20], test, predeclared=True)['eligible'])
        self.assertFalse(evaluate_calibrator(calibration, test, predeclared=False)['eligible'])
        with self.assertRaises(ValueError):
            evaluate_calibrator(calibration, [calibration[0]] + test, predeclared=True)

    def test_bootstrap_identifies_good_and_uninformative_scores(self):
        rows = self.rows('heldout')
        interval = brier_improvement_interval(rows, baseline=.5, slope=1, intercept=0, samples=300)
        self.assertGreater(interval['lower_95'], 0)
        constant = [{**row, 'probability':.5} for row in rows]
        no_gain = brier_improvement_interval(constant, baseline=.5, slope=1, intercept=0, samples=300)
        self.assertEqual(no_gain['lower_95'], 0)
        with self.assertRaises(ValueError):
            brier_improvement_interval([rows[0], rows[0]], baseline=.5, slope=1, intercept=0)

    def test_small_bin_reliability_includes_uncertainty(self):
        lower, upper = wilson_accuracy_interval(10, 10)
        self.assertLess(lower, .8)
        self.assertAlmostEqual(upper, 1.0)
        lower, upper = wilson_accuracy_interval(0, 10)
        self.assertAlmostEqual(lower, 0.0)
        self.assertGreater(upper, .2)
        with self.assertRaises(ValueError):
            wilson_accuracy_interval(11, 10)

    def test_scope_and_head_orientation(self):
        prompt = 'What is 2 + 3? Answer with only the number.'
        self.assertEqual(stated_integer(prompt, '2 + 3 = 5.'), 5)
        self.assertEqual(stated_integer(prompt, 'The answer is 0.'), 0)
        self.assertIsNone(stated_integer(prompt, 'The answer is 5, and water is red.'))
        self.assertIsNone(stated_integer('What is 3 + 2? Answer with only the number.', '5'))
        self.assertIsNone(stated_integer('What is 30 - 1? Answer with only the number.', '29'))
        self.assertEqual(arithmetic_state(prompt, '5')['answer']['units'], '5')
        self.assertEqual(arithmetic_state(prompt, '6')['answer']['units'], '6')
        self.assertAlmostEqual(answer_probability_from_head('correct', .9), .9)
        self.assertAlmostEqual(answer_probability_from_head('incorrect', .9), .1)
        with self.assertRaises(ValueError):
            answer_probability_from_head('correct', float('nan'))

    def test_trusted_artifact_is_checked_against_model_prompt_and_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            checkpoint, model = directory/'verifier.pt', directory/'model.gguf'
            checkpoint.write_bytes(b'verifier')
            model.write_bytes(b'model')
            system = 'test system prompt'
            cal, test = self.rows('calibration'), self.rows('test')
            result = evaluate_calibrator(cal,test,predeclared=True)
            bundle = {'format':'hariai_openjev_answer_calibration_v1',
                'domain':'single_turn_integer_arithmetic_v1', 'state_encoding':'arithmetic_digits_v1',
                'score_semantics':'jev_choice_confidence_oriented_to_answer_correct_v1',
                'source':'model_generated_independently_graded', 'problem_grouping':'commutative_addition_v1',
                'predeclared_split':True, 'model_sha256':sha256_file(model),
                'system_prompt_sha256':hashlib.sha256(system.encode()).hexdigest(),
                'protocol':{'backend':'local','max_tokens':192,'temperature':0.2,'top_p':0.9,
                            'repeat_penalty':1.1,'history':'single_turn'},
                'verifier_file':checkpoint.name,'verifier_sha256':sha256_file(checkpoint),
                'slope':result['slope'],'intercept':result['intercept'],'rows':cal+test}
            artifact = directory/'confidence.json'
            artifact.write_text(json.dumps(bundle),encoding='utf-8')
            with patch('hari_ai.runtime.openjev.OpenJevAdapter') as adapter:
                adapter.return_value.checkpoint_scope = 'single_turn_integer_arithmetic_v1'
                adapter.return_value.state_encoding = 'arithmetic_digits_v1'
                adapter.return_value.decide.return_value = {'choice':'correct','confidence_raw':.8}
                verifier = OpenJevAnswerConfidence(str(artifact),model,system)
                answer = verifier.estimate('What is 2 + 3? Answer with only the number.', '5', earlier_turns=0, finish_reason='stop')
                self.assertGreater(answer, .8)
                self.assertIsNone(verifier.estimate('How are you?', '5', earlier_turns=0, finish_reason='stop'))
                self.assertIsNone(verifier.estimate('What is 2 + 3? Answer with only the number.', '5', earlier_turns=1, finish_reason='stop'))
                self.assertIsNone(verifier.estimate('What is 2 + 3? Answer with only the number.', '5', earlier_turns=0, finish_reason='length'))
            with self.assertRaisesRegex(ValueError, 'prompt'):
                OpenJevAnswerConfidence(str(artifact),model,'different prompt')
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'weights'):
                OpenJevAnswerConfidence(str(artifact),model,system)

    def test_chat_uses_verified_score_only_for_supported_response(self):
        class Local:
            path = Path('model.gguf')
            last_finish_reason = None
            def chat_stream(self, messages):
                yield '5'
                self.last_finish_reason = 'stop'
        settings = ChatSettings(answer_confidence_artifact='test-artifact.json')
        chat = ChatEngine(settings=settings,local_model=Local())
        self.assertEqual(''.join(chat.chat_stream('What is 2 + 3? Answer with only the number.')), '5')
        self.assertEqual(chat.last_answer_confidence, 1.0)
        self.assertEqual(chat.last_confidence_source, 'verified arithmetic')
        self.assertEqual(chat.history[-1].confidence, 1.0)
        ''.join(chat.chat_stream('What is 2 + 3? Answer with only the number.'))
        self.assertIsNone(chat.last_answer_confidence)


if __name__ == '__main__':
    unittest.main()
