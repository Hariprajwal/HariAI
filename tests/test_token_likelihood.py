"""The displayed signal is observed token probability, never answer accuracy."""
import ctypes
import math
import unittest
from pathlib import Path

import numpy as np

from hari_ai.runtime.token_likelihood import TokenLikelihoodRecorder


class FakeContext:
    def __init__(self):
        self.logits = np.array([0.0, math.log(2), 0.0], dtype=np.float32)

    def get_logits_ith(self, index):
        if index != -1:
            raise ValueError('Unexpected logits row')
        return self.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class FakeModel:
    def __init__(self):
        self._ctx = FakeContext()
        self.n_tokens = 4
        self.next_tokens = iter([0, 1, 2])

    def sample(self, **kwargs):
        return next(self.next_tokens)

    def n_vocab(self):
        return 3

    def token_eos(self):
        return 2

    def detokenize(self, tokens):
        return b''.join({0:b'A', 1:b'B', 2:b''}[x] for x in tokens)


class TokenLikelihoodTests(unittest.TestCase):
    def test_emitted_tokens_use_model_probabilities_and_exclude_eos(self):
        model = FakeModel()
        with TokenLikelihoodRecorder(model) as recorder:
            self.assertEqual([model.sample(idx=3) for _ in range(3)], [0, 1, 2])
        self.assertAlmostEqual(recorder.mean_probability('AB'), .375, places=6)
        self.assertNotIn('sample', vars(model))
        self.assertIsNone(recorder.mean_probability('different text'))

    def test_restores_sampler_after_exception(self):
        model = FakeModel()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with TokenLikelihoodRecorder(model):
                model.sample(idx=3)
                raise RuntimeError('interrupted')
        self.assertNotIn('sample', vars(model))

    def test_nonfinite_logits_and_unsupported_runtime_do_not_guess(self):
        model = FakeModel()
        model._ctx.logits[0] = float('nan')
        with TokenLikelihoodRecorder(model) as recorder:
            self.assertEqual(model.sample(idx=3), 0)
        self.assertIsNone(recorder.mean_probability('A'))
        with TokenLikelihoodRecorder(object()) as unsupported:
            self.assertTrue(unsupported.failed)
        self.assertIsNone(unsupported.mean_probability('answer'))

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'hari_ai/assets/hariai-chat.gguf').is_file(),
                         'local GGUF research model is not installed')
    def test_real_model_matches_public_logprob_api(self):
        from llama_cpp import Llama
        from hari_ai.runtime.local_model import LocalModel, bundled_model_path
        messages = [{'role':'user', 'content':'hello'}]
        local = LocalModel(context_size=512)
        try:
            actual_text = ''.join(local.chat_stream(messages, max_tokens=64))
            actual = local.last_token_likelihood
        finally:
            local.close()
        reference = Llama(model_path=str(bundled_model_path()), n_ctx=512, n_threads=4,
                          chat_format='chatml', logits_all=True, verbose=False, seed=42)
        try:
            probabilities, text = [], []
            for chunk in reference.create_chat_completion(messages=messages, max_tokens=64,
                    temperature=.2, top_p=.9, repeat_penalty=1.1, stream=True,
                    logprobs=True, top_logprobs=1):
                choice = chunk['choices'][0]
                text.append(choice.get('delta', {}).get('content', '') or '')
                probabilities.extend(math.exp(item['logprob']) for item in
                                     ((choice.get('logprobs') or {}).get('content') or []))
            self.assertEqual(actual_text, ''.join(text))
            self.assertTrue(probabilities)
            self.assertIsNotNone(actual)
            self.assertAlmostEqual(actual, sum(probabilities)/len(probabilities), places=5)
        finally:
            reference.close()
