"""Measure base-model likelihood of emitted tokens, not answer correctness.

llama-cpp-python 0.3.8 exposes final logits through its context. Capturing only
the current token avoids retaining a context-by-vocabulary logits matrix. If
the installed runtime lacks this interface, collection fails closed and chat
continues normally.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class TokenLikelihoodRecorder:
    """Observe sampling without changing logits or the model's chosen token."""

    def __init__(self, model):
        self.model = model
        self.tokens: list[int] = []
        self.log_probabilities: list[float] = []
        self.failed = False
        self._original_sample = None
        self._had_instance_sample = False
        self._instance_sample = None
        self._installed = False

    def __enter__(self):
        try:
            context = self.model._ctx
            if not callable(context.get_logits_ith) or not callable(self.model.sample):
                raise AttributeError('Token logits are unavailable.')
            self._had_instance_sample = 'sample' in vars(self.model)
            self._instance_sample = vars(self.model).get('sample')
            self._original_sample = self.model.sample
            self.model.sample = self._sample
            self._installed = True
        except (AttributeError, TypeError, RuntimeError, OSError):
            self.failed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._installed:
            if self._had_instance_sample:
                self.model.sample = self._instance_sample
            else:
                del self.model.sample
        return False

    def _sample(self, *args, **kwargs):
        # Sampling is delegated first and is never affected by observation.
        token = self._original_sample(*args, **kwargs)
        if not self.failed:
            try:
                idx = kwargs.get('idx')
                relative = idx - self.model.n_tokens if idx is not None else -1
                pointer = self.model._ctx.get_logits_ith(relative)
                if not pointer:
                    raise ValueError('Runtime returned no token logits.')
                logits = np.ctypeslib.as_array(pointer, shape=(self.model.n_vocab(),))
                maximum = float(np.max(logits))
                selected = float(logits[token])
                if not math.isfinite(maximum) or not math.isfinite(selected):
                    raise ValueError('Runtime returned non-finite token logits.')
                log_probability = selected - maximum - math.log(float(np.exp(logits-maximum).sum()))
                if not math.isfinite(log_probability) or log_probability > 1e-6:
                    raise ValueError('Invalid token log probability.')
                self.tokens.append(int(token))
                self.log_probabilities.append(min(0.0, log_probability))
            except (AttributeError, TypeError, ValueError, IndexError, OverflowError,
                    FloatingPointError, RuntimeError, OSError):
                self.failed = True
                self.tokens.clear()
                self.log_probabilities.clear()
        return token

    def mean_probability(self, visible_text: str) -> Optional[float]:
        """Return the average emitted-token probability, or abstain on mismatch."""
        if self.failed or not visible_text or not self.tokens:
            return None
        tokens = list(self.tokens)
        scores = list(self.log_probabilities)
        try:
            stop_ids = {self.model.token_eos()}
            eot = getattr(getattr(self.model, '_model', None), 'token_eot', None)
            if callable(eot):
                stop_ids.add(eot())
            while tokens and tokens[-1] in stop_ids:
                tokens.pop()
                scores.pop()
            if not tokens or self.model.detokenize(tokens) != visible_text.encode('utf-8'):
                return None
            return sum(math.exp(value) for value in scores)/len(scores)
        except (AttributeError, TypeError, ValueError, UnicodeError, RuntimeError, OSError):
            return None
