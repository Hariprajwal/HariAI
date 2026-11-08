"""Standalone GGUF inference: no server, Ollama, or network calls."""
from __future__ import annotations

import os
import atexit
import threading
from pathlib import Path
from contextlib import nullcontext
from typing import Iterator


def bundled_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'assets' / 'hariai-chat.gguf'


class LocalModel:
    def __init__(self, path: str = '', *, context_size: int = 2048, threads: int = 4):
        self.path = Path(path).expanduser() if path else bundled_model_path()
        self.context_size = context_size
        self.threads = threads
        self._model = None
        self.last_finish_reason = None
        self.last_token_likelihood = None
        self.measure_token_likelihood = True
        self._token_likelihood_compatible = None
        self._generation_lock = threading.Lock()

    def _load(self):
        if self._model is None:
            if not self.path.is_file():
                raise FileNotFoundError('HariAI model is missing. Install the model-bearing wheel or '
                                        'use `hari settings model-file PATH.gguf`.')
            try:
                import llama_cpp
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError('Install llama-cpp-python to run the local model.') from exc
            # The lightweight observer uses low-level sampling details verified
            # against the public logprob API for this exact version. Unknown
            # versions can still chat, but do not get an unverified score.
            self._token_likelihood_compatible = getattr(llama_cpp, '__version__', '') == '0.3.8'
            self._model = Llama(model_path=str(self.path), n_ctx=self.context_size,
                n_threads=min(self.threads, os.cpu_count() or 1), verbose=False,
                chat_format='chatml', seed=42)
            atexit.register(self.close)
        return self._model

    def _fit_messages(self, messages: list[dict[str, str]], max_tokens: int):
        """Drop complete oldest turns, retaining system and the current user turn."""
        model = self._load()
        kept = [dict(item) for item in messages]
        while True:
            # Same ChatML delimiters as llama-cpp-python's chatml formatter.
            formatted = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in kept)
            formatted += '<|im_start|>assistant\n'
            count = len(model.tokenize(formatted.encode('utf-8'), special=True))
            if count + max_tokens + 8 <= self.context_size:
                return kept
            if len(kept) <= 2:
                raise ValueError('Your message is too long for this model. Shorten it or start a new chat.')
            # The chat history consists of system, user/assistant pairs, latest user.
            del kept[1:3]

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int = 192) -> Iterator[str]:
        # llama-cpp contexts and the sampling observer are not safe to share
        # across overlapping generations. Separate LocalModel instances may run
        # concurrently; reject overlap on this one rather than corrupting it.
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError('This local model is already generating a response.')
        try:
            yield from self._chat_stream_locked(messages, max_tokens)
        finally:
            self._generation_lock.release()

    def _chat_stream_locked(self, messages: list[dict[str, str]], max_tokens: int) -> Iterator[str]:
        self.last_finish_reason = None
        self.last_token_likelihood = None
        if not 1 <= max_tokens < self.context_size:
            raise ValueError('max_tokens must fit inside the context window.')
        fitted = self._fit_messages(messages, max_tokens)
        model = self._load()
        if self.measure_token_likelihood and self._token_likelihood_compatible is not False:
            from hari_ai.runtime.token_likelihood import TokenLikelihoodRecorder
            recorder_context = TokenLikelihoodRecorder(model)
        else:
            recorder_context = nullcontext(None)
        pieces = []
        with recorder_context as recorder:
            chunks = model.create_chat_completion(messages=fitted, max_tokens=max_tokens,
                temperature=0.2, top_p=0.9, repeat_penalty=1.1, stream=True)
            for chunk in chunks:
                choice = chunk['choices'][0]
                if choice.get('finish_reason'):
                    self.last_finish_reason = choice['finish_reason']
                text = choice.get('delta', {}).get('content', '')
                if text:
                    pieces.append(text)
                    yield text
        if recorder is not None:
            self.last_token_likelihood = recorder.mean_probability(''.join(pieces))

    def close(self):
        if self._model is not None:
            self._model.close()
            self._model = None
