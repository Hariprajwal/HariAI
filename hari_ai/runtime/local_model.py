"""Standalone GGUF inference: no server, Ollama, or network calls."""
from __future__ import annotations

import os
import atexit
from pathlib import Path
from typing import Iterator


def bundled_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'assets' / 'hariai-chat.gguf'


class LocalModel:
    def __init__(self, path: str = '', *, context_size: int = 2048, threads: int = 4):
        self.path = Path(path).expanduser() if path else bundled_model_path()
        self.context_size = context_size
        self.threads = threads
        self._model = None

    def _load(self):
        if self._model is None:
            if not self.path.is_file():
                raise FileNotFoundError('HariAI model is missing. Install the model-bearing wheel or '
                                        'use `hari settings model-file PATH.gguf`.')
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError('Install llama-cpp-python to run the local model.') from exc
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
        if not 1 <= max_tokens < self.context_size:
            raise ValueError('max_tokens must fit inside the context window.')
        fitted = self._fit_messages(messages, max_tokens)
        chunks = self._load().create_chat_completion(messages=fitted, max_tokens=max_tokens,
            temperature=0.2, top_p=0.9, repeat_penalty=1.1, stream=True)
        for chunk in chunks:
            text = chunk['choices'][0].get('delta', {}).get('content', '')
            if text:
                yield text

    def close(self):
        if self._model is not None:
            self._model.close()
            self._model = None
