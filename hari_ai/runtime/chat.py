"""Conversational chat using the explicitly selected local backend."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager
from hari_ai.profile.creator import get_creator_profile
from hari_ai.runtime.engine import GenerationConfig, InferenceEngine
from hari_ai.runtime.ollama import OllamaClient
from hari_ai.runtime.settings import ChatSettings, load_settings


@dataclass
class ChatMessage:
    role: str
    content: str
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class ChatEngine:
    """Manage a conversation; HariNano is the default backend."""

    def __init__(self, memory_manager: Optional[HierarchicalMemoryManager] = None,
                 system_prompt: Optional[str] = None,
                 inference_engine: Optional[InferenceEngine] = None,
                 ollama_client: Optional[OllamaClient] = None,
                 settings: Optional[ChatSettings] = None):
        self.memory = memory_manager or HierarchicalMemoryManager()
        self.creator_profile = get_creator_profile()
        self.settings = settings or load_settings()
        self.settings.validate()
        self.engine = inference_engine
        if self.engine is None and self.settings.backend == 'nano':
            self.engine = InferenceEngine(model_path_or_name=self.settings.nano_checkpoint or 'nano',
                                          tokenizer_path=self.settings.nano_tokenizer or None)
        self.ollama = ollama_client or OllamaClient(
            endpoint=self.settings.ollama_host, model=self.settings.ollama_model or None)
        self.system_prompt = system_prompt or 'You are HariAI, a helpful assistant.'
        self.history: List[ChatMessage] = [ChatMessage(role='system', content=self.system_prompt)]

    def clear_history(self) -> None:
        self.history = [ChatMessage(role='system', content=self.system_prompt)]
        self.memory.working.clear()

    def get_user_turns(self) -> List[str]:
        return [message.content for message in self.history if message.role == 'user']

    def chat_stream(self, user_input: str) -> Iterator[str]:
        prompt = user_input.strip()
        if not prompt:
            return
        self.history.append(ChatMessage(role='user', content=prompt))
        self.memory.working.append_thought(f'User: {prompt}')

        identity = self.creator_profile.answer_identity_question(prompt)
        if identity:
            answer = identity
        elif self.settings.backend == 'ollama':
            messages = [{'role': 'system', 'content': self.system_prompt}] + [
                {'role': message.role, 'content': message.content}
                for message in self.history[-24:] if message.role in ('user', 'assistant')
            ]
            answer = self.ollama.chat(messages)
            if not answer:
                answer = ('Could not reach the selected Ollama model. Check `ollama list` and '
                          '`hari settings show`, or switch back with `hari settings backend nano`.')
        elif not getattr(self.engine, 'checkpoint_loaded', False):
            answer = ('HariNano checkpoint not found. Configure one with `hari settings nano-model CHECKPOINT TOKENIZER`. '
                      'The current research checkpoints are not yet capable general chat models.')
        else:
            # Match the prompt format used by the prototype training script.
            answer = self.engine.generate('User: ' + prompt + '\nHariAI:',
                                          GenerationConfig(max_new_tokens=96, temperature=0.0)).strip()
            if not answer or not any(ch.isalnum() for ch in answer):
                answer = ('HariNano could not produce a usable answer. Its current checkpoint is an early '
                          'research model and needs substantially more language and instruction training. '
                          'You can optionally configure Ollama with `hari settings backend ollama`.')

        self.history.append(ChatMessage(role='assistant', content=answer))
        self.memory.episodic.record(EpisodicEntry(entry_id=f'chat-{int(time.time() * 1000)}',
            timestamp=time.time(), user_input=prompt, assistant_output=answer, action_taken='chat',
            was_successful=False))
        yield answer
