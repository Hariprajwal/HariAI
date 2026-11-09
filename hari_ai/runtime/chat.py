"""Conversational chat using the explicitly selected local backend."""
from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager
from hari_ai.profile.creator import get_creator_profile
from hari_ai.runtime.ollama import OllamaClient
from hari_ai.runtime.settings import ChatSettings, load_settings
from hari_ai.runtime.local_model import LocalModel

if TYPE_CHECKING:
    from hari_ai.runtime.engine import InferenceEngine
else:
    InferenceEngine = Any


@dataclass
class ChatMessage:
    role: str
    content: str
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


def needs_c_program_details(prompt: str, earlier_user_turns: List[str]) -> bool:
    """Ask for a task only when the request contains no program specification."""
    text = ' '.join(re.sub(r'[^a-z0-9\s]', ' ', prompt.lower()).split())
    vague = re.fullmatch(
        r'(?:ok(?:ay)? )?(?:give|show|write|generate|create)(?: me)? '
        r'(?:(?:the|a|an) )?(?:(?:complete|ocmeplete|full) )?c (?:program|code)(?: please)?', text)
    if not vague:
        return False
    # If an earlier turn specified a programming problem, keep the conversation
    # intact and let the model see that context.
    for earlier in earlier_user_turns:
        words = re.findall(r'[a-z0-9]+', earlier.lower())
        if any(word in ('program', 'code', 'function') for word in words) and len(words) > 8:
            return False
    return True


class ChatEngine:
    """Manage a conversation using the bundled pretrained-derived local model."""

    def __init__(self, memory_manager: Optional[HierarchicalMemoryManager] = None,
                 system_prompt: Optional[str] = None,
                 inference_engine: Optional[InferenceEngine] = None,
                 ollama_client: Optional[OllamaClient] = None,
                 settings: Optional[ChatSettings] = None,
                 local_model: Optional[LocalModel] = None):
        self.memory = memory_manager or HierarchicalMemoryManager()
        self.creator_profile = get_creator_profile()
        self.settings = settings or load_settings()
        self.settings.validate()
        self.engine = inference_engine
        self.local_model = local_model or LocalModel(self.settings.model_file)
        if self.engine is None and self.settings.backend == 'nano':
            from hari_ai.runtime.engine import InferenceEngine
            self.engine = InferenceEngine(model_path_or_name=self.settings.nano_checkpoint or 'nano',
                                          tokenizer_path=self.settings.nano_tokenizer or None)
        self.ollama = ollama_client or OllamaClient(
            endpoint=self.settings.ollama_host, model=self.settings.ollama_model or None)
        self.system_prompt = system_prompt or (
            'You are HariAI, a helpful assistant created by K R Hari Prajwal. '
            'Answer clearly and briefly. Be honest when you do not know.')
        self.history: List[ChatMessage] = [ChatMessage(role='system', content=self.system_prompt)]
        self.last_reply_succeeded = False
        self.last_answer_confidence: Optional[float] = None
        self.last_confidence_source: Optional[str] = None
        self.last_token_likelihood: Optional[float] = None

    def _assess_answer(self, prompt: str, answer: str, earlier_turns: int) -> None:
        """Report only a deterministic verdict in the supported math scope."""
        from hari_ai.runtime.verified_arithmetic import verify_arithmetic
        if not earlier_turns and self.local_model.last_finish_reason == 'stop':
            checked = verify_arithmetic(prompt, answer)
            if checked is not None:
                self.last_answer_confidence = float(checked)
                self.last_confidence_source = 'verified arithmetic'
                return
    def clear_history(self) -> None:
        self.history = [ChatMessage(role='system', content=self.system_prompt)]
        self.memory.working.clear()

    def _clean_repetitive_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Deduplicate consecutive identical assistant replies in context to break repetition loops."""
        cleaned: list[dict[str, str]] = []
        last_assistant_content: Optional[str] = None
        for m in messages:
            if m['role'] == 'assistant':
                content = m['content'].strip().lower()
                if last_assistant_content and content == last_assistant_content:
                    continue
                last_assistant_content = content
            cleaned.append(m)
        return cleaned

    def get_user_turns(self) -> List[str]:
        return [message.content for message in self.history if message.role == 'user']

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """Commit complete turns only; closing a stream rolls back a partial turn."""
        history_start = len(self.history)
        working_start = len(self.memory.working.scratchpad)
        completed = False
        try:
            yield from self._chat_stream_impl(user_input)
            completed = True
            if not self.last_reply_succeeded:
                del self.memory.working.scratchpad[working_start:]
            if self.last_reply_succeeded and len(self.history) > history_start:
                self.memory.episodic.record(EpisodicEntry(
                    entry_id=f'chat-{time.time_ns()}', timestamp=time.time(),
                    user_input=user_input.strip(), assistant_output=self.history[-1].content,
                    action_taken='chat', was_successful=False))
        finally:
            if not completed:
                del self.history[history_start:]
                del self.memory.working.scratchpad[working_start:]
                self.last_reply_succeeded = False
                self.last_answer_confidence = None
                self.last_confidence_source = None
                self.last_token_likelihood = None

    def _chat_stream_impl(self, user_input: str) -> Iterator[str]:
        self.last_reply_succeeded = False
        self.last_answer_confidence = None
        self.last_confidence_source = None
        self.last_token_likelihood = None
        self.local_model.last_finish_reason = None
        prompt = user_input.strip()
        if not prompt:
            return
        self.history.append(ChatMessage(role='user', content=prompt))
        self.memory.working.append_thought(f'User: {prompt}')

        identity = self.creator_profile.answer_identity_question(prompt)
        earlier = [m.content for m in self.history[:-1] if m.role == 'user']
        if needs_c_program_details(prompt, earlier):
            answer = 'What should the C program do? For example, should it sort numbers or search a list?'
            self.last_reply_succeeded = True
        elif identity:
            answer = identity
            self.last_reply_succeeded = True
        elif self.settings.backend == 'local':
            raw_messages = [{'role': message.role, 'content': message.content} for message in self.history]
            messages = self._clean_repetitive_history(raw_messages)
            pieces = []
            if isinstance(self.local_model, LocalModel):
                self.local_model.measure_token_likelihood = self.settings.show_confidence

            recent_assistant_replies = [
                m.content.strip().lower()
                for m in self.history[:-1]
                if m.role == 'assistant' and m.content.strip()
            ][-2:]

            try:
                # Loop Detection Filter: check if generation loops on recent assistant replies
                if isinstance(self.local_model, LocalModel) and recent_assistant_replies:
                    gen_pieces = []
                    for piece in self.local_model.chat_stream(messages):
                        gen_pieces.append(piece)
                    candidate = ''.join(gen_pieces).strip()

                    if candidate and candidate.lower() in recent_assistant_replies:
                        # Reject output and re-sample with higher temperature and purged repetitive turns
                        purged_messages = [
                            m for m in messages
                            if m['role'] != 'assistant' or m['content'].strip().lower() not in recent_assistant_replies
                        ]
                        for piece in self.local_model.chat_stream(
                            purged_messages,
                            temperature=0.85,
                            repeat_penalty=1.35,
                        ):
                            pieces.append(piece)
                            yield piece
                    else:
                        for piece in gen_pieces:
                            pieces.append(piece)
                            yield piece
                else:
                    for piece in self.local_model.chat_stream(messages):
                        pieces.append(piece)
                        yield piece
            except (OSError, RuntimeError, ValueError) as exc:
                # Failed turns must not pollute the next context as assistant answers.
                self.history.pop()
                yield f'Local model error: {exc}'
                return
            answer = ''.join(pieces).strip()
            if not answer:
                self.history.pop()
                yield 'The model produced no text. Please rephrase your question.'
                return
            self.history.append(ChatMessage(role='assistant', content=answer))
            self.last_reply_succeeded = True
            self.last_token_likelihood = getattr(self.local_model, 'last_token_likelihood', None)
            self._assess_answer(prompt, answer, len(earlier))
            self.history[-1].confidence = self.last_answer_confidence
            return
        elif self.settings.backend == 'ollama':
            messages = [{'role': 'system', 'content': self.system_prompt}] + [
                {'role': message.role, 'content': message.content}
                for message in self.history[-24:] if message.role in ('user', 'assistant')
            ]
            answer = self.ollama.chat(messages)
            if not answer:
                answer = ('Could not reach the selected Ollama model. Check `ollama list` and '
                          '`hari settings show`, or switch back with `hari settings backend local`.')
            else:
                self.last_reply_succeeded = True
        elif not getattr(self.engine, 'checkpoint_loaded', False):
            answer = ('HariNano checkpoint not found. Configure one with `hari settings nano-model CHECKPOINT TOKENIZER`. '
                      'The current research checkpoints are not yet capable general chat models.')
        else:
            # Match the prompt format used by the prototype training script.
            from hari_ai.runtime.engine import GenerationConfig
            answer = self.engine.generate('User: ' + prompt + '\nHariAI:',
                                          GenerationConfig(max_new_tokens=96, temperature=0.0)).strip()
            if not answer or not any(ch.isalnum() for ch in answer):
                answer = ('HariNano could not produce a usable answer. Its current checkpoint is an early '
                          'research model and needs substantially more language and instruction training. '
                          'You can optionally configure Ollama with `hari settings backend ollama`.')
            else:
                self.last_reply_succeeded = True

        self.history.append(ChatMessage(role='assistant', content=answer))
        yield answer
