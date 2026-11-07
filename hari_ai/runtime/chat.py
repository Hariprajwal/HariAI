"""Conversational chat backed by an installed local Ollama model."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager
from hari_ai.profile.creator import get_creator_profile
from hari_ai.runtime.engine import GenerationConfig, InferenceEngine
from hari_ai.runtime.ollama import OllamaClient


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class ChatEngine:
    """Manages an active conversational session using the custom HariNano LLM."""

    def __init__(
        self,
        memory_manager: Optional[HierarchicalMemoryManager] = None,
        system_prompt: Optional[str] = None,
        inference_engine: Optional[InferenceEngine] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.memory = memory_manager or HierarchicalMemoryManager()
        self.creator_profile = get_creator_profile()
        self.engine = inference_engine or InferenceEngine(model_path_or_name="nano")
        self.ollama = ollama_client or OllamaClient()
        self.system_prompt = system_prompt or (
            "You are HariAI, a helpful assistant. HariNano is an early from-scratch "
            "model experiment; when Ollama is available, responses come from the "
            "selected local Ollama model. Be accurate and clear about uncertainty. "
            "Do not claim code or reasoning was verified unless a tool verified it."
        )
        self.history: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]

    def clear_history(self) -> None:
        """Resets the conversation history back to the system prompt."""
        self.history = [ChatMessage(role="system", content=self.system_prompt)]
        self.memory.working.clear()

    def get_user_turns(self) -> List[str]:
        """Returns all user prompts in this session."""
        return [m.content for m in self.history if m.role == "user"]

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """Processes user input, updates context, and yields streaming response tokens with confidence."""
        clean_input = user_input.strip()
        if not clean_input:
            return

        # Add user message to session history
        self.history.append(ChatMessage(role="user", content=clean_input))
        self.memory.working.append_thought(f"User: {clean_input}")

        # Formulate intelligent conversational response with calibrated confidence
        response_text, confidence_score = self._synthesize_response(clean_input)

        # Append confidence readout
        if confidence_score < 0:
            confidence_footer = "\n\033[90m[Ollama local model | confidence not calibrated]\033[0m\n"
        else:
            confidence_footer = "\n\033[90m[Offline fallback | confidence not calibrated]\033[0m\n"
        full_assistant_message = response_text + confidence_footer

        # Record assistant reply in history
        self.history.append(
            ChatMessage(
                role="assistant",
                content=full_assistant_message,
                confidence=(confidence_score / 100.0 if confidence_score >= 0 else None),
            )
        )

        # Log into Episodic Memory for Dream Engine sleep consolidation
        self.memory.episodic.record(
            EpisodicEntry(
                entry_id=f"chat-{int(time.time() * 1000)}",
                timestamp=time.time(),
                user_input=clean_input,
                assistant_output=full_assistant_message,
                action_taken="chat",
                was_successful=True,
            )
        )

        # Stream words/tokens for a lively terminal typing effect
        words = re.split(r"(\s+)", response_text)
        for w in words:
            yield w
            time.sleep(0.010)

        # Stream confidence footer
        yield confidence_footer

    def _synthesize_response(self, prompt: str) -> Tuple[str, float]:
        """Respond with Ollama when available, otherwise use offline handlers."""
        lower = prompt.lower()

        # 1. Identity & Creator Queries (Always strictly preserved)
        identity_resp = self.creator_profile.answer_identity_question(prompt)
        if identity_resp:
            return identity_resp, 99.2

        if any(w in lower for w in ["who are you", "what are you", "introduce yourself", "your name"]):
            text = (
                f"Hello! I'm HariAI, an experimental assistant project created by "
                f"{self.creator_profile.name}. Chat uses an installed local Ollama model; "
                "the standalone HariNano model is still a research prototype."
            )
            return text, 98.9

        # 2. Greetings and Casual Conversation
        if lower in ["hi", "hello", "hey", "hola", "greetings", "hi there", "hello!"]:
            return (
                "Hello there! It's great to chat with you. I'm HariAI, your compact intelligent assistant.\n"
                "What would you like to explore, calculate, or code today?",
                97.5,
            )

        if any(w in lower for w in ["how are you", "how are you doing", "how's it going"]):
            return (
                "I'm ready to help. If Ollama is running, I can answer with one of your installed local models.\n"
                "What would you like to work on?",
                96.8,
            )

        if "joke" in lower:
            return (
                "Why do programmers prefer dark mode?\n\n"
                "Because light attracts bugs! 😄\n\n"
                "And here's an AI one for you:\n"
                "Why did the neural network go to sleep?\n"
                "To run its Dream Engine and consolidate its weights without catastrophic forgetting!",
                95.0,
            )

        if any(w in lower for w in ["thank you", "thanks", "thx", "awesome", "great job"]):
            return (
                "You're very welcome! It's a pleasure working with you. Let me know if you need anything else!",
                98.0,
            )

        # 3. Context & Multi-turn Inquiries
        user_turns = self.get_user_turns()
        if "first question" in lower or "first thing i asked" in lower:
            if len(user_turns) > 1:
                return f"Your first question in this session was: \"{user_turns[0]}\".", 98.5
            return f"This is your first question: \"{prompt}\".", 98.5

        if "what did i say earlier" in lower or "previous question" in lower or "last prompt" in lower:
            if len(user_turns) > 1:
                prev = user_turns[-2]
                return f"Earlier you asked: \"{prev}\".", 98.0
            return "This is the start of our conversation.", 98.0

        # Prefer a real installed instruction model for open-ended responses.
        # Only a failed/offline Ollama call falls through to the legacy offline handlers.
        recent_messages = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": message.role, "content": message.content}
                for message in self.history[-24:]
                if message.role in ("user", "assistant")
            ],
        ]
        ollama_response = self.ollama.chat(recent_messages)
        if ollama_response:
            return ollama_response, -1.0

        return (
            "I couldn't reach an installed Ollama chat model. Start Ollama and "
            "check `ollama list`, or set HARIAI_OLLAMA_MODEL to an installed model. "
            "HariNano's research checkpoints are not yet trained for general chat.",
            0.0,
        )
