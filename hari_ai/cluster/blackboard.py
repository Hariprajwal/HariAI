"""Shared Blackboard Architecture for HariAI Cluster."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from hari_ai.cluster.protocol import ClusterMessage, TaskStatus


class SharedBlackboard:
    """Thread-safe central blackboard state with execution budget and loop-prevention guards."""

    def __init__(
        self,
        max_messages: int = 50,
        max_turns_per_agent: int = 10,
        timeout_seconds: float = 60.0,
    ):
        self.max_messages = max_messages
        self.max_turns_per_agent = max_turns_per_agent
        self.timeout_seconds = timeout_seconds

        self._lock = threading.Lock()
        self.start_time = time.time()
        self.messages: List[ClusterMessage] = []
        self.artifacts: Dict[str, Any] = {}
        self.agent_turn_counts: Dict[str, int] = {}
        self.task_states: Dict[str, TaskStatus] = {}

    def post_message(self, message: ClusterMessage) -> None:
        """Posts a message to the shared message bus with loop and budget validation."""
        with self._lock:
            # 1. Timeout Check
            if (time.time() - self.start_time) > self.timeout_seconds:
                raise TimeoutError(
                    f"Cluster execution exceeded timeout of {self.timeout_seconds}s."
                )

            # 2. Total Message Budget Check
            if len(self.messages) >= self.max_messages:
                raise RuntimeError(
                    f"Cluster message budget exceeded ({self.max_messages} msgs). Aborting to prevent infinite loop."
                )

            # 3. Agent Turn Count Check
            sender = message.sender
            self.agent_turn_counts[sender] = self.agent_turn_counts.get(sender, 0) + 1
            if self.agent_turn_counts[sender] > self.max_turns_per_agent:
                raise RuntimeError(
                    f"Agent '{sender}' exceeded turn budget of {self.max_turns_per_agent} turns."
                )

            # 4. Anti-Loop Ping-Pong Detection (check for circular alternation)
            if len(self.messages) >= 3:
                prev1 = self.messages[-1]
                prev2 = self.messages[-2]
                if (
                    prev1.sender == message.receiver
                    and prev1.receiver == message.sender
                    and prev2.sender == message.sender
                    and prev2.receiver == message.receiver
                    and prev1.objective == message.objective
                ):
                    raise RuntimeError(
                        f"Anti-loop guard triggered: Circular delegation detected between '{message.sender}' and '{message.receiver}'."
                    )

            self.messages.append(message)
            self.task_states[message.task_id] = message.status

    def set_artifact(self, key: str, value: Any) -> None:
        """Stores a shared data artifact (e.g. code snippet, proof trace, benchmark score)."""
        with self._lock:
            self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.artifacts.get(key, default)

    def get_messages_for(self, receiver: str) -> List[ClusterMessage]:
        with self._lock:
            return [m for m in self.messages if m.receiver in (receiver, "all")]

    def get_task_history(self, task_id: str) -> List[ClusterMessage]:
        with self._lock:
            return [m for m in self.messages if m.task_id == task_id]
