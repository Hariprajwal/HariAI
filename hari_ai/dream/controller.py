"""Prototype wake/sleep controller.

The built-in cycle inspects memory only. Offline candidate training and its
measured regression gate are separate experiments; this controller does not
train, verify, or promote a checkpoint.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from hari_ai.memory.hierarchy import (
    EpisodicEntry,
    HierarchicalMemoryManager,
    ProceduralRecipe,
    SemanticFact,
)


class SystemState(str, enum.Enum):
    WAKE = "wake"
    SLEEP = "sleep"


class SleepTrigger(str, enum.Enum):
    MANUAL = "manual_command"
    IDLE_TIMEOUT = "idle_timeout"
    SCHEDULED = "scheduled_interval"
    MEMORY_THRESHOLD = "memory_accumulation_threshold"
    CLUSTER_REQUEST = "cluster_wide_request"


@dataclass
class DreamCycleResult:
    """Report generated upon completion of a sleep cycle."""

    cycle_id: str
    trigger: SleepTrigger
    start_time: float
    end_time: float
    experiences_inspected: int
    duplicates_removed: int
    facts_consolidated: int
    recipes_extracted: int
    synthetic_exercises_generated: int
    regression_tests_passed: bool
    candidate_checkpoint_id: Optional[str] = None
    promoted: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class DreamEngineController:
    """Manages the Wake/Sleep lifecycle and continual learning pipeline."""

    def __init__(
        self,
        memory_manager: HierarchicalMemoryManager,
        idle_threshold_seconds: float = 300.0,
        memory_threshold_entries: int = 50,
        regression_evaluator: Optional[Callable[[], bool]] = None,
    ):
        self.memory = memory_manager
        self.idle_threshold_seconds = idle_threshold_seconds
        self.memory_threshold_entries = memory_threshold_entries
        self.regression_evaluator = regression_evaluator

        self.current_state = SystemState.WAKE
        self.last_user_activity = time.time()
        self.sleep_cycle_history: List[DreamCycleResult] = []
        self._preempt_requested = False

    def notify_user_activity(self) -> None:
        """Call whenever user interacts with the system. Resets idle timer and wakes system."""
        self.last_user_activity = time.time()
        if self.current_state == SystemState.SLEEP:
            self.preempt_sleep("User input received")

    def should_enter_sleep(self, current_time: Optional[float] = None) -> Optional[SleepTrigger]:
        """Checks if idle duration or memory thresholds warrant entering sleep."""
        if self.current_state != SystemState.WAKE:
            return None

        now = current_time or time.time()
        unconsolidated = len(self.memory.episodic.get_unconsolidated_experiences())

        if unconsolidated >= self.memory_threshold_entries:
            return SleepTrigger.MEMORY_THRESHOLD

        if (now - self.last_user_activity) >= self.idle_threshold_seconds:
            return SleepTrigger.IDLE_TIMEOUT

        return None

    def preempt_sleep(self, reason: str = "Immediate user intervention") -> None:
        """Requests cooperative cancellation of the prototype sleep cycle."""
        self._preempt_requested = True
        self.current_state = SystemState.WAKE

    def run_sleep_cycle(
        self, trigger: SleepTrigger = SleepTrigger.MANUAL
    ) -> DreamCycleResult:
        """Inspects experiences without creating or promoting model updates."""
        self.current_state = SystemState.SLEEP
        self._preempt_requested = False
        t_start = time.time()
        cycle_id = f"dream-cycle-{int(t_start)}"

        # Step 1: Inspect experiences
        experiences = self.memory.episodic.get_unconsolidated_experiences()
        inspected = len(experiences)
        duplicates_removed = 0
        facts_consolidated = 0
        recipes_extracted = 0
        synthetic_count = 0

        # Step 2: Extract lessons and consolidate memory
        for exp in experiences:
            if self._preempt_requested:
                break

            # Experiences are observations, not automatically verified lessons.
            # No conversation is used as model training data by this controller.

        # Step 3: Run regression benchmark check
        regression_passed = False
        promoted = False
        candidate_checkpoint = None

        if not self._preempt_requested and self.regression_evaluator is not None:
            regression_passed = self.regression_evaluator()

        t_end = time.time()
        self.current_state = SystemState.WAKE

        result = DreamCycleResult(
            cycle_id=cycle_id,
            trigger=trigger,
            start_time=t_start,
            end_time=t_end,
            experiences_inspected=inspected,
            duplicates_removed=duplicates_removed,
            facts_consolidated=facts_consolidated,
            recipes_extracted=recipes_extracted,
            synthetic_exercises_generated=synthetic_count,
            regression_tests_passed=regression_passed,
            candidate_checkpoint_id=candidate_checkpoint,
            promoted=promoted,
            details={"preempted": self._preempt_requested,
                     "status": "inspection_only",
                     "message": "No training, verification, or checkpoint promotion occurred."},
        )

        self.sleep_cycle_history.append(result)
        return result

