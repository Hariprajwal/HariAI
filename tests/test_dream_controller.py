"""Unit tests for the HariAI Dream Engine Controller: Consolidation, Generative Replay & Auto-Dream."""

import unittest
from hari_ai.dream.controller import (
    DreamEngineController,
    SleepTrigger,
    SystemState,
)
from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager


class TestDreamEngineController(unittest.TestCase):
    def setUp(self):
        self.mem = HierarchicalMemoryManager()
        self.controller = DreamEngineController(
            memory_manager=self.mem,
            idle_threshold_seconds=10.0,
            memory_threshold_entries=3,
        )

    def tearDown(self):
        self.controller.stop_auto_dreamer()

    def test_initial_state(self):
        self.assertEqual(self.controller.current_state, SystemState.WAKE)

    def test_sleep_triggers_on_memory_threshold(self):
        # Insert 3 experiences
        for i in range(3):
            self.mem.episodic.record(
                EpisodicEntry(
                    entry_id=f"e-{i}",
                    timestamp=100.0 + i,
                    user_input=f"test query {i}",
                    assistant_output=f"test answer {i}",
                )
            )
        trigger = self.controller.should_enter_sleep()
        self.assertEqual(trigger, SleepTrigger.MEMORY_THRESHOLD)

    def test_sleep_triggers_on_idle_timeout(self):
        # Simulated time 20 seconds later
        now = self.controller.last_user_activity + 20.0
        trigger = self.controller.should_enter_sleep(current_time=now)
        self.assertEqual(trigger, SleepTrigger.IDLE_TIMEOUT)

    def test_run_sleep_cycle_generative_replay_and_consolidation(self):
        # Add experience with a failure
        self.mem.episodic.record(
            EpisodicEntry(
                entry_id="fail-1",
                timestamp=100.0,
                user_input="Write C memory management",
                assistant_output="char *p = malloc(10);",
                was_successful=False,
                user_feedback="Forgot to free pointer",
            )
        )
        self.mem.episodic.record(
            EpisodicEntry(
                entry_id="succ-1",
                timestamp=101.0,
                user_input="Write python loop",
                assistant_output="for i in range(10): print(i)",
                action_taken="python_codegen",
                was_successful=True,
            )
        )

        result = self.controller.run_sleep_cycle(trigger=SleepTrigger.MANUAL)
        self.assertEqual(result.experiences_inspected, 2)
        # Synthetic exercises generated via Generative Replay
        self.assertGreater(result.synthetic_exercises_generated, 0)
        self.assertGreaterEqual(result.facts_consolidated, 1)
        self.assertGreaterEqual(result.recipes_extracted, 1)
        # Default invariant regression suite passed
        self.assertTrue(result.regression_tests_passed)
        self.assertTrue(result.promoted)
        self.assertIsNotNone(result.candidate_checkpoint_id)
        self.assertEqual(self.controller.current_state, SystemState.WAKE)

        # Verify second cycle skips already consolidated entries
        result2 = self.controller.run_sleep_cycle(trigger=SleepTrigger.MANUAL)
        self.assertEqual(result2.duplicates_removed, 2)
        self.assertEqual(result2.synthetic_exercises_generated, 0)

    def test_user_preemption_wakes_sleep(self):
        self.controller.current_state = SystemState.SLEEP
        self.controller.notify_user_activity()
        self.assertEqual(self.controller.current_state, SystemState.WAKE)

    def test_auto_dreamer_lifecycle(self):
        self.assertFalse(self.controller.auto_dream_active)
        self.controller.start_auto_dreamer(check_interval=0.1)
        self.assertTrue(self.controller.auto_dream_active)
        self.controller.stop_auto_dreamer()
        self.assertFalse(self.controller.auto_dream_active)


if __name__ == "__main__":
    unittest.main()
