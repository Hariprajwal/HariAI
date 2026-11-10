"""Unit tests for HariAI Autonomous Computer Control Agent."""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from hari_ai.agent.autonomous import (
    AutonomousAgent,
    AutonomousOverlay,
    AutonomousResult,
    AutonomousStep,
    System1FastReflex,
    System2DeliberatePlanner,
    System3WebSearchFallback,
)
from hari_ai.agent.desktop import DesktopCancelled, DesktopDriver, Region


class TestAutonomousAgent(unittest.TestCase):
    def setUp(self):
        self.region = Region(left=0, top=0, width=800, height=600)
        self.mock_driver = MagicMock(spec=DesktopDriver)
        self.mock_driver.execute.return_value = "Action executed successfully"

    def test_system1_fast_reflex(self):
        s1 = System1FastReflex(self.mock_driver)
        res = s1.execute_action("wait", {"milliseconds": 50}, self.region)
        self.mock_driver.execute.assert_called_once()
        self.assertEqual(res, "Action executed successfully")

    def test_system2_deliberate_planner(self):
        s2 = System2DeliberatePlanner()
        thought, action, is_done = s2.think_and_plan("Convert doc.pdf to docx", 0, [])
        self.assertIn("<|thought|>", thought)
        self.assertIn("kind", action)
        self.assertFalse(is_done)

        # Completion step
        thought_done, action_done, is_done = s2.think_and_plan("Convert doc.pdf to docx", 4, [
            MagicMock(spec=AutonomousStep) for _ in range(4)
        ])
        self.assertTrue(is_done)

    def test_system3_web_search_fallback(self):
        mock_search = MagicMock()
        mock_search.search_and_summarize.return_value = "Search result guidance summary"
        s3 = System3WebSearchFallback(mock_search)
        res = s3.search_for_solution("how to convert pptx to docx")
        self.assertEqual(res, "Search result guidance summary")

    def test_autonomous_run_completion(self):
        agent = AutonomousAgent(driver=self.mock_driver)
        result = agent.run("Automate file organization", region=self.region, max_steps=5, show_overlay=False)
        self.assertIsInstance(result, AutonomousResult)
        self.assertTrue(result.success)
        self.assertEqual(result.status, "completed")
        self.assertGreater(result.total_steps, 0)

    def test_autonomous_takeover_interrupt(self):
        agent = AutonomousAgent(driver=self.mock_driver)

        # Simulate user triggering takeover after 1 step
        def trigger_takeover(*args, **kwargs):
            agent.stop()
            raise DesktopCancelled("User hit take over")

        self.mock_driver.execute.side_effect = trigger_takeover
        result = agent.run("Long running task", region=self.region, max_steps=10, show_overlay=False)
        self.assertTrue(result.interrupted_by_user)
        self.assertEqual(result.status, "interrupted_by_user")

    def test_overlay_creation_and_close(self):
        stop_event = threading.Event()
        overlay = AutonomousOverlay(stop_event)
        # Should initialize without throwing exceptions
        self.assertFalse(overlay.is_active)
        overlay.close()


if __name__ == "__main__":
    unittest.main()

