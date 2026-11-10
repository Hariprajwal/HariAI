"""The public autonomous API delegates to the real local-vision controller."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from hari_ai.agent.autonomous import AutonomousAgent, AutonomousOverlay
from hari_ai.agent.controller import DesktopAgentController
from hari_ai.agent.desktop import DesktopDriver


class TestAutonomousAgent(unittest.TestCase):
    def test_unconfigured_agent_does_not_fake_task_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopAgentController(
                config_path=Path(temporary) / 'agent.json',
                driver=Mock(spec=DesktopDriver), overlay_factory=Mock(),
            )
            agent = AutonomousAgent(controller=controller)
            result = agent.run('Organize a file')
        self.assertFalse(result.success)
        self.assertEqual(result.status, 'needs_attention')
        self.assertIn('local vision model', result.error)

    def test_overlay_can_be_constructed_without_claiming_it_is_running(self):
        overlay = AutonomousOverlay()
        self.assertFalse(overlay.is_active)
        overlay.close()


if __name__ == '__main__':
    unittest.main()
