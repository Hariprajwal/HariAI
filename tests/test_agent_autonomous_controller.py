"""Autonomous desktop loop tests use inert model and driver fixtures only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from hari_ai.agent.controller import DesktopAgentController
from hari_ai.agent.desktop import Observation, Region


class TestAutonomousDesktopController(unittest.TestCase):
    def test_auto_observes_executes_then_reports_model_completion_without_claiming_verification(self):
        region = Region(0, 0, 100, 100)
        observation = Observation(b'png', 'digest', region, 0.0)
        driver = Mock()
        driver.capture.return_value = observation
        driver.execute.return_value = 'Wait completed.'
        vision = Mock()
        vision.decide.side_effect = [
            json.dumps({'summary': 'Wait for the visible control to load.', 'done': False,
                        'action': {'kind': 'wait', 'args': {'milliseconds': 0}}}),
            json.dumps({'summary': 'The requested screen is visible.', 'done': True, 'action': None}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopAgentController(
                config_path=Path(temporary) / 'agent.json', driver=driver,
                vision_factory=lambda **_: vision, overlay_factory=Mock(),
            )
            controller.model = 'fixture-vision'
            controller.set_region(region)

            result = controller.command('auto inspect the test screen')

        vision.capabilities.assert_called_once()
        self.assertEqual(vision.decide.call_count, 2)
        self.assertEqual(driver.capture.call_count, 2)
        self.assertTrue(all(call.args[0] == region for call in driver.capture.call_args_list))
        driver.execute.assert_called_once()
        self.assertIn('Review the target application', result)
        self.assertEqual(controller.state, 'Model reports complete')
        self.assertEqual(controller.steps, 1)

    def test_auto_requires_explicit_model_and_region_before_capture(self):
        driver = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopAgentController(config_path=Path(temporary) / 'agent.json',
                                                driver=driver, overlay_factory=Mock())
            with self.assertRaisesRegex(ValueError, 'vision model'):
                controller.command('auto inspect this')
        driver.capture.assert_not_called()

    def test_host_can_be_configured_before_the_model(self):
        vision_factory = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopAgentController(config_path=Path(temporary) / 'agent.json',
                                                driver=Mock(), vision_factory=vision_factory,
                                                overlay_factory=Mock())
            result = controller.command('host http://127.0.0.1:11435')
            self.assertIn('http://127.0.0.1:11435', result)
            self.assertEqual(controller.model, '')
            self.assertEqual(controller.host, 'http://127.0.0.1:11435')
            self.assertIn('Select a vision model', controller.status_text())


if __name__ == '__main__':
    unittest.main()
