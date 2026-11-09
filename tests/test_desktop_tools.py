"""Desktop validation and execution tests using fake devices only."""

import hashlib
import importlib
import json
import threading
import types
import unittest
from unittest.mock import Mock, patch

from hari_ai.agent import desktop
from hari_ai.agent.desktop import (
    Action, Decision, DesktopCancelled, DesktopDriver, DesktopError, Region,
    parse_decision,
)


BOX = [100, 200, 300, 400]


def decision(kind="click", args=None, **fields):
    result = {
        "summary": "Select the input.", "done": False,
        "action": {"kind": kind, "args": {"box": BOX} if args is None else args},
    }
    result.update(fields)
    return json.dumps(result)


class TestDecisionValidation(unittest.TestCase):
    def test_complete_and_action_decisions(self):
        parsed = parse_decision(decision())
        self.assertEqual(parsed.action, Action("click", {"box": BOX}))
        self.assertFalse(parsed.done)
        self.assertEqual(parse_decision(decision(action=None, done=True)),
                         Decision("Select the input.", None, True))

    def test_rejects_non_json_and_ambiguous_objects(self):
        invalid = [
            "click(10,20)", "```json\n" + decision() + "\n```",
            decision() + decision(), "[]", "null", "true", "{}",
            '{"summary":"x","done":true,"done":false,"action":null}',
            '{"summary":"x","done":false,"action":{"kind":"click",'
            '"args":{"box":[0,0,1,1],"box":[0,0,1000,1000]}}}',
            decision(extra="untrusted"), decision(done=1), decision(summary=[]),
            decision(done=True), decision(action=None),
            decision(action=[{"kind": "click", "args": {"box": BOX}}]),
            decision(action={"kind": "click", "args": {"box": BOX}, "python": "x"}),
            decision("exec", {"code": "__import__('os').system('anything')"}),
            decision("click", {"box": BOX, "text": "extra"}),
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_decision(raw)

    def test_rejects_bad_boxes(self):
        for box in ([0, 0, 1001, 20], [-1, 0, 1, 20], [100, 20, 0, 30],
                    [0, 30, 100, 20], [False, 0, 1, 1], [0, 0, 1],
                    [0, 0, "1000", 1000], [0, 0, float("nan"), 1],
                    [0, 0, float("inf"), 1], [0, 0, float("-inf"), 1],
                    {"x": 1}, None):
            with self.subTest(box=box), self.assertRaises(ValueError):
                parse_decision(decision(args={"box": box}))
        with self.assertRaises(ValueError):
            parse_decision(decision().replace("100,", "1e999,"))
        with self.assertRaises(ValueError):
            Action("click", {"box": (0, 0, 1, 1)})

    def test_every_argument_schema_is_closed(self):
        valid = [
            ("click", {"box": BOX}), ("right_click", {"box": BOX}),
            ("double_click", {"box": BOX}), ("hover", {"box": BOX}),
            ("type", {"box": BOX, "text": "hello"}), ("press", {"key": "enter"}),
            ("hotkey", {"keys": ["ctrl", "a"]}),
            ("scroll", {"box": BOX, "amount": -10}), ("wait", {"milliseconds": 1000}),
        ]
        for kind, args in valid:
            with self.subTest(kind=kind):
                self.assertEqual(parse_decision(decision(kind, args)).action, Action(kind, args))
                for key in args:
                    with self.assertRaises(ValueError):
                        Action(kind, {name: value for name, value in args.items() if name != key})
                with self.assertRaises(ValueError):
                    Action(kind, dict(args, extra=True))

    def test_rejects_focusless_type_unbounded_actions_and_unsafe_keys(self):
        invalid = [
            ("type", {"text": "hello"}), ("type", {"box": BOX, "text": "x" * 2001}),
            ("type", {"box": BOX, "text": 123}), ("press", {"key": "win"}),
            ("press", {"key": "f12"}), ("press", {"key": ["enter"]}),
            ("hotkey", {"keys": ["win", "r"]}),
            ("hotkey", {"keys": ["ctrl", "shift", "enter"]}),
            ("hotkey", {"keys": ["ctrl", "alt", "delete"]}),
            ("hotkey", {"keys": ["ctrl", "shift", "j"]}),
            ("hotkey", {"keys": ["ctrl", False]}), ("hotkey", {"keys": "ctrl+a"}),
            ("scroll", {"box": BOX, "amount": 11}),
            ("scroll", {"box": BOX, "amount": -11}),
            ("scroll", {"box": BOX, "amount": True}),
            ("scroll", {"box": BOX, "amount": 1.0}),
            ("wait", {"milliseconds": -1}), ("wait", {"milliseconds": 1001}),
            ("wait", {"milliseconds": True}), ("wait", {"milliseconds": 0.5}),
        ]
        for kind, args in invalid:
            with self.subTest(kind=kind, args=args), self.assertRaises(ValueError):
                parse_decision(decision(kind, args))

    def test_region_limits_and_negative_monitor_offsets(self):
        self.assertEqual(Region(-100, -100, 4000, 4000).width, 4000)
        for values in ((0, 0, 9, 10), (0, 0, 10, 0), (0, 0, 4001, 4000),
                       (False, 0, 10, 10), (0, 0, 10.0, 10)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Region(*values)

    def test_action_does_not_alias_input_lists(self):
        box = BOX.copy()
        action = Action("click", {"box": box})
        box[0] = 10000
        self.assertEqual(action.args["box"], BOX)


class FakeFailSafe(Exception):
    pass


class FakeScreen:
    def __init__(self):
        self.monitors = [
            {"left": -200, "top": 0, "width": 1200, "height": 800},
            {"left": 0, "top": 0, "width": 1000, "height": 800},
            {"left": -200, "top": 0, "width": 100, "height": 800},
        ]
        self.grab = Mock(side_effect=lambda region: types.SimpleNamespace(
            width=region["width"], height=region["height"], rgb=b"fake RGB",
        ))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeClipboard:
    def __init__(self):
        self.value = "original clipboard"
        self.copy = Mock(side_effect=self._copy)
        self.paste = Mock(side_effect=lambda: self.value)

    def _copy(self, value):
        self.value = value


class TestDesktopDriver(unittest.TestCase):
    def setUp(self):
        self.screen = FakeScreen()
        self.gui = types.SimpleNamespace(
            FAILSAFE=False, FailSafeException=FakeFailSafe,
            click=Mock(), moveTo=Mock(), press=Mock(), hotkey=Mock(), scroll=Mock(),
        )
        self.clipboard = FakeClipboard()
        self.image = Mock()
        self.image.frombytes.return_value.save.side_effect = (
            lambda buffer, format: buffer.write(b"\x89PNG fake image")
        )
        self.modules = {
            "mss": types.SimpleNamespace(mss=lambda: self.screen),
            "PIL.Image": self.image, "pyautogui": self.gui, "pyperclip": self.clipboard,
        }
        self.loader = patch.object(desktop.importlib, "import_module", side_effect=self.modules.__getitem__)
        self.loader_mock = self.loader.start()
        self.addCleanup(self.loader.stop)
        self.sleep = patch.object(desktop.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.driver = DesktopDriver()
        self.region = Region(10, 20, 200, 100)

    def test_capture_contains_only_selected_rectangle_and_digest(self):
        with patch.object(desktop.time, "monotonic", return_value=123.25):
            observation = self.driver.capture(self.region)
        self.screen.grab.assert_called_once_with({"left": 10, "top": 20, "width": 200, "height": 100})
        self.assertEqual(observation.region, self.region)
        self.assertEqual(observation.captured_at, 123.25)
        self.assertEqual(observation.sha256, hashlib.sha256(observation.png).hexdigest())
        self.assertEqual(observation.png, b"\x89PNG fake image")
        self.assertNotIn("pyautogui", [call.args[0] for call in self.loader_mock.call_args_list])

    def test_capture_rejects_monitor_gaps_and_spanning_rectangles(self):
        for region in (Region(-150, 0, 200, 20), Region(-99, 0, 50, 20),
                       Region(990, 0, 20, 20), Region(0, 790, 20, 20)):
            with self.subTest(region=region), self.assertRaises(DesktopError):
                self.driver.capture(region)
        self.screen.grab.assert_not_called()
        self.driver.capture(Region(-200, 0, 100, 800))
        self.assertEqual(self.screen.grab.call_count, 1)

    def test_capture_rejects_unexpected_image_dimensions(self):
        self.screen.grab.side_effect = None
        self.screen.grab.return_value = types.SimpleNamespace(width=1, height=1, rgb=b"x")
        with self.assertRaises(DesktopError):
            self.driver.capture(self.region)
        self.image.frombytes.assert_not_called()

    def test_click_variants_and_inclusive_coordinate_boundaries(self):
        for kind, button, clicks in (("click", "left", 1), ("right_click", "right", 1),
                                     ("double_click", "left", 2)):
            with self.subTest(kind=kind):
                self.driver.execute(Action(kind, {"box": [1000, 1000, 1000, 1000]}), self.region)
                self.gui.click.assert_called_with(209, 119, button=button, clicks=clicks,
                                                 interval=0.1 if clicks == 2 else 0.0)
        self.driver.execute(Action("click", {"box": [0, 0, 0, 0]}), self.region)
        self.assertEqual(self.gui.click.call_args.args, (10, 20))
        self.driver.execute(Action("click", {"box": [0, 0, 1000, 1000]}), self.region)
        self.assertEqual(self.gui.click.call_args.args, (110, 70))
        self.assertTrue(self.gui.FAILSAFE)

    def test_many_box_centers_remain_in_selected_region(self):
        for x in (0, 1, 250.5, 500, 999.999, 1000):
            for y in (0, 1, 500, 999, 1000):
                self.driver.execute(Action("hover", {"box": [x, y, x, y]}), self.region)
                px, py = self.gui.moveTo.call_args.args
                self.assertTrue(10 <= px <= 209 and 20 <= py <= 119)

    def test_scroll_moves_to_approved_target_before_scroll(self):
        calls = Mock()
        calls.attach_mock(self.gui.moveTo, "moveTo")
        calls.attach_mock(self.gui.scroll, "scroll")
        self.driver.execute(Action("scroll", {"box": BOX, "amount": -3}), self.region)
        self.assertEqual([call[0] for call in calls.mock_calls], ["moveTo", "scroll"])
        self.gui.scroll.assert_called_once_with(-3)

    def test_type_focuses_target_and_restores_clipboard(self):
        calls = Mock()
        calls.attach_mock(self.gui.click, "click")
        calls.attach_mock(self.clipboard.copy, "copy")
        calls.attach_mock(self.gui.hotkey, "hotkey")
        self.driver.execute(Action("type", {"box": BOX, "text": "Hello \u0c95\u0ca8\u0ccd\u0ca8\u0ca1"}), self.region)
        self.assertEqual([call[0] for call in calls.mock_calls], ["click", "copy", "hotkey", "copy"])
        self.gui.click.assert_called_once_with(50, 50, button="left")
        self.gui.hotkey.assert_called_once_with("ctrl", "v")
        self.assertEqual(self.clipboard.value, "original clipboard")

    def test_type_restores_clipboard_when_paste_fails(self):
        self.gui.hotkey.side_effect = FakeFailSafe("corner")
        with self.assertRaisesRegex(DesktopError, "fail-safe"):
            self.driver.execute(Action("type", {"box": BOX, "text": "new text"}), self.region)
        self.assertEqual(self.clipboard.value, "original clipboard")
        self.assertEqual(self.gui.hotkey.call_count, 1)

    def test_typing_stops_before_clipboard_when_focus_fails(self):
        self.gui.click.side_effect = RuntimeError("focus failed")
        with self.assertRaises(DesktopError):
            self.driver.execute(Action("type", {"box": BOX, "text": "new text"}), self.region)
        self.gui.hotkey.assert_not_called()
        self.clipboard.copy.assert_not_called()

    def test_clipboard_write_failure_restores_without_paste(self):
        def copy(value):
            self.clipboard.value = value
            if value == "new text":
                raise RuntimeError("backend write error")
        self.clipboard.copy.side_effect = copy
        with self.assertRaises(DesktopError):
            self.driver.execute(Action("type", {"box": BOX, "text": "new text"}), self.region)
        self.assertEqual(self.clipboard.value, "original clipboard")
        self.gui.hotkey.assert_not_called()

    def test_cancellation_before_effect_loads_no_dependencies(self):
        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(DesktopCancelled):
            self.driver.execute(Action("click", {"box": BOX}), self.region, stopped)
        self.loader_mock.assert_not_called()
        self.gui.click.assert_not_called()

    def test_cancellation_between_focus_and_type_stops_paste(self):
        stopped = threading.Event()
        self.gui.click.side_effect = lambda *args, **kwargs: stopped.set()
        with self.assertRaises(DesktopCancelled):
            self.driver.execute(Action("type", {"box": BOX, "text": "new text"}), self.region, stopped)
        self.gui.hotkey.assert_not_called()
        self.clipboard.copy.assert_not_called()

    def test_cancellation_after_clipboard_write_restores_clipboard(self):
        stopped = threading.Event()
        def copy(value):
            self.clipboard.value = value
            if value == "new text":
                stopped.set()
        self.clipboard.copy.side_effect = copy
        with self.assertRaises(DesktopCancelled):
            self.driver.execute(Action("type", {"box": BOX, "text": "new text"}), self.region, stopped)
        self.gui.hotkey.assert_not_called()
        self.assertEqual(self.clipboard.value, "original clipboard")

    def test_failsafe_aborts_scroll_without_second_effect(self):
        self.gui.moveTo.side_effect = FakeFailSafe("corner")
        with self.assertRaisesRegex(DesktopError, "fail-safe"):
            self.driver.execute(Action("scroll", {"box": BOX, "amount": 1}), self.region)
        self.assertTrue(self.gui.FAILSAFE)
        self.gui.scroll.assert_not_called()

    def test_execution_revalidates_mutated_action(self):
        action = Action("click", {"box": BOX})
        action.args["box"][0] = -20
        with self.assertRaises(ValueError):
            self.driver.execute(action, self.region)
        self.loader_mock.assert_not_called()

    def test_execution_checks_monitor_changes_without_capture(self):
        self.screen.monitors = self.screen.monitors[:1]
        with self.assertRaises(DesktopError):
            self.driver.execute(Action("click", {"box": BOX}), self.region)
        self.gui.click.assert_not_called()
        self.screen.grab.assert_not_called()

    def test_wait_is_interruptible_and_never_loads_desktop(self):
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True
        with self.assertRaises(DesktopCancelled):
            self.driver.execute(Action("wait", {"milliseconds": 500}), self.region, stop)
        stop.wait.assert_called_once_with(0.5)
        self.loader_mock.assert_not_called()


class TestDesktopImports(unittest.TestCase):
    def test_module_import_does_not_load_desktop_dependencies(self):
        # Executing a fresh module avoids invalidating the classes imported above.
        spec = importlib.util.find_spec("hari_ai.agent.desktop")
        fresh = types.ModuleType("_hari_desktop_import_test")
        import sys
        with patch.dict(sys.modules, {fresh.__name__: fresh}), \
                patch.object(importlib, "import_module", side_effect=AssertionError("eager dependency")):
            exec(compile(spec.loader.get_source(spec.name), spec.origin, "exec"), fresh.__dict__)


if __name__ == "__main__":
    unittest.main()
