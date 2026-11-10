"""Validated desktop primitives; optional desktop dependencies load on use only.

Coordinates refer to the selected screenshot, on a normalized 0..1000 scale.
This module deliberately accepts data, never generated Python or shell code.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import time
from dataclasses import dataclass
from typing import Any


MAX_REGION_PIXELS = 16_000_000
MAX_TEXT_LENGTH = 2_000
PRESS_KEYS = frozenset({
    "enter", "tab", "escape", "backspace", "left", "right", "up", "down",
    "home", "end", "pageup", "pagedown",
})
HOTKEYS = frozenset({
    ("ctrl", key) for key in ("a", "c", "v", "x", "z", "y", "f", "l", "r", "t")
} | {
    ("ctrl", "shift", "z"), ("ctrl", "tab"), ("ctrl", "shift", "tab"),
    ("shift", "tab"), ("alt", "left"), ("alt", "right"),
})
_ACTION_FIELDS = {
    "click": {"box"}, "right_click": {"box"}, "double_click": {"box"},
    "hover": {"box"}, "type": {"box", "text"}, "press": {"box", "key"},
    "hotkey": {"box", "keys"}, "scroll": {"box", "amount"}, "wait": {"milliseconds"},
}


class DesktopError(RuntimeError):
    """A desktop operation failed; callers must stop the current agent run."""


class DesktopCancelled(DesktopError):
    """The user stopped the desktop operation before its next effect."""


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (
            self.left, self.top, self.width, self.height,
        )):
            raise ValueError("Region coordinates and dimensions must be integers.")
        if self.width < 10 or self.height < 10:
            raise ValueError("Select a region at least 10 pixels wide and high.")
        if self.width * self.height > MAX_REGION_PIXELS:
            raise ValueError("The selected region exceeds 16 million pixels.")


@dataclass(frozen=True)
class Observation:
    png: bytes
    sha256: str
    region: Region
    captured_at: float


def _validate_box(box: Any) -> None:
    if type(box) is not list or len(box) != 4:
        raise ValueError("box must be [xmin, ymin, xmax, ymax].")
    if any(type(value) not in (int, float) or not 0 <= value <= 1000 for value in box):
        raise ValueError("box coordinates must be finite numbers from 0 to 1000.")
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("box minimum coordinates must not exceed its maximums.")


def _bounded_integer(value: Any, minimum: int, maximum: int, name: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")


@dataclass(frozen=True)
class Action:
    kind: str
    args: dict

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _ACTION_FIELDS:
            raise ValueError("Unknown desktop action.")
        if type(self.args) is not dict or set(self.args) != _ACTION_FIELDS[self.kind]:
            raise ValueError(f"Invalid argument fields for {self.kind}.")
        args = self.args
        if "box" in args:
            _validate_box(args["box"])
        if self.kind == "type":
            if type(args["text"]) is not str or len(args["text"]) > MAX_TEXT_LENGTH:
                raise ValueError("type text must be a string of at most 2000 characters.")
        elif self.kind == "press":
            if type(args["key"]) is not str or args["key"] not in PRESS_KEYS:
                raise ValueError("Unsupported press key.")
        elif self.kind == "hotkey":
            keys = args["keys"]
            if (type(keys) is not list or any(type(key) is not str for key in keys)
                    or tuple(keys) not in HOTKEYS):
                raise ValueError("Unsupported hotkey combination.")
        elif self.kind == "scroll":
            _bounded_integer(args["amount"], -10, 10, "scroll amount")
        elif self.kind == "wait":
            _bounded_integer(args["milliseconds"], 0, 1000, "wait milliseconds")
        # Detach lists from the provider's original response. execute() also takes
        # a validated snapshot, since a frozen dataclass cannot freeze a dict.
        object.__setattr__(self, "args", {
            key: value.copy() if type(value) is list else value
            for key, value in args.items()
        })


@dataclass(frozen=True)
class Decision:
    summary: str
    action: Action | None
    done: bool

    def __post_init__(self) -> None:
        if type(self.summary) is not str or type(self.done) is not bool:
            raise ValueError("Decision summary must be a string and done must be boolean.")
        if self.done and self.action is not None:
            raise ValueError("A completed decision cannot include an action.")
        if not self.done and not isinstance(self.action, Action):
            raise ValueError("An incomplete decision must include exactly one action.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}.")


def parse_decision(raw: str) -> Decision:
    """Parse one complete, strict JSON decision, without guessing or repairs."""
    if type(raw) is not str or len(raw) > 32_768:
        raise ValueError("Decision must be a JSON string of at most 32768 characters.")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Decision must be one complete JSON object.") from exc
    if type(data) is not dict or set(data) != {"summary", "done", "action"}:
        raise ValueError("Decision requires exactly summary, done, and action.")
    action_data = data["action"]
    action = None
    if action_data is not None:
        if type(action_data) is not dict or set(action_data) != {"kind", "args"}:
            raise ValueError("Action requires exactly kind and args.")
        action = Action(action_data["kind"], action_data["args"])
    return Decision(summary=data["summary"], action=action, done=data["done"])


def _load_dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DesktopError(
            "Desktop dependencies are unavailable. Install HariAI with: pip install '.[desktop]'"
        ) from exc


def _check_stopped(stop_event: Any) -> None:
    if stop_event is not None and stop_event.is_set():
        raise DesktopCancelled("Desktop action stopped by the user.")


def _validate_region_on_monitor(region: Region, monitors: list[dict]) -> None:
    # monitors[0] is mss's aggregate desktop. It can contain gaps between
    # displays, so only real individual monitors can authorize a rectangle.
    if not any(
        monitor["left"] <= region.left
        and monitor["top"] <= region.top
        and region.left + region.width <= monitor["left"] + monitor["width"]
        and region.top + region.height <= monitor["top"] + monitor["height"]
        for monitor in monitors[1:]
    ):
        raise DesktopError("The selected region must fit inside one active monitor.")


def _box_center(box: list[int | float], region: Region) -> tuple[int, int]:
    # floor(x + .5) gives deterministic nearest-pixel rounding. The inclusive
    # normalized endpoint 1000 must map to width/height - 1, never outside.
    x = math.floor((box[0] + box[2]) * (region.width - 1) / 2000 + 0.5)
    y = math.floor((box[1] + box[3]) * (region.height - 1) / 2000 + 0.5)
    return region.left + x, region.top + y


class DesktopDriver:
    """Capture selected pixels and execute one validated action at a time.

    Exceptions are terminal for the current run. There are no retries or
    fallback actions. The controller owns the run lifecycle and user approval.
    """

    def select_region(self) -> Region | None:
        """Explicit drag selection on the primary monitor; never auto-fullscreen."""
        import tkinter as tk
        chosen = []
        start = []
        window = tk.Tk()
        window.attributes('-fullscreen', True)
        window.attributes('-alpha', 0.35)
        window.attributes('-topmost', True)
        canvas = tk.Canvas(window, background='#180c27', cursor='crosshair', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        canvas.create_text(30, 30, anchor='nw', fill='white',
                           text='HariAI: drag over the target application. Esc cancels.', font=('Segoe UI', 16))
        rectangle = canvas.create_rectangle(0, 0, 0, 0, outline='#c17aff', width=3)
        def press(event):
            start[:] = [event.x, event.y]
        def drag(event):
            if start:
                canvas.coords(rectangle, *start, event.x, event.y)
        def release(event):
            if start:
                left, top = min(start[0], event.x), min(start[1], event.y)
                width, height = abs(event.x - start[0]), abs(event.y - start[1])
                if width >= 10 and height >= 10:
                    chosen.append(Region(left, top, width, height))
                    window.destroy()
        canvas.bind('<ButtonPress-1>', press)
        canvas.bind('<B1-Motion>', drag)
        canvas.bind('<ButtonRelease-1>', release)
        window.bind('<Escape>', lambda _: window.destroy())
        window.mainloop()
        return chosen[0] if chosen else None

    def primary_screen_region(self) -> Region:
        """Returns the primary monitor region."""
        try:
            mss = _load_dependency("mss")
            with mss.mss() as screen:
                m = screen.monitors[1] if len(screen.monitors) > 1 else screen.monitors[0]
                return Region(int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"]))
        except Exception:
            return Region(0, 0, 1920, 1080)

    def capture(self, region: Region) -> Observation:
        region = Region(region.left, region.top, region.width, region.height)
        try:
            mss = _load_dependency("mss")
            image = _load_dependency("PIL.Image")
            with mss.mss() as screen:
                _validate_region_on_monitor(region, screen.monitors)
                shot = screen.grab({
                    "left": region.left, "top": region.top,
                    "width": region.width, "height": region.height,
                })
                captured_at = time.monotonic()
                if (shot.width, shot.height) != (region.width, region.height):
                    raise DesktopError("The screenshot dimensions changed; select the region again.")
                pixels = image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
                buffer = io.BytesIO()
                pixels.save(buffer, format="PNG")
                png = buffer.getvalue()
            return Observation(png, hashlib.sha256(png).hexdigest(), region, captured_at)
        except DesktopError:
            raise
        except Exception as exc:
            raise DesktopError(f"Desktop capture failed: {exc}") from exc

    def execute(self, action: Action, region: Region, stop_event: Any = None) -> str:
        """Execute one action, checking cancellation before every input effect."""
        _check_stopped(stop_event)
        action = Action(action.kind, action.args)
        region = Region(region.left, region.top, region.width, region.height)
        if action.kind == "wait":
            seconds = action.args["milliseconds"] / 1000
            if stop_event is None:
                time.sleep(seconds)
            elif stop_event.wait(seconds):
                raise DesktopCancelled("Desktop wait stopped by the user.")
            _check_stopped(stop_event)
            return "Wait completed."

        gui = None
        try:
            # Check the live display arrangement again before moving the pointer.
            mss = _load_dependency("mss")
            with mss.mss() as screen:
                _validate_region_on_monitor(region, screen.monitors)
            _check_stopped(stop_event)
            gui = _load_dependency("pyautogui")
            gui.FAILSAFE = True
            args = action.args
            target = _box_center(args["box"], region) if "box" in args else None

            def effect(function: Any, *values: Any, **kwargs: Any) -> None:
                _check_stopped(stop_event)
                function(*values, **kwargs)

            if action.kind in {"click", "right_click", "double_click"}:
                effect(gui.click, *target,
                       button="right" if action.kind == "right_click" else "left",
                       clicks=2 if action.kind == "double_click" else 1,
                       interval=0.1 if action.kind == "double_click" else 0.0)
            elif action.kind == "hover":
                effect(gui.moveTo, *target)
            elif action.kind == "scroll":
                effect(gui.moveTo, *target)
                effect(gui.scroll, args["amount"])
            elif action.kind == "press":
                effect(gui.click, *target, button="left")
                effect(gui.press, args["key"])
            elif action.kind == "hotkey":
                effect(gui.click, *target, button="left")
                effect(gui.hotkey, *args["keys"])
            elif action.kind == "type":
                clipboard = _load_dependency("pyperclip")
                previous = None
                try:
                    previous = clipboard.paste()
                except Exception:
                    pass  # Some clipboard backends support writes but not reads.
                effect(gui.click, *target, button="left")
                clipboard_changed = False
                try:
                    _check_stopped(stop_event)
                    # Mark before copy: a backend may update the clipboard and
                    # then fail, in which case restoration is still necessary.
                    clipboard_changed = True
                    clipboard.copy(args["text"])
                    effect(gui.hotkey, "ctrl", "v")
                    # Allow the destination to read the clipboard before restoring
                    # it, even when pyautogui's global pause was set to zero.
                    if stop_event is None:
                        time.sleep(0.1)
                    else:
                        stop_event.wait(0.1)
                finally:
                    if clipboard_changed and type(previous) is str:
                        try:
                            clipboard.copy(previous)
                        except Exception:
                            pass  # Restoration is best effort, including on failure.
            _check_stopped(stop_event)
            return f"{action.kind} completed."
        except DesktopError:
            raise
        except Exception as exc:
            fail_safe = getattr(gui, "FailSafeException", None)
            if isinstance(fail_safe, type) and isinstance(exc, fail_safe):
                raise DesktopError("Desktop fail-safe activated; the agent run must stop.") from exc
            raise DesktopError(f"Desktop action failed: {exc}") from exc
