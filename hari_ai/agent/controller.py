"""Observe, propose, approve, act, and observe again for a selected screen region."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shlex
import threading
import time
from typing import Callable

from hari_ai.agent.desktop import DesktopCancelled, DesktopDriver, Region, parse_decision
from hari_ai.agent.ocr import LocalScreenOCR, OCRUnavailable, planner_context
from hari_ai.agent.overlay import TakeoverOverlay
from hari_ai.agent.vision import OllamaVisionClient
from hari_ai.runtime.settings import settings_path


@dataclass(frozen=True)
class AgentSnapshot:
    state: str
    model: str
    goal: str
    steps: int
    max_steps: int
    region: str
    pending_id: str
    last_action: str


class DesktopAgentController:
    """Task state is session-only. Screenshots never enter chat memory or logs."""

    def __init__(self, *, config_path: Path | None = None, driver=None, vision_factory=None,
                 max_steps: int = 8, max_observation_age: float = 90, overlay_factory=None):
        self.config_path = config_path or settings_path().with_name('desktop-agent.json')
        self.driver = driver or DesktopDriver()
        self.vision_factory = vision_factory or OllamaVisionClient
        self.overlay_factory = overlay_factory or TakeoverOverlay
        self.max_steps = max_steps
        self.max_observation_age = max_observation_age
        self.model = ''
        self.host = 'http://127.0.0.1:11434'
        self.ocr_enabled = False
        self.ocr = LocalScreenOCR()
        self.region: Region | None = None
        self.state = 'Not configured'
        self.goal = ''
        self.steps = 0
        self.pending_id = ''
        self.last_action = ''
        self.pending = None
        self.observation = None
        self.history: list[dict] = []
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.overlay = None
        self._lock = threading.Lock()
        self.config_error = ''
        try:
            if self.config_path.is_file():
                config = json.loads(self.config_path.read_text(encoding='utf-8'))
                if not isinstance(config, dict):
                    raise ValueError('configuration must be an object')
                self.model = config.get('model', '')
                self.host = config.get('host', self.host)
                self.ocr_enabled = bool(config.get('ocr_enabled', False))
                self.vision_factory(host=self.host, model=self.model)
                self.state = 'Select a region' if self.model else 'Select a vision model'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.model = ''
            self.config_error = f'Agent configuration could not be loaded: {exc}'

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(self.state, self.model or 'Not selected', self.goal, self.steps,
                             self.max_steps, self.region_label(), self.pending_id, self.last_action)

    def region_label(self) -> str:
        r = self.region
        return f'{r.left}, {r.top} · {r.width}×{r.height}' if r else 'Not selected'

    def _discard_plan(self) -> None:
        self.pending = self.observation = None
        self.pending_id = ''

    def _save_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({
            'model': self.model, 'host': self.host, 'ocr_enabled': self.ocr_enabled,
        }, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, self.config_path)

    def stop(self) -> None:
        self.stop_event.set()
        if self._lock.acquire(blocking=False):
            try:
                self._discard_plan()
                self.state = 'Stopped'
            finally:
                self._lock.release()
        else:
            self.state = 'Stopping'

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise RuntimeError('Agent task stopped.')

    def configure(self, model: str, host: str | None = None) -> str:
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError('Specify the name of an installed local vision model.')
        selected_host = host or self.host
        self.vision_factory(host=selected_host, model=model)
        self.model, self.host = model, selected_host
        self._save_config()
        self._discard_plan()
        self.state = 'Ready' if self.region else 'Select a region'
        return f'Desktop vision model: {model}\nLocal service: {selected_host}\nOrdinary chat remains on its selected backend.'

    def configure_host(self, host: str) -> str:
        """Store a loopback service endpoint before or after choosing a model."""
        self.vision_factory(host=host, model=self.model)
        self.host = host
        self._save_config()
        self._discard_plan()
        self.state = 'Ready' if self.model and self.region else ('Select a region' if self.model else 'Select a vision model')
        return f'Desktop vision service: {host}\nChoose a vision model with /agent model NAME.'

    def configure_ocr(self, enabled: bool) -> str:
        self.ocr_enabled = enabled
        self._save_config()
        return ('Local OCR enabled. It uses only pre-provisioned local weights and never downloads them automatically.'
                if enabled else 'Local OCR disabled. Desktop tasks will use the selected vision model only.')

    def set_region(self, region: Region) -> str:
        self.region = region
        self._discard_plan()
        self.state = 'Ready' if self.model else 'Select a vision model'
        return f'Desktop scope: {self.region_label()}\nOnly this region will be captured when you run an agent task.'

    def status_text(self) -> str:
        s = self.snapshot()
        return (f'Desktop agent: {s.state}\nVision model: {s.model}\nRegion: {s.region}\n'
                f'Actions executed: {s.steps}/{s.max_steps}\n'
                f'Pending approval: {s.pending_id or "none"}\n'
                f'Local OCR: {"enabled" if self.ocr_enabled else "off"} (optional; no automatic downloads)\n'
                'Screenshots stay in this session and go only to the configured local vision service.\n'
                'No background watching or automatic message replies are running.'
                + (f'\n{self.config_error}' if self.config_error else ''))

    def _plan(self, emit: Callable[[str], None]) -> str:
        self._check_stop()
        if self.steps >= self.max_steps:
            self.state = 'Step limit reached'
            self._discard_plan()
            return f'Stopped after {self.steps} actions. Review the result before starting another task.'
        self.state = 'Observing'
        emit('Observing the selected region…\n')
        client = self.vision_factory(host=self.host, model=self.model)
        # Verify capability before capturing any pixels.
        client.capabilities()
        self._check_stop()
        observation = self.driver.capture(self.region)
        self.state = 'Planning'
        emit('Planning the next desktop action…\n')
        raw = client.decide(self.goal, observation.png, self.history, stop_event=self.stop_event,
                            ocr_context=self._ocr_context(observation.png, emit))
        self._check_stop()
        decision = parse_decision(raw)
        if decision.done:
            self.state = 'Model reports complete'
            self._discard_plan()
            return f'Model reports the task is complete: {decision.summary}\nReview the target application to confirm the result.'
        self.pending, self.observation = decision, observation
        self.pending_id = secrets.token_hex(3)
        self.state = 'Awaiting approval'
        self._check_stop()
        action = decision.action
        return (f'{decision.summary}\n\nProposed action: {action.kind}\n'
                f'Arguments: {json.dumps(action.args, ensure_ascii=False)}\n'
                f'Region: {self.region_label()}\n\n'
                f'Run /agent approve {self.pending_id} to execute this action and plan the next step.\n'
                'Run /agent stop to discard it. No desktop action has executed yet.')

    def run(self, goal: str, emit: Callable[[str], None]) -> str:
        if not self.model:
            raise ValueError('Select a local vision model first: /agent model llama3.2-vision:latest')
        if self.region is None:
            raise ValueError('Select the target first with /agent select or /agent region LEFT TOP WIDTH HEIGHT.')
        if not goal.strip() or len(goal) > 4000:
            raise ValueError('Describe the desktop task in 1–4000 characters.')
        self.stop_event.clear()
        self._discard_plan()
        self.goal = goal.strip()
        self.steps = 0
        self.last_action = ''
        self.history = []
        return self._plan(emit)

    def _ocr_context(self, png: bytes, emit: Callable[[str], None]) -> str:
        if not self.ocr_enabled:
            return ''
        try:
            lines = self.ocr.read(png)
        except OCRUnavailable as exc:
            emit(f'Local OCR unavailable: {exc} Continuing with vision-only planning.\n')
            return ''
        context = planner_context(lines)
        emit(f'Local OCR found {len(lines)} text line(s); treating text as untrusted context.\n')
        return context

    def read_text(self, emit: Callable[[str], None]) -> str:
        """Read the selected region locally; this does not contact the vision model or click anything."""
        if self.region is None:
            raise ValueError('Select the target first with /agent select or /agent region LEFT TOP WIDTH HEIGHT.')
        self.stop_event.clear()
        self.state = 'Reading selected region'
        emit('Reading text in the selected region locally…\n')
        observation = self.driver.capture(self.region)
        self._check_stop()
        lines = self.ocr.read(observation.png)
        self.state = 'Ready' if self.model else 'Select a vision model'
        if not lines:
            return 'No readable text was found in the selected region. No action was taken.'
        rendered = '\n'.join(
            f'[{line.confidence:.0%} · {line.box[0]},{line.box[1]}–{line.box[2]},{line.box[3]}] {line.text}'
            for line in lines
        )
        return ('Local OCR result (recognition scores, not answer accuracy or action approval):\n' + rendered)

    def approve(self, approval_id: str, emit: Callable[[str], None]) -> str:
        self._check_stop()
        if not self.pending or not self.observation or not approval_id or approval_id != self.pending_id:
            raise ValueError('That approval does not match a pending action. Use /agent run to create a new plan.')
        if time.monotonic() - self.observation.captured_at > self.max_observation_age:
            self._discard_plan()
            self.state = 'Observation expired'
            raise ValueError('The observation expired. Run /agent refresh for a fresh proposal.')
        self.state = 'Checking target'
        current = self.driver.capture(self.region)
        self._check_stop()
        if current.sha256 != self.observation.sha256:
            self._discard_plan()
            self.state = 'Screen changed'
            raise ValueError('The selected screen changed. Nothing was executed. Run /agent refresh and review the new action.')
        action = self.pending.action
        self._discard_plan()  # An approval can be consumed only once.
        self.state = 'Executing'
        emit(f'Executing approved {action.kind}…\n')
        result = self.driver.execute(action, self.region, stop_event=self.stop_event)
        self.steps += 1
        self.last_action = action.kind
        self.history.append({'action': {'kind': action.kind, 'args': action.args}, 'result': result})
        emit(f'{result}\n')
        self._check_stop()
        return self._plan(emit)

    def run_autonomous(self, goal: str, emit: Callable[[str], None]) -> str:
        """Run a bounded task using the selected local vision model and live takeover controls."""
        if not self.model:
            raise ValueError('Select a local vision model first: /agent model llama3.2-vision:latest')
        if self.region is None:
            raise ValueError('Select the target first with /agent select or /agent region LEFT TOP WIDTH HEIGHT.')
        if not goal.strip() or len(goal) > 4000:
            raise ValueError('Describe the desktop task in 1–4000 characters.')

        self.stop_event.clear()
        self.pause_event.clear()
        self._discard_plan()
        self.goal = goal.strip()
        self.steps = 0
        self.last_action = ''
        self.history = []
        self.state = 'Starting autonomous task'
        self.overlay = self.overlay_factory(self.stop_event, self.pause_event)
        self.overlay.start()
        emit('HariAI autonomous desktop control is active.\n')
        emit(f'Goal: {self.goal}\nRegion: {self.region_label()}\n')
        emit('Use Pause in the floating control or press Esc to stop; Take over returns control immediately.\n')

        try:
            client = self.vision_factory(host=self.host, model=self.model)
            # Verify local vision capability before capturing any pixels.
            client.capabilities()
            while self.steps < self.max_steps:
                while self.pause_event.is_set() and not self.stop_event.wait(0.1):
                    self.state = 'Paused'
                self._check_stop()
                self.state = 'Observing'
                emit(f'\nStep {self.steps + 1}/{self.max_steps}: observing the selected region…\n')
                observation = self.driver.capture(self.region)
                self.state = 'Planning'
                raw = client.decide(self.goal, observation.png, self.history, stop_event=self.stop_event,
                                    ocr_context=self._ocr_context(observation.png, emit))
                self._check_stop()
                decision = parse_decision(raw)
                emit(decision.summary + '\n')
                if decision.done:
                    self.state = 'Model reports complete'
                    self._discard_plan()
                    return (f'Model reports the task is complete: {decision.summary}\n'
                            'Review the target application to confirm the result.')

                action = decision.action
                self.state = f'Executing {action.kind}'
                emit(f'Executing {action.kind}…\n')
                result = self.driver.execute(action, self.region, stop_event=self.stop_event)
                self.steps += 1
                self.last_action = action.kind
                self.history.append({'summary': decision.summary,
                                     'action': {'kind': action.kind}, 'result': result})
                emit(result + '\n')

            self.state = 'Step limit reached'
            return (f'Autonomous run paused at its {self.max_steps}-action limit. '
                    'Review the target, then start another task if needed.')
        except (DesktopCancelled, RuntimeError):
            if self.stop_event.is_set():
                self.state = 'Stopped'
                self._discard_plan()
                return 'Autonomous task stopped. Control is back with you.'
            self.state = 'Needs attention'
            raise
        except Exception:
            self.state = 'Needs attention'
            self._discard_plan()
            raise
        finally:
            if self.overlay:
                self.overlay.close()
                self.overlay = None

    def command(self, argument: str, emit: Callable[[str], None] = lambda _: None) -> str:
        command, _, rest = argument.strip().partition(' ')
        command = command.lower() or 'status'
        if command == 'stop':
            self.stop()
            return 'Desktop agent stopped. Pending actions discarded.'
        if command == 'status':
            return self.status_text()
        if command in ('help', '?'):
            return ('/agent model NAME              Select an installed local vision model\n'
                    '/agent host http://127.0.0.1:11434  Set the local service\n'
                    '/agent select                  Switch windows, then drag to select the target\n'
                    '/agent region LEFT TOP WIDTH HEIGHT  Set a region in screen pixels\n'
                    '/agent run TASK                Observe and propose the next action\n'
                    '/agent approve ID              Execute the displayed action once\n'
                    '/agent auto TASK               Execute a bounded task with Pause/Take over controls\n'
                    '/agent ocr on|off              Enable or disable optional local OCR\n'
                    '/agent read                    Read selected-region text locally (no action)\n'
                    '/agent refresh                 Re-observe and propose a fresh action\n'
                    '/agent stop                    Stop and discard pending actions\n'
                    '/agent status                  See task state\n\n'
                    'Requires the desktop extras and a vision-capable local model.\n'
                    'Text-only HariAI Chat is not used to interpret screenshots.\n'
                    'Keep the selected target visible. Move the mouse to a screen corner to abort execution.')
        if not self._lock.acquire(blocking=False):
            return 'The desktop agent is busy. Use /agent stop to request cancellation.'
        try:
            if command == 'model':
                return self.configure(rest.strip())
            if command == 'host':
                return self.configure_host(rest.strip())
            if command == 'ocr':
                choice = rest.strip().lower()
                if choice not in ('on', 'off'):
                    raise ValueError('Usage: /agent ocr on|off')
                return self.configure_ocr(choice == 'on')
            if command == 'read':
                return self.read_text(emit)
            if command == 'region':
                parts = shlex.split(rest)
                if len(parts) != 4:
                    raise ValueError('Usage: /agent region LEFT TOP WIDTH HEIGHT')
                return self.set_region(Region(*(int(part) for part in parts)))
            if command == 'select':
                self.stop_event.clear()
                self.state = 'Selecting region'
                emit('Switch to the target window. Region selection opens in 3 seconds…\n')
                if self.stop_event.wait(3):
                    return 'Selection cancelled.'
                region = self.driver.select_region()
                self._check_stop()
                if region is None:
                    self.state = 'Ready' if self.region and self.model else 'Select a region'
                    return 'Selection cancelled; the existing region was kept.'
                return self.set_region(region)
            if command == 'run':
                return self.run(rest, emit)
            if command in ('auto', 'autonomous'):
                return self.run_autonomous(rest, emit)
            if command == 'approve':
                return self.approve(rest.strip(), emit)
            if command == 'refresh':
                if not self.goal or self.region is None or not self.model:
                    raise ValueError('Start a task with /agent run first.')
                self.stop_event.clear()
                self._discard_plan()
                return self._plan(emit)
            return 'Unknown agent command. Enter /agent help.'
        except Exception:
            self._discard_plan()
            if self.stop_event.is_set():
                self.state = 'Stopped'
            elif self.state not in ('Screen changed', 'Observation expired'):
                self.state = 'Needs attention'
            raise
        finally:
            self._lock.release()
