"""Compatibility API for bounded autonomous desktop tasks.

Desktop observations and decisions use the configured local vision model through
DesktopAgentController. This module intentionally contains no simulated planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional

from hari_ai.agent.controller import DesktopAgentController
from hari_ai.agent.desktop import DesktopDriver, Region
from hari_ai.agent.overlay import TakeoverOverlay

AutonomousOverlay = TakeoverOverlay


@dataclass
class AutonomousStep:
    step_number: int
    thought: str
    action_kind: str
    action_args: Dict[str, Any]
    result: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AutonomousResult:
    goal: str
    success: bool
    status: str
    total_steps: int
    summary: str
    steps: List[AutonomousStep]
    interrupted_by_user: bool = False
    error: Optional[str] = None


class AutonomousAgent:
    """Public convenience wrapper around the production desktop controller.

    Call ``controller.configure(model)`` and set a region before ``run``. The
    returned ``success`` stays false because model-reported completion is not an
    independent check of the target application.
    """

    def __init__(self, driver: Optional[DesktopDriver] = None,
                 engine_getter: Optional[Callable[[], Any]] = None,
                 search_engine: Any = None, *, controller: Optional[DesktopAgentController] = None,
                 vision_factory=None, overlay_factory=None):
        del engine_getter, search_engine  # Kept for source compatibility with the former stub.
        kwargs = {}
        if driver is not None:
            kwargs['driver'] = driver
        if vision_factory is not None:
            kwargs['vision_factory'] = vision_factory
        if overlay_factory is not None:
            kwargs['overlay_factory'] = overlay_factory
        self.controller = controller or DesktopAgentController(**kwargs)

    def stop(self) -> None:
        self.controller.stop()

    def run(self, goal: str, region: Optional[Region] = None, max_steps: int = 8,
            emit: Callable[[str], None] = print, show_overlay: bool = True) -> AutonomousResult:
        if region is not None:
            self.controller.set_region(region)
        self.controller.max_steps = max(1, min(int(max_steps), 32))
        if not show_overlay:
            self.controller.overlay_factory = lambda *_: _NullOverlay()
        try:
            summary = self.controller.command('auto ' + goal, emit=emit)
            status = self.controller.state.lower().replace(' ', '_')
            return AutonomousResult(
                goal=goal.strip(), success=False, status=status,
                total_steps=self.controller.steps, summary=summary,
                steps=[AutonomousStep(
                    step_number=index + 1,
                    thought=str(item.get('summary', '')),
                    action_kind=str((item.get('action') or {}).get('kind', '')),
                    action_args={}, result=str(item.get('result', '')),
                ) for index, item in enumerate(self.controller.history)],
                interrupted_by_user=self.controller.state == 'Stopped',
            )
        except Exception as exc:
            return AutonomousResult(goal=goal.strip(), success=False, status='needs_attention',
                                    total_steps=self.controller.steps, summary=str(exc),
                                    steps=[], error=str(exc))


class _NullOverlay:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass
