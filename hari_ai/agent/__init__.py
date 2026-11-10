"""Explicit desktop tasks with observable, bounded action execution."""
from hari_ai.agent.autonomous import AutonomousAgent, AutonomousOverlay, AutonomousResult
from hari_ai.agent.controller import DesktopAgentController
from hari_ai.agent.desktop import Action, Decision, DesktopCancelled, DesktopDriver, DesktopError, Region

__all__ = [
    "Action",
    "AutonomousAgent",
    "AutonomousOverlay",
    "AutonomousResult",
    "Decision",
    "DesktopCancelled",
    "DesktopDriver",
    "DesktopError",
    "DesktopAgentController",
    "Region",
]

