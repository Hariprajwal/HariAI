"""Inter-Agent Communication Protocol for HariAI Cluster.

Supports:
- Predefined specialized roles: Coordinator, Researcher, Coder, Reasoner, Decision Specialist, Verifier, Security Auditor
- Universal Mode: Adaptive generalist worker executing end-to-end tasks
- Custom User-Defined Roles: Runtime registration of dynamic domain-specific roles
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class AgentRole(str, enum.Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    CODER = "coder"
    REASONER = "reasoner"
    DECISION_SPECIALIST = "decision_specialist"
    VERIFIER = "verifier"
    SECURITY_AUDITOR = "security_auditor"  # New security and permission boundary specialist
    UNIVERSAL = "universal"  # Universal generalist mode
    CUSTOM = "custom"  # User-defined dynamic role


@dataclass
class CustomRoleDefinition:
    """User-defined custom agent role specification."""

    name: str
    description: str
    instructions: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClusterMessage:
    """Standardized message envelope for inter-agent communication."""

    sender: str
    receiver: str
    task_id: str
    objective: str
    status: TaskStatus = TaskStatus.IN_PROGRESS
    findings: str = ""
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    requested_action: Optional[str] = None
    is_complete: bool = False
    context_ref: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterMessage":
        data = data.copy()
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ClusterMessage":
        return cls.from_dict(json.loads(json_str))
