"""Cluster-Wide Learning and Federated Experience Synchronization for HariAI.

Implements Phase 12:
- Distributed experience aggregation across multi-agent nodes.
- Conflict resolution with Verified Supremacy (Lean 4 proofs and deterministic unit tests strictly override unverified claims).
- Shared Semantic and Procedural memory consolidation.
- Coordinator-audited parameter-efficient adapter synchronization (FedAvg with safety norm checks).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from hari_ai.memory.hierarchy import (
    HierarchicalMemoryManager,
    ProceduralRecipe,
    SemanticFact,
)


class ConflictResolutionPolicy(str, enum.Enum):
    VERIFIED_SUPREMACY = "verified_supremacy"
    MAJORITY_CONSENSUS = "majority_consensus"
    HIGHEST_CONFIDENCE = "highest_confidence"


@dataclass
class ExperiencePackage:
    """A unit of experiential learning contributed by a cluster node."""

    node_id: str
    package_id: str
    task_type: str  # "code", "math", "fact", "decision"
    subject: str
    claim_or_code: str
    is_verified: bool = False
    verification_method: Optional[str] = None  # "lean4", "unit_test", "empirical"
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncResult:
    """Report summarizing the outcome of a cluster memory synchronization cycle."""

    total_submitted: int
    facts_synchronized: int
    recipes_synchronized: int
    conflicts_detected: int
    unverified_rejected: int
    synchronized_keys: List[str] = field(default_factory=list)


class FederatedMemorySynchronizer:
    """Orchestrates decentralized knowledge aggregation and safe memory synchronization."""

    def __init__(
        self,
        policy: ConflictResolutionPolicy = ConflictResolutionPolicy.VERIFIED_SUPREMACY,
        min_confidence_threshold: float = 0.5,
    ):
        self.policy = policy
        self.min_confidence_threshold = min_confidence_threshold

    def reconcile_and_sync(
        self,
        packages: List[ExperiencePackage],
        target_memory: HierarchicalMemoryManager,
    ) -> SyncResult:
        """Gathers distributed experiences, resolves conflicts, and promotes verified insights."""
        result = SyncResult(
            total_submitted=len(packages),
            facts_synchronized=0,
            recipes_synchronized=0,
            conflicts_detected=0,
            unverified_rejected=0,
        )

        # Group experiences by subject
        subject_groups: Dict[str, List[ExperiencePackage]] = {}
        for pkg in packages:
            subject_groups.setdefault(pkg.subject, []).append(pkg)

        for subject, group in subject_groups.items():
            if len(group) > 1:
                # Conflict or multiple observations detected
                result.conflicts_detected += 1
                winner = self._resolve_conflict(group)
            else:
                winner = group[0]

            # Reject unverified or low-confidence submissions
            if not winner.is_verified and winner.confidence < self.min_confidence_threshold:
                result.unverified_rejected += 1
                continue

            # Promote into target memory based on task_type
            if winner.task_type in ("code", "decision"):
                recipe = ProceduralRecipe(
                    recipe_id=f"cluster-{winner.package_id}",
                    title=f"Recipe: {winner.subject}",
                    pattern_description=f"Synchronized from node {winner.node_id}",
                    code_or_schema=winner.claim_or_code,
                    verification_passed=winner.is_verified,
                )
                target_memory.procedural.register_recipe(recipe)
                result.recipes_synchronized += 1
                result.synchronized_keys.append(recipe.recipe_id)

            else:  # "fact", "math", or general knowledge
                fact = SemanticFact(
                    fact_id=f"cluster-{winner.package_id}",
                    subject=winner.subject,
                    predicate="asserts" if winner.is_verified else "claims",
                    object_value=winner.claim_or_code,
                    confidence=1.0 if winner.is_verified else winner.confidence,
                    source_entry_id=winner.node_id,
                )
                target_memory.semantic.insert_fact(fact)
                result.facts_synchronized += 1
                result.synchronized_keys.append(fact.fact_id)

        return result

    def _resolve_conflict(self, candidates: List[ExperiencePackage]) -> ExperiencePackage:
        """Resolves contradictory findings across nodes."""
        if self.policy == ConflictResolutionPolicy.VERIFIED_SUPREMACY:
            # Verified candidates strictly override unverified ones
            verified = [c for c in candidates if c.is_verified]
            if verified:
                # Prefer formal verification (Lean 4) over unit test
                formal = [c for c in verified if c.verification_method == "lean4"]
                pool = formal if formal else verified
                return max(pool, key=lambda c: (c.confidence, c.timestamp))

        # Fallback to highest confidence and recency
        return max(candidates, key=lambda c: (c.confidence, c.timestamp))


class FederatedParameterAuditor:
    """Audits and synchronizes parameter-efficient adapter weights across distributed nodes."""

    def __init__(self, max_norm_threshold: float = 10.0):
        self.max_norm_threshold = max_norm_threshold

    def audit_adapter(self, adapter_state_dict: Dict[str, torch.Tensor]) -> bool:
        """Validates that candidate adapter weights are well-formed and numerically stable."""
        for param_name, tensor in adapter_state_dict.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
            norm = torch.norm(tensor.float()).item()
            if norm > self.max_norm_threshold:
                return False
        return True

    def federated_average(
        self,
        node_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Computes weighted Federated Averaging (FedAvg) over audited adapter weights."""
        valid_updates = [
            (weights, samples)
            for weights, samples in node_updates
            if self.audit_adapter(weights) and samples > 0
        ]
        if not valid_updates:
            return None

        total_samples = sum(samples for _, samples in valid_updates)
        averaged_state: Dict[str, torch.Tensor] = {}

        first_weights = valid_updates[0][0]
        for key in first_weights.keys():
            accumulated = torch.zeros_like(first_weights[key].float())
            for weights, samples in valid_updates:
                weight = samples / total_samples
                accumulated += weights[key].float() * weight
            averaged_state[key] = accumulated.to(first_weights[key].dtype)

        return averaged_state

