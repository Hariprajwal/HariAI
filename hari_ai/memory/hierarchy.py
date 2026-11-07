"""Hierarchical multi-level memory architecture for HariAI.

Implements Levels 1 through 5:
- Level 1: Working Memory (active conversation & transient context)
- Level 2: Episodic Memory (interaction history & raw experience logs)
- Level 3: Semantic Memory (consolidated facts & conceptual graph)
- Level 4: Procedural Memory (executable code patterns & decision policies)
- Level 5: Parametric Memory (frozen base weights + modular low-rank adapters)
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MemoryLevel(str, enum.Enum):
    WORKING = "level_1_working"
    EPISODIC = "level_2_episodic"
    SEMANTIC = "level_3_semantic"
    PROCEDURAL = "level_4_procedural"
    PARAMETRIC = "level_5_parametric"


@dataclass
class WorkingMemory:
    """Level 1: Transient working memory for active inference."""

    session_id: str
    active_prompt: str = ""
    scratchpad: List[str] = field(default_factory=list)
    context_tokens: List[int] = field(default_factory=list)

    def append_thought(self, thought: str) -> None:
        self.scratchpad.append(thought)

    def clear(self) -> None:
        self.active_prompt = ""
        self.scratchpad.clear()
        self.context_tokens.clear()


@dataclass
class EpisodicEntry:
    """A single logged interaction or experience."""

    entry_id: str
    timestamp: float
    user_input: str
    assistant_output: str
    action_taken: Optional[str] = None
    was_successful: bool = True
    user_feedback: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EpisodicMemory:
    """Level 2: Episodic memory buffer storing raw interaction experiences."""

    entries: List[EpisodicEntry] = field(default_factory=list)

    def record(self, entry: EpisodicEntry) -> None:
        self.entries.append(entry)

    def get_unconsolidated_experiences(self, limit: int = 100) -> List[EpisodicEntry]:
        return self.entries[-limit:]

    def get_recent_failures(self) -> List[EpisodicEntry]:
        return [e for e in self.entries if not e.was_successful or e.user_feedback]


@dataclass
class SemanticFact:
    """A consolidated conceptual fact or entity relationship."""

    fact_id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float
    source_entry_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SemanticMemory:
    """Level 3: Consolidated facts and concept knowledge store."""

    facts: Dict[str, SemanticFact] = field(default_factory=dict)

    def insert_fact(self, fact: SemanticFact) -> None:
        self.facts[fact.fact_id] = fact

    def query(self, subject: str) -> List[SemanticFact]:
        return [f for f in self.facts.values() if f.subject.lower() == subject.lower()]


@dataclass
class ProceduralRecipe:
    """A verified tool recipe, algorithm template, or workflow."""

    recipe_id: str
    title: str
    pattern_description: str
    code_or_schema: str
    verification_passed: bool
    version: int = 1


@dataclass
class ProceduralMemory:
    """Level 4: Procedural memory storing verified code and decision policies."""

    recipes: Dict[str, ProceduralRecipe] = field(default_factory=dict)

    def register_recipe(self, recipe: ProceduralRecipe) -> None:
        self.recipes[recipe.recipe_id] = recipe

    def lookup(self, query: str) -> List[ProceduralRecipe]:
        q = query.lower()
        return [
            r for r in self.recipes.values()
            if q in r.title.lower() or q in r.pattern_description.lower()
        ]


class HierarchicalMemoryManager:
    """Coordinator across all memory levels."""

    def __init__(self, session_id: str = "default_session"):
        self.working = WorkingMemory(session_id=session_id)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns statistics across all memory tiers."""
        return {
            "working_scratchpad_items": len(self.working.scratchpad),
            "episodic_entries_count": len(self.episodic.entries),
            "recent_failures_count": len(self.episodic.get_recent_failures()),
            "semantic_facts_count": len(self.semantic.facts),
            "procedural_recipes_count": len(self.procedural.recipes),
        }

