"""Unit tests for the hierarchical memory architecture."""

import unittest
from hari_ai.memory.hierarchy import (
    EpisodicEntry,
    HierarchicalMemoryManager,
    ProceduralRecipe,
    SemanticFact,
)


class TestMemoryHierarchy(unittest.TestCase):
    def setUp(self):
        self.mgr = HierarchicalMemoryManager(session_id="test_session")

    def test_working_memory(self):
        self.mgr.working.append_thought("Analyzing algorithm complexity")
        self.mgr.working.append_thought("Complexity is O(n log n)")
        self.assertEqual(len(self.mgr.working.scratchpad), 2)
        self.mgr.working.clear()
        self.assertEqual(len(self.mgr.working.scratchpad), 0)

    def test_episodic_memory_tracking(self):
        entry_success = EpisodicEntry(
            entry_id="exp-1",
            timestamp=1000.0,
            user_input="Write quicksort in python",
            assistant_output="def quicksort(arr): ...",
            action_taken="code_generation",
            was_successful=True,
        )
        entry_failure = EpisodicEntry(
            entry_id="exp-2",
            timestamp=1001.0,
            user_input="Fix segmentation fault in pointer code",
            assistant_output="int *p = NULL; *p = 5;",
            action_taken="code_debugging",
            was_successful=False,
            user_feedback="Null pointer dereference still occurs",
        )
        self.mgr.episodic.record(entry_success)
        self.mgr.episodic.record(entry_failure)

        self.assertEqual(len(self.mgr.episodic.entries), 2)
        failures = self.mgr.episodic.get_recent_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].entry_id, "exp-2")

    def test_semantic_memory(self):
        fact = SemanticFact(
            fact_id="fact-1",
            subject="HariAI",
            predicate="target_size",
            object_value="below 50MB",
            confidence=0.99,
        )
        self.mgr.semantic.insert_fact(fact)
        results = self.mgr.semantic.query("HariAI")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].object_value, "below 50MB")

    def test_procedural_memory(self):
        recipe = ProceduralRecipe(
            recipe_id="recipe-git-init",
            title="Initialize and commit local git",
            pattern_description="How to initialize git repository and commit",
            code_or_schema="git init && git add . && git commit -m 'Initial commit'",
            verification_passed=True,
        )
        self.mgr.procedural.register_recipe(recipe)
        lookup = self.mgr.procedural.lookup("git")
        self.assertEqual(len(lookup), 1)
        self.assertEqual(lookup[0].recipe_id, "recipe-git-init")


if __name__ == "__main__":
    unittest.main()
