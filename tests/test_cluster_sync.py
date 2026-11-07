"""Unit tests for Phase 12 Cluster-Wide Learning and Federated Experience Synchronization."""

import unittest
import torch
from hari_ai.cluster.sync import (
    ConflictResolutionPolicy,
    ExperiencePackage,
    FederatedMemorySynchronizer,
    FederatedParameterAuditor,
)
from hari_ai.memory.hierarchy import HierarchicalMemoryManager


class TestClusterSync(unittest.TestCase):
    def setUp(self):
        self.memory = HierarchicalMemoryManager(session_id="cluster_sync_test")
        self.synchronizer = FederatedMemorySynchronizer(
            policy=ConflictResolutionPolicy.VERIFIED_SUPREMACY,
            min_confidence_threshold=0.5,
        )
        self.auditor = FederatedParameterAuditor(max_norm_threshold=10.0)

    def test_sync_facts_into_semantic_memory(self):
        pkg = ExperiencePackage(
            node_id="worker-node-1",
            package_id="exp-fact-1",
            task_type="fact",
            subject="HariAI Architecture",
            claim_or_code="UAIA-Nano has 46.4M parameters and fits in under 50 MB.",
            is_verified=True,
            verification_method="deterministic_spec",
            confidence=1.0,
        )
        result = self.synchronizer.reconcile_and_sync([pkg], self.memory)

        self.assertEqual(result.total_submitted, 1)
        self.assertEqual(result.facts_synchronized, 1)
        self.assertEqual(result.recipes_synchronized, 0)
        self.assertEqual(result.unverified_rejected, 0)

        # Verify fact was stored in semantic memory
        queried = self.memory.semantic.query("HariAI Architecture")
        self.assertEqual(len(queried), 1)
        self.assertIn("46.4M", queried[0].object_value)

    def test_sync_recipes_into_procedural_memory(self):
        pkg = ExperiencePackage(
            node_id="worker-node-2",
            package_id="exp-code-1",
            task_type="code",
            subject="binary_search",
            claim_or_code="def binary_search(arr, x): ...",
            is_verified=True,
            verification_method="unit_test",
            confidence=1.0,
        )
        result = self.synchronizer.reconcile_and_sync([pkg], self.memory)

        self.assertEqual(result.total_submitted, 1)
        self.assertEqual(result.recipes_synchronized, 1)
        self.assertEqual(result.facts_synchronized, 0)

        # Verify recipe was stored in procedural memory
        recipes = self.memory.procedural.lookup("binary_search")
        self.assertEqual(len(recipes), 1)
        self.assertTrue(recipes[0].verification_passed)

    def test_conflict_resolution_verified_supremacy(self):
        # Node 1 submits unverified empirical guess
        pkg_unverified = ExperiencePackage(
            node_id="worker-node-1",
            package_id="guess-1",
            task_type="fact",
            subject="Quantum Supremacy",
            claim_or_code="Claim A: Unverified observation",
            is_verified=False,
            confidence=0.8,
            timestamp=10.0,
        )
        # Node 2 submits formally verified theorem
        pkg_verified = ExperiencePackage(
            node_id="worker-node-2",
            package_id="proof-1",
            task_type="fact",
            subject="Quantum Supremacy",
            claim_or_code="Theorem: Formally verified Lean 4 proof",
            is_verified=True,
            verification_method="lean4",
            confidence=1.0,
            timestamp=12.0,
        )

        result = self.synchronizer.reconcile_and_sync(
            [pkg_unverified, pkg_verified], self.memory
        )

        self.assertEqual(result.total_submitted, 2)
        self.assertEqual(result.conflicts_detected, 1)
        self.assertEqual(result.facts_synchronized, 1)

        # Formally verified theorem must win
        queried = self.memory.semantic.query("Quantum Supremacy")
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried[0].source_entry_id, "worker-node-2")
        self.assertIn("Lean 4 proof", queried[0].object_value)

    def test_rejection_of_low_confidence_unverified(self):
        pkg_low_conf = ExperiencePackage(
            node_id="worker-node-3",
            package_id="low-conf-1",
            task_type="fact",
            subject="Speculative Hypothesis",
            claim_or_code="Uncertain hallucination",
            is_verified=False,
            confidence=0.2,  # below 0.5 threshold
        )
        result = self.synchronizer.reconcile_and_sync([pkg_low_conf], self.memory)

        self.assertEqual(result.total_submitted, 1)
        self.assertEqual(result.unverified_rejected, 1)
        self.assertEqual(result.facts_synchronized, 0)
        self.assertEqual(len(self.memory.semantic.query("Speculative Hypothesis")), 0)

    def test_federated_parameter_auditor(self):
        # Normal valid adapter
        valid_adapter = {
            "lora_A": torch.randn(4, 32) * 0.1,
            "lora_B": torch.randn(64, 4) * 0.1,
        }
        self.assertTrue(self.auditor.audit_adapter(valid_adapter))

        # Poisoned adapter with NaN
        nan_adapter = {
            "lora_A": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
        }
        self.assertFalse(self.auditor.audit_adapter(nan_adapter))

        # Explosive adapter exceeding L2 norm threshold
        huge_adapter = {
            "lora_A": torch.ones(10, 10) * 100.0,
        }
        self.assertFalse(self.auditor.audit_adapter(huge_adapter))

    def test_federated_averaging(self):
        # Node 1: weights filled with 1.0, 100 samples
        adapter_1 = {"weight": torch.full((2, 2), 1.0)}
        # Node 2: weights filled with 3.0, 100 samples
        adapter_2 = {"weight": torch.full((2, 2), 3.0)}

        # FedAvg should yield exactly (1.0 * 100 + 3.0 * 100) / 200 = 2.0
        averaged = self.auditor.federated_average(
            [(adapter_1, 100), (adapter_2, 100)]
        )
        self.assertIsNotNone(averaged)
        self.assertTrue(torch.allclose(averaged["weight"], torch.full((2, 2), 2.0)))


if __name__ == "__main__":
    unittest.main()

