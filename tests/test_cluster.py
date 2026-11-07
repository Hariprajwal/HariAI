"""Unit tests for HariAI Multi-Agent Cluster Architecture."""

import unittest
from hari_ai.cluster.blackboard import SharedBlackboard
from hari_ai.cluster.coordinator import Cluster, ClusterCoordinator
from hari_ai.cluster.protocol import AgentRole, ClusterMessage, TaskStatus


class TestClusterArchitecture(unittest.TestCase):
    def test_cluster_message_serialization(self):
        msg = ClusterMessage(
            sender=AgentRole.COORDINATOR.value,
            receiver=AgentRole.RESEARCHER.value,
            task_id="t-101",
            objective="Analyze complexity",
            status=TaskStatus.IN_PROGRESS,
            confidence=0.95,
        )

        json_str = msg.to_json()
        restored = ClusterMessage.from_json(json_str)

        self.assertEqual(restored.sender, msg.sender)
        self.assertEqual(restored.receiver, msg.receiver)
        self.assertEqual(restored.task_id, msg.task_id)
        self.assertEqual(restored.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(restored.confidence, 0.95)

    def test_shared_blackboard_and_artifacts(self):
        bb = SharedBlackboard()
        bb.set_artifact("code", "def foo(): pass")
        self.assertEqual(bb.get_artifact("code"), "def foo(): pass")
        self.assertIsNone(bb.get_artifact("missing_key"))

    def test_execution_budget_guard(self):
        bb = SharedBlackboard(max_messages=3)
        # Post 3 messages
        for i in range(3):
            bb.post_message(
                ClusterMessage(
                    sender=f"worker_{i}",
                    receiver="coordinator",
                    task_id="t-1",
                    objective="task",
                )
            )

        # 4th message should raise message budget error
        with self.assertRaises(RuntimeError) as ctx:
            bb.post_message(
                ClusterMessage(
                    sender="worker_99",
                    receiver="coordinator",
                    task_id="t-1",
                    objective="task",
                )
            )
        self.assertIn("Cluster message budget exceeded", str(ctx.exception))

    def test_anti_loop_circular_delegation_guard(self):
        bb = SharedBlackboard(max_messages=20)
        # Ping-pong between agent_a and agent_b with identical objective
        m1 = ClusterMessage(sender="agent_a", receiver="agent_b", task_id="t", objective="repeat")
        m2 = ClusterMessage(sender="agent_b", receiver="agent_a", task_id="t", objective="repeat")

        bb.post_message(m1)
        bb.post_message(m2)
        bb.post_message(m1)

        # 4th message completes the alternating loop
        with self.assertRaises(RuntimeError) as ctx:
            bb.post_message(m2)
        self.assertIn("Anti-loop guard triggered", str(ctx.exception))

    def test_cluster_coordinator_end_to_end_execution(self):
        cluster = Cluster()
        cluster.start(workers=3)

        result = cluster.run("Sort an array of integers with verified tests")

        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["verification_passed"])
        self.assertIn("assert ", result["tests"])
        self.assertGreaterEqual(result["total_cluster_messages"], 4)
        self.assertIn("coder", result["agents_involved"])
        self.assertIn("verifier", result["agents_involved"])


if __name__ == "__main__":
    unittest.main()
