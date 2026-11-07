"""Unit tests for HariAI Multi-Agent Cluster Architecture with Custom Roles and Universal Mode."""

import unittest
from hari_ai.cluster.blackboard import SharedBlackboard
from hari_ai.cluster.coordinator import Cluster, ClusterCoordinator
from hari_ai.cluster.protocol import AgentRole, ClusterMessage, CustomRoleDefinition, TaskStatus


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
        for i in range(3):
            bb.post_message(
                ClusterMessage(
                    sender=f"worker_{i}",
                    receiver="coordinator",
                    task_id="t-1",
                    objective="task",
                )
            )

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
        m1 = ClusterMessage(sender="agent_a", receiver="agent_b", task_id="t", objective="repeat")
        m2 = ClusterMessage(sender="agent_b", receiver="agent_a", task_id="t", objective="repeat")

        bb.post_message(m1)
        bb.post_message(m2)
        bb.post_message(m1)

        with self.assertRaises(RuntimeError) as ctx:
            bb.post_message(m2)
        self.assertIn("Anti-loop guard triggered", str(ctx.exception))

    def test_specialized_cluster_with_security_auditor(self):
        cluster = Cluster()
        cluster.start(workers=4)

        result = cluster.run("Sort an array of integers with verified tests")

        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["verification_passed"])
        self.assertTrue(result["security_cleared"])
        self.assertIn("security_auditor", result["agents_involved"])
        self.assertIn("coder", result["agents_involved"])
        self.assertIn("verifier", result["agents_involved"])

    def test_universal_mode_execution(self):
        cluster = Cluster(universal_mode=True)
        cluster.start()

        result = cluster.run("General problem solving under universal mode")

        self.assertEqual(result["mode"], "universal")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["verification_passed"])
        self.assertIn("universal", result["agents_involved"])

    def test_custom_user_defined_roles(self):
        cluster = Cluster()
        # Register a custom DevOps/Deployer role
        custom_role = cluster.register_custom_role(
            name="DevOps_Engineer",
            description="Manages container manifests and deployment sanity",
            instructions="Check Dockerfile syntax and entrypoint configurations.",
            capabilities=["docker", "ci_cd", "deployment"],
        )
        self.assertEqual(custom_role.name, "DevOps_Engineer")

        cluster.start()
        result = cluster.run(
            "Build and package solution with deployment checks",
            roles=["DevOps_Engineer"],
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertIn("DevOps_Engineer", result["agents_involved"])


if __name__ == "__main__":
    unittest.main()
