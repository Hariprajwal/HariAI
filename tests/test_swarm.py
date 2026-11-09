"""Unit tests for MultiHariSwarm multi-bot dialogue engine."""

import unittest
from hari_ai.cluster.swarm import MultiHariSwarm, SwarmBotPersona, DEFAULT_SWARM_PERSONAS


class TestMultiHariSwarm(unittest.TestCase):
    def setUp(self):
        self.swarm = MultiHariSwarm()

    def test_initial_status(self):
        status = self.swarm.get_status()
        self.assertFalse(status["active"])
        self.assertIn("Standalone", status["mode"])

    def test_run_dialogue_generates_all_bot_turns(self):
        turns = list(self.swarm.run_dialogue("Distributed Cache Invalidation", rounds=1))
        self.assertEqual(len(turns), 3)

        self.assertEqual(turns[0].bot_name, "Hari-Alpha")
        self.assertEqual(turns[0].role, "Architect & Proposer")
        self.assertIn("Proposed Architecture", turns[0].content)

        self.assertEqual(turns[1].bot_name, "Hari-Beta")
        self.assertEqual(turns[1].role, "Critic & Security Auditor")
        self.assertIn("Critical Analysis", turns[1].content)

        self.assertEqual(turns[2].bot_name, "Hari-Gamma")
        self.assertEqual(turns[2].role, "Synthesizer & Arbiter")
        self.assertIn("Unified Consensus", turns[2].content)

        # Status after dialogue
        status = self.swarm.get_status()
        self.assertEqual(status["workers"], "3 stand-by")
        self.assertEqual(status["activity"], "Consensus reached")

    def test_custom_persona_injection(self):
        custom_personas = [
            SwarmBotPersona("Hari-1", "Proposer", "System prompt 1"),
            SwarmBotPersona("Hari-2", "Critic", "System prompt 2"),
        ]
        custom_swarm = MultiHariSwarm(personas=custom_personas)
        self.assertEqual(custom_swarm.workers_count, 2)
        turns = list(custom_swarm.run_dialogue("Quantum Algorithm", rounds=1))
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].bot_name, "Hari-1")
        self.assertEqual(turns[1].bot_name, "Hari-2")


if __name__ == "__main__":
    unittest.main()
