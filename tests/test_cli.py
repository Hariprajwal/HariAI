"""Unit tests for the HariAI standalone CLI interface."""

import subprocess
import sys
import unittest


class TestHariCLI(unittest.TestCase):
    def test_cli_info_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "info"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("K R Hari Prajwal", proc.stdout)
        self.assertIn("Under 35 MB", proc.stdout)
        self.assertIn("21 years old", proc.stdout)

    def test_cli_version_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "version"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("HariAI version 0.1.0", proc.stdout)

    def test_cli_decide_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "decide", "Mass deletion requested"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Choice", proc.stdout)
        self.assertIn("Score", proc.stdout)
        self.assertIn("Noul", proc.stdout)
        self.assertIn("REQUIRES APPROVAL", proc.stdout)

    def test_cli_dream_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "dream", "now"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("HariAI Dream Engine", proc.stdout)
        self.assertIn("Dream cycle completed", proc.stdout)

    def test_cli_think_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "think", "Sort array in O(n log n)"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("[Verified Solution]", proc.stdout)

    def test_cli_search_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "search", "transformer attention"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("HariNano Search Engine", proc.stdout)
        self.assertIn("Search & Research Briefing", proc.stdout)


if __name__ == "__main__":
    unittest.main()

