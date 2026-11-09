"""Unit tests for the HariAI standalone CLI interface."""

import subprocess
import io
import importlib
import os
import sys
import unittest
from unittest.mock import patch


class TestHariCLI(unittest.TestCase):
    def test_interactive_launch_defaults_to_dashboard(self):
        cli = importlib.import_module('hari_ai.cli.main')
        with patch.object(cli.sys.stdin, 'isatty', return_value=True), \
                patch.object(cli.sys.stdout, 'isatty', return_value=True), \
                patch.dict(os.environ, {'TERM': 'xterm-256color'}), \
                patch('hari_ai.cli.dashboard.HariDashboard') as dashboard, patch.object(cli, 'repl') as plain:
            cli.launch_chat()
            dashboard.return_value.run.assert_called_once()
            plain.assert_not_called()

    def test_plain_and_redirected_launch_do_not_start_dashboard(self):
        cli = importlib.import_module('hari_ai.cli.main')
        with patch.object(cli, 'repl') as plain, patch('hari_ai.cli.dashboard.HariDashboard') as dashboard:
            cli.launch_chat(plain=True)
            with patch.object(cli.sys.stdin, 'isatty', return_value=False):
                cli.launch_chat()
            self.assertEqual(plain.call_count, 2)
            dashboard.assert_not_called()

    def test_dashboard_settings_preserve_quoted_windows_path(self):
        cli = importlib.import_module('hari_ai.cli.main')
        with patch.object(cli, 'cmd_settings') as settings:
            cli.dashboard_command('/settings', 'model-file "C:\\My Models\\assistant.gguf"')
            self.assertEqual(settings.call_args.args[0], ['model-file', 'C:\\My Models\\assistant.gguf'])

    def test_confidence_is_grey_in_terminal_and_plain_with_no_color(self):
        cli = importlib.import_module('hari_ai.cli.main')
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        with patch.object(cli._CHAT_ENGINE.settings, 'show_confidence', True), \
                patch.object(cli._CHAT_ENGINE, 'last_reply_succeeded', True), \
                patch.object(cli.sys, 'stdout', output), patch.dict(os.environ, {}, clear=True):
            cli._show_confidence_status()
            self.assertEqual(output.getvalue(), '\x1b[90mToken likelihood: unavailable\x1b[0m\n')
            output.seek(0)
            output.truncate(0)
            with patch.dict(os.environ, {'NO_COLOR': '1'}):
                cli._show_confidence_status()
                self.assertEqual(output.getvalue(), 'Token likelihood: unavailable\n')
            output.seek(0)
            output.truncate(0)
            with patch.object(cli._CHAT_ENGINE, 'last_answer_confidence', .73):
                cli._show_confidence_status()
                self.assertEqual(output.getvalue(), '\x1b[90mConfidence: 73% (arithmetic only)\x1b[0m\n')
            output.seek(0)
            output.truncate(0)
            with patch.object(cli._CHAT_ENGINE, 'last_answer_confidence', 0.0), \
                    patch.object(cli._CHAT_ENGINE, 'last_confidence_source', 'verified arithmetic'):
                cli._show_confidence_status()
                self.assertEqual(output.getvalue(), '\x1b[90mConfidence: 0% (verified arithmetic)\x1b[0m\n')
            output.seek(0)
            output.truncate(0)
            with patch.object(cli._CHAT_ENGINE, 'last_token_likelihood', .725):
                cli._show_confidence_status()
                self.assertEqual(output.getvalue(), '\x1b[90mToken likelihood: 72%\x1b[0m\n')

    def test_direct_identity_shows_unavailable_confidence(self):
        proc = subprocess.run([sys.executable, '-m', 'hari_ai.cli', 'ask', 'who are u?'],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('K R Hari Prajwal', proc.stdout)
        self.assertIn('Token likelihood: unavailable', proc.stdout)
        self.assertNotIn('%', proc.stdout)
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

    def test_status_reports_real_backend_and_experimental_learning(self):
        proc = subprocess.run([sys.executable, '-m', 'hari_ai.cli', 'status'],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('HariAI status', proc.stdout)
        self.assertIn('no automatic model updates', proc.stdout)
        self.assertIn('answer accuracy is not calibrated', proc.stdout)

    def test_cli_version_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "hari_ai.cli", "version"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        from hari_ai.version import __version__
        self.assertIn(f"HariAI version {__version__}", proc.stdout)

    def test_repl_reports_running_version(self):
        from hari_ai.version import __version__
        proc = subprocess.run([sys.executable, '-m', 'hari_ai.cli'],
                              input='/version\n/exit\n', capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertIn(f'HariAI v{__version__}', proc.stdout)
        self.assertIn(f'HariAI version {__version__}', proc.stdout)

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

