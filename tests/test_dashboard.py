"""Exercise the actual terminal app with controlled local inference workers."""
from __future__ import annotations

import asyncio
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.text import Text
from textual.widgets import Button, Input

from hari_ai.cli.dashboard import DIM, TerminalImage, ChatTurn, CreatorCard, HariDashboard
from hari_ai.cli.art import find_mahakal_path
from hari_ai.dream.controller import DreamEngineController
from hari_ai.memory.hierarchy import HierarchicalMemoryManager
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.settings import ChatSettings


class ControlledLocalModel:
    """Expose one chunk immediately and gate the rest without timing inference."""

    def __init__(self, path: Path):
        self.path = path
        self.context_size = 2048
        self.threads = 2
        self._model = None
        self.last_finish_reason = None
        self.last_token_likelihood = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = threading.Event()
        self.failure = ''
        self.pieces = ['Hello ', 'from HariAI.']
        self.calls = []

    def chat_stream(self, messages):
        self.calls.append(messages)
        self.started.set()
        self.closed.clear()
        try:
            if self.failure:
                raise RuntimeError(self.failure)
            yield self.pieces[0]
            if not self.release.wait(timeout=10):
                raise RuntimeError('Test did not release the model worker')
            yield from self.pieces[1:]
            self.last_finish_reason = 'stop'
            self.last_token_likelihood = 0.83
        finally:
            self.closed.set()


class TestHariDashboard(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / 'test.gguf'
        self.model_path.write_bytes(b'placeholder')
        self.memory = HierarchicalMemoryManager()
        self.dream = DreamEngineController(self.memory)
        self.local = ControlledLocalModel(self.model_path)
        self.chat = self.new_engine(self.local)
        self.commands = []
        self.app = HariDashboard(lambda: self.chat, self.dream, self.command)

    def tearDown(self):
        self.local.release.set()

    def new_engine(self, model):
        return ChatEngine(settings=ChatSettings(), memory_manager=self.memory, local_model=model)

    def command(self, name, argument):
        self.commands.append((name, argument))
        if name == '/settings' and argument == 'replace':
            replacement = ControlledLocalModel(self.model_path)
            replacement.context_size = 512
            replacement.pieces = ['Replacement ', 'model reply.']
            self.chat = self.new_engine(replacement)
            return 'Model settings updated.'
        if name == '/info':
            return 'K R Hari Prajwal created HariAI.'
        if name == '/help':
            return '/status /info /memory /dream /settings /clear'
        return f'{name}: {argument or "current state"}'

    async def wait_for(self, pilot, condition):
        async def poll():
            while not condition():
                await pilot.pause(0.01)
        await asyncio.wait_for(poll(), timeout=8)
        await pilot.pause()

    def replies(self):
        return list(self.app.query('.assistant-turn').results(ChatTurn))

    async def test_responsive_layout_and_f2_panel_navigation(self):
        async with self.app.run_test(size=(160, 54)) as pilot:
            self.assertTrue(self.app.query_one('#sidebar').display)
            self.assertTrue(self.app.query_one('#main-pane').display)
            self.assertEqual(len(self.app.query('#sidebar .panel')), 6)
            image = self.app.query_one('#mahakal-image')
            if TerminalImage is not None:
                self.assertIsInstance(image, TerminalImage)
                self.assertEqual(Path(image.image), find_mahakal_path())
                self.assertEqual(image.region.height, 18)
                self.assertLess(image.region.width, self.app.query_one('#mahakal').region.width)
                self.assertEqual(TerminalImage.__name__, 'HalfcellImage')
            rendered = self.app.export_screenshot()
            self.assertEqual(rendered.count('Capabilities'), 1)
            self.assertEqual(rendered.count('Ready'), 1)
            self.assertIs(self.app.focused, self.app.query_one('#prompt'))
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertFalse(self.app.query_one('#sidebar').display)
            self.assertTrue(self.app.query_one('#main-pane').display)
            await pilot.press('f2')
            self.assertTrue(self.app.query_one('#sidebar').display)
            self.assertFalse(self.app.query_one('#main-pane').display)
            await pilot.press('f2')
            self.assertFalse(self.app.query_one('#sidebar').display)
            self.assertTrue(self.app.query_one('#main-pane').display)
            self.assertIs(self.app.focused, self.app.query_one('#prompt'))
            await pilot.resize_terminal(160, 54)
            await pilot.pause()
            await pilot.press('f2')
            self.assertTrue(self.app.query_one('#sidebar').display)
            self.assertTrue(self.app.query_one('#main-pane').display)

    async def test_streaming_reply_finishes_with_grey_likelihood_and_live_counts(self):
        self.local.release.clear()
        async with self.app.run_test(size=(160, 54)) as pilot:
            try:
                await self.app.submit_prompt('Say hello')
                await self.wait_for(pilot, lambda: self.replies()[-1].content == 'Hello ')
                reply = self.replies()[-1]
                self.assertTrue(self.app.busy)
                self.assertFalse(reply.note.display)
                self.assertTrue(all(button.disabled for button in self.app.query(Button)))
                self.local.release.set()
                await self.wait_for(pilot, lambda: not self.app.busy)
                self.assertEqual(reply.content, 'Hello from HariAI.')
                self.assertTrue(reply.note.display)
                self.assertIsInstance(reply.note.content, Text)
                self.assertEqual(reply.note.content.plain, 'Token likelihood: 83%')
                self.assertEqual(reply.note.content.style, DIM)
                self.assertEqual(self.app.generated_chars, len(reply.content))
                self.assertEqual(self.app.status.snapshot().turns, 1)
                self.assertEqual(self.app.status.snapshot().episodic_entries, 1)
                self.assertTrue(all(not button.disabled for button in self.app.query(Button)))
            finally:
                self.local.release.set()

    async def test_escape_stops_partial_reply_rolls_back_and_can_chat_again(self):
        self.local.release.clear()
        async with self.app.run_test(size=(160, 54)) as pilot:
            try:
                await self.app.submit_prompt('Explain a binary tree')
                await self.wait_for(pilot, lambda: self.replies()[-1].content == 'Hello ')
                reply = self.replies()[-1]
                await pilot.press('escape')
                self.local.release.set()
                await self.wait_for(pilot, lambda: not self.app.busy)
                self.assertTrue(self.local.closed.is_set())
                self.assertEqual(reply.content, 'Hello ')
                self.assertIn('Stopped', reply.note.content.plain)
                self.assertNotIn('83%', reply.note.content.plain)
                self.assertEqual(len(self.chat.history), 1)
                self.assertEqual(self.memory.working.scratchpad, [])
                self.assertEqual(self.memory.episodic.entries, [])
                await self.app.submit_prompt('Try this new question')
                await self.wait_for(pilot, lambda: not self.app.busy)
                self.assertEqual(self.replies()[-1].content, 'Hello from HariAI.')
                self.assertIn('83%', self.replies()[-1].note.content.plain)
            finally:
                self.local.release.set()

    async def test_model_error_recovers_on_next_request(self):
        self.local.failure = 'Model cannot be read'
        async with self.app.run_test(size=(160, 54)) as pilot:
            await self.app.submit_prompt('Explain gravity')
            await self.wait_for(pilot, lambda: not self.app.busy)
            reply = self.replies()[-1]
            self.assertIn('Model cannot be read', reply.content)
            self.assertTrue(self.app.last_error)
            self.assertFalse(reply.note.display)
            self.assertEqual(len(self.chat.history), 1)
            self.local.failure = ''
            await self.app.submit_prompt('Try gravity again')
            await self.wait_for(pilot, lambda: not self.app.busy)
            self.assertEqual(self.app.last_error, '')
            self.assertEqual(self.replies()[-1].content, 'Hello from HariAI.')
            self.assertEqual(self.app.status.snapshot().turns, 1)

    async def test_busy_session_rejects_clear_and_settings_mutation(self):
        self.local.release.clear()
        async with self.app.run_test(size=(160, 54)) as pilot:
            try:
                await self.app.submit_prompt('Explain sorting')
                await self.wait_for(pilot, lambda: self.replies()[-1].content == 'Hello ')
                old_engine = self.chat
                old_history = list(self.chat.history)
                old_turn_count = len(self.app.query(ChatTurn))
                await pilot.press('ctrl+l')
                await self.app.submit_prompt('/settings replace')
                self.assertIs(self.chat, old_engine)
                self.assertEqual(self.chat.history, old_history)
                self.assertEqual(len(self.app.query(ChatTurn)), old_turn_count)
                self.assertEqual(self.commands, [])
                self.assertTrue(self.app.busy)
                self.local.release.set()
                await self.wait_for(pilot, lambda: not self.app.busy)
            finally:
                self.local.release.set()

    async def test_quick_profile_and_commands_use_current_engine(self):
        async with self.app.run_test(size=(160, 70)) as pilot:
            button = self.app.query_one('#quick-info', Button)
            self.app.query_one('#sidebar').scroll_end(animate=False)
            await pilot.pause()
            button.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click('#quick-info')
            await self.wait_for(pilot, lambda: len(self.app.query(CreatorCard)) == 1)
            self.assertEqual(self.commands, [('/info', '')])
            self.assertIn('K R Hari Prajwal', self.replies()[-1].content)
            for command in ('/status', '/memory', '/dream', '/help'):
                await self.app.submit_prompt(command)
                await self.wait_for(pilot, lambda: not self.app.busy)
            self.assertEqual([name for name, _ in self.commands], ['/info', '/status', '/memory', '/dream', '/help'])
            old_engine = self.chat
            await self.app.submit_prompt('/settings replace')
            await self.wait_for(pilot, lambda: not self.app.busy)
            self.assertIsNot(self.chat, old_engine)
            self.assertEqual(self.app.status.snapshot().context_tokens, 512)
            await self.app.submit_prompt('Use the selected model')
            await self.wait_for(pilot, lambda: not self.app.busy)
            self.assertEqual(self.replies()[-1].content, 'Replacement model reply.')
            self.assertEqual(len(old_engine.history), 1)
            await pilot.press('ctrl+l')
            self.assertEqual(len(self.chat.history), 1)
            self.assertEqual(len(self.app.query(ChatTurn)), 1)
            self.assertEqual(len(self.app.query(CreatorCard)), 0)

    async def test_input_history_and_literal_generated_markup(self):
        self.local.pieces = ['[bold]literal[/bold] ', '\x1b[31mtext\x00']
        async with self.app.run_test(size=(160, 54)) as pilot:
            prompt = self.app.query_one('#prompt', Input)
            prompt.value = 'Explain markup'
            await pilot.press('enter')
            await self.wait_for(pilot, lambda: not self.app.busy)
            reply = self.replies()[-1]
            self.assertIsInstance(reply.body.content, Text)
            self.assertIn('[bold]literal[/bold]', reply.body.content.plain)
            self.assertNotIn('\x1b', reply.body.content.plain)
            self.assertNotIn('\x00', reply.body.content.plain)
            prompt.value = 'an unsent draft'
            await pilot.press('up')
            self.assertEqual(prompt.value, 'Explain markup')
            await pilot.press('down')
            self.assertEqual(prompt.value, 'an unsent draft')

    async def test_real_cli_command_adapter_stays_inside_dashboard(self):
        cli = importlib.import_module('hari_ai.cli.main')
        self.app.command_handler = cli.dashboard_command
        with patch.object(cli, '_CHAT_ENGINE', self.chat), \
                patch.object(cli, '_MEMORY_MANAGER', self.memory), \
                patch.object(cli, '_DREAM_CONTROLLER', self.dream):
            async with self.app.run_test(size=(160, 54)) as pilot:
                for command, expected in [('/status', 'HariAI status'),
                                          ('/memory status', 'episodic_entries_count'),
                                          ('/dream status', 'Dream Engine State'),
                                          ('/info', 'K R Hari Prajwal')]:
                    await self.app.submit_prompt(command)
                    try:
                        await self.wait_for(pilot, lambda: not self.app.busy)
                    except asyncio.TimeoutError:
                        self.fail(f'{command} hung; active={self.app.active_command!r}, '
                                  f'error={self.app.last_error!r}, '
                                  f'reply={self.replies()[-1].content[:160]!r}')
                    self.assertIn(expected, self.replies()[-1].content)
                    self.assertEqual(self.app.last_error, '')
                self.assertEqual(self.local.calls, [])
                self.assertEqual(len(self.chat.history), 1)

    async def test_silent_command_switches_mode(self):
        await self.app.submit_prompt('/silent')
        self.assertTrue(self.app.switch_to_silent)


if __name__ == '__main__':
    unittest.main()
