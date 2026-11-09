"""Live terminal dashboard. Inference stays on a serialized background worker."""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Footer, Input, Static

from hari_ai.cli.art import get_mahakal_text
from hari_ai.cli.confidence import confidence_lines
from hari_ai.profile.creator import get_creator_profile
from hari_ai.runtime.session_status import SessionStatus
from hari_ai.version import __version__

PURPLE = '#c17aff'
DIM = '#93889f'
GREEN = '#6de894'
AMBER = '#e9be75'

# Terminal control characters must never be executable output. Rich Text also
# prevents prompts and generated text from becoming styling or link markup.
_CONTROLS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')


def safe_text(value: str) -> str:
    return _CONTROLS.sub('', value).replace('\t', '    ')


def rows(*items: tuple[str, object]) -> Table:
    table = Table.grid(padding=(0, 1), expand=False)
    table.add_column(style=DIM, no_wrap=True)
    table.add_column(overflow='fold')
    for key, value in items:
        table.add_row(Text(key), Text(safe_text(str(value))))
    return table


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}h {minutes:02}m {seconds:02}s' if hours else f'{minutes}m {seconds:02}s'


class Brand(Static):
    def render(self):
        if self.app.size.width < 75 or self.app.size.height < 32:
            return Text.assemble(('HariAI', f'bold {PURPLE}'),
                                 (f'  v{__version__}\n', DIM),
                                 ('Local AI · Created by K R Hari Prajwal', DIM))
        logo = [
            '██╗  ██╗ █████╗ ██████╗ ██╗ █████╗ ██╗',
            '██║  ██║██╔══██╗██╔══██╗██║██╔══██╗██║',
            '███████║███████║██████╔╝██║███████║██║',
            '██╔══██║██╔══██║██╔══██╗██║██╔══██║██║',
            '██║  ██║██║  ██║██║  ██║██║██║  ██║██║',
            '╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝',
        ]
        text = Text(justify='center')
        for line, color in zip(logo, ['#d7adff', '#c68cff', '#b56cf4', '#a25ce0', '#9150c9', '#7d43b1']):
            text.append(line + '\n', color)
        text.append(f'HariAI v{__version__}  ·  Local AI, your terminal\n', 'bold')
        text.append('Chat  |  Code  |  Memory  |  Research\n', DIM)
        text.append('Created by K R Hari Prajwal', DIM)
        return text


class Panel(Static):
    def __init__(self, title: str, *, id: str):
        super().__init__(id=id, classes='panel', markup=False)
        self.border_title = title


class ChatTurn(Vertical):
    def __init__(self, role: str, content: str = ''):
        super().__init__(classes=f'turn {role}-turn')
        self.role = role
        self.content = content
        self.body = Static(self._body_text(), classes='reply-body', markup=False)
        self.note = Static('', classes='reply-note', markup=False)
        self.note.display = False

    def compose(self) -> ComposeResult:
        if self.role == 'user':
            yield Static(Text.assemble(('you > ', 'bold'), (safe_text(self.content), '')), classes='user-line')
        else:
            yield self.body
            yield self.note

    def _body_text(self) -> Text:
        return Text.assemble(('HariAI ✦ :  ', f'bold {PURPLE}'), (safe_text(self.content), ''))

    def append(self, piece: str) -> None:
        self.content += piece
        self.body.update(self._body_text())

    def annotate(self, lines: list[str]) -> None:
        self.note.display = bool(lines)
        self.note.update(Text('\n'.join(lines), style=DIM))


class CreatorCard(Static):
    def __init__(self):
        super().__init__(classes='panel creator-card', markup=False)
        self.border_title = '✦ Creator Profile'

    def render(self):
        profile = get_creator_profile()
        info = rows(('Name', profile.name), ('Origin', profile.origin_region),
                    ('At creation', f'{profile.creation_age} years old'),
                    ('About', profile.professional_identity),
                    ('Inspiration', profile.spiritual_inspiration),
                    ('Motto', profile.personal_motto), ('GitHub', profile.github_url))
        mahakal_art = get_mahakal_text(width=16, height_lines=8, bg_hex='#0c0912')
        card = Table.grid(padding=(0, 2), expand=False)
        card.add_column()
        card.add_column()
        card.add_row(mahakal_art, info)
        return card


class ReplyPiece(Message):
    def __init__(self, text: str):
        super().__init__()
        self.text = text


class TaskFinished(Message):
    def __init__(self, *, cancelled: bool = False, error: str = '', notes: list[str] | None = None,
                 creator: bool = False):
        super().__init__()
        self.cancelled, self.error = cancelled, error
        self.notes, self.creator = notes or [], creator


class HariDashboard(App):
    """Default interactive UI; takes live engine and command adapters from CLI."""

    TITLE = 'HariAI Terminal'
    CSS_PATH = 'dashboard.tcss'
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding('ctrl+q', 'quit_session', 'Quit', priority=True),
        Binding('ctrl+c', 'stop_or_quit', 'Stop / quit', show=False, priority=True),
        Binding('escape', 'stop', 'Stop', priority=True),
        Binding('ctrl+l', 'clear_chat', 'New chat', priority=True),
        Binding('f2', 'toggle_panels', 'Panels', priority=True),
        Binding('f1', 'help', 'Help', priority=True),
    ]

    def __init__(self, engine_getter: Callable, dream_controller, command_handler: Callable, swarm=None):
        super().__init__()
        self.engine_getter = engine_getter
        self.dream = dream_controller
        self.command_handler = command_handler
        self.swarm = swarm
        self.switch_to_silent = False
        self.status = SessionStatus(engine_getter, dream_controller)
        self.busy = False
        self.task_is_chat = False
        self.activity = 'Ready'
        self.task_started = 0.0
        self.last_elapsed = 0.0
        self.generated_chars = 0
        self.stop_requested = threading.Event()
        self.exit_requested = False
        self.active_reply: ChatTurn | None = None
        self.panels_override: bool | None = None
        self.last_error = ''
        self._input_history: list[str] = []
        self._history_index = 0
        self._draft = ''

    def compose(self) -> ComposeResult:
        with Horizontal(id='titlebar'):
            yield Static(Text.assemble(('● ', '#ff6a80'), ('● ', '#f3c762'), ('●', '#65d892')), id='lights')
            yield Static('hari — HariAI Terminal', id='window-title', markup=False)
            yield Static(f'v{__version__}', id='version', markup=False)
        with Horizontal(id='workspace'):
            with Vertical(id='main-pane'):
                with VerticalScroll(id='conversation'):
                    yield Brand(id='brand')
                    yield Panel('✦ HariAI', id='overview')
                    with Vertical(id='messages'):
                        yield ChatTurn('assistant', 'How can I help you today?')
                yield Static('Ready · /help for commands', id='activity', markup=False)
                yield Input(placeholder='Send a message  ·  /help for commands', id='prompt')
            with VerticalScroll(id='sidebar'):
                yield Panel('◈ HariAI Status', id='runtime-status')
                yield Panel('✧ Capabilities', id='capabilities')
                yield Panel('◉ Learning · Dream Engine', id='learning')
                yield Panel('⌘ Cluster · Multi-HariAI', id='cluster')
                with Vertical(id='quick-commands', classes='panel'):
                    yield Static('Quick Commands', classes='panel-heading')
                    yield Button('/status     Live system details', id='quick-status')
                    yield Button('/swarm      Multi-bot debate', id='quick-swarm')
                    yield Button('/silent     Minimal terminal', id='quick-silent')
                    yield Button('/dream      Sleep consolidation', id='quick-dream')
                    yield Button('/memory     Session memory', id='quick-memory')
                    yield Button('/info       Creator profile', id='quick-info')
                    yield Button('/settings   Configure model', id='quick-settings')
                    yield Button('/clear      Start a new chat', id='quick-clear')
                    yield Button('/help       All commands', id='quick-help')
        yield Footer()

    def on_mount(self) -> None:
        self.theme = 'textual-dark'
        self.query_one('#quick-commands').border_title = '⌁ Quick Commands'
        self.refresh_panels()
        self.set_interval(1, self.refresh_panels)
        self._apply_layout()
        self.query_one('#prompt', Input).focus()

    def on_resize(self) -> None:
        if self.query('#sidebar'):
            self.call_after_refresh(self._apply_layout)
            self.query_one('#brand').refresh(layout=True)

    def _apply_layout(self) -> None:
        narrow = self.size.width < 112
        visible = self.panels_override if self.panels_override is not None else not narrow
        self.query_one('#sidebar').display = visible
        self.query_one('#main-pane').display = not (narrow and visible)
        self.set_class(narrow, 'narrow')
        self.set_class(self.size.height < 32, 'short')

    def action_toggle_panels(self) -> None:
        self.panels_override = not self.query_one('#sidebar').display
        self._apply_layout()
        if self.query_one('#main-pane').display:
            self.query_one('#prompt', Input).focus()

    def refresh_panels(self) -> None:
        # A queued timer may arrive after screen descendants have unmounted.
        if not self.query('#runtime-status'):
            return
        state = self.status.snapshot()
        elapsed = time.monotonic() - self.task_started if self.busy else self.last_elapsed
        if self.last_error:
            readiness = 'Needs attention'
        elif self.busy:
            readiness = self.activity
        else:
            readiness = state.readiness
        size = f'{state.model_bytes / 1_000_000:.1f} MB' if state.model_bytes is not None else '—'
        ram = f'{state.process_mb:.0f} MB' if state.process_mb is not None else 'unavailable'
        free = f'{state.available_ram_gb:.1f} GB' if state.available_ram_gb is not None else 'unavailable'
        cpu = f'{state.cpu_percent:.0f}%' if state.cpu_percent is not None else '—'
        runtime = self.query_one('#runtime-status', Panel)
        runtime.border_subtitle = f'● {readiness}'
        runtime.styles.border_subtitle_color = AMBER if self.last_error or not state.model_available else GREEN
        runtime.update(rows(('Model', state.model_name), ('Size', size), ('Mode', f'Chat · {state.backend}'),
                            ('Device', state.device), ('Uptime', duration(state.uptime_seconds))))
        overview = self.query_one('#overview', Panel)
        overview.border_title = f'✦ HariAI · {readiness}'
        overview.update(rows(('Model', f'{state.model_name} · {size}'),
                             ('Runtime', f'{state.device}' + (f' · {state.threads} threads' if state.threads else '')),
                             ('Context', f'{state.context_tokens:,} tokens' if state.context_tokens else 'Backend managed'),
                             ('Memory', f'{state.turns} turns · {state.episodic_entries} session experiences'),
                             ('Learning', 'Manual inspection · no background training'),
                             ('Cluster', 'Single session · no workers started')))
        capabilities = Text()
        for label, enabled in [('Conversation & explanations', state.model_available),
                               ('Code generation (unverified)', state.model_available),
                               ('Explicit local memory', True), ('Web search via /search', True),
                               ('Automatic training', False), ('Autonomous tool execution', False)]:
            capabilities.append('✓ ' if enabled else '○ ', GREEN if enabled else DIM)
            capabilities.append(label + '\n', '' if enabled else DIM)
        capabilities.append(f'Context: {state.turns} turns · {state.working_items} memory items\n', DIM)
        capabilities.append(f'App RAM: {ram} · Free: {free}\nSystem CPU: {cpu}', DIM)
        self.query_one('#capabilities', Panel).update(capabilities)
        auto_dream = getattr(self.dream, 'auto_dream_active', False)
        learning_state = 'Consolidating' if state.dream_state == 'sleep' else ('Active · auto' if auto_dream else 'Idle · manual')
        rehearsal_count = len(getattr(self.dream, 'synthetic_rehearsal_buffer', []))
        self.query_one('#learning', Panel).update(rows(
            ('State', learning_state),
            ('Cycles', state.dream_cycles), ('Experiences', state.episodic_entries),
            ('Facts', f'{state.semantic_facts} · Recipes: {state.recipes}'),
            ('Rehearsal', f'{rehearsal_count} exercises'),
            ('Weights', 'Continual LoRA gate')))
        swarm_status = self.swarm.get_status() if self.swarm else {
            'session': 'This terminal', 'workers': '0 · not started',
            'activity': self.activity if self.busy else 'Idle', 'mode': 'Standalone chat'
        }
        cluster_rows = rows(
            ('Session', swarm_status['session']), ('Workers', swarm_status['workers']),
            ('Activity', swarm_status['activity']), ('Mode', swarm_status['mode']))
        mahakal_cluster = get_mahakal_text(width=10, height_lines=4, bg_hex='#0c0912')
        cluster_grid = Table.grid(padding=(0, 1), expand=False)
        cluster_grid.add_column()
        cluster_grid.add_column()
        cluster_grid.add_row(cluster_rows, mahakal_cluster)
        self.query_one('#cluster', Panel).update(cluster_grid)
        if self.busy:
            suffix = 'Esc to stop' if self.task_is_chat else 'Command in progress'
            activity = f'{self.activity} · {elapsed:.1f}s · {self.generated_chars} characters · {suffix}'
        elif self.last_error:
            activity = 'Needs attention · see the last message · /settings to configure'
        elif self.last_elapsed:
            task_label = 'reply' if self.task_is_chat else 'command'
            activity = f'Ready · last {task_label} {elapsed:.1f}s · {self.generated_chars} characters'
        else:
            activity = 'Ready · /help for commands'
        self.query_one('#activity', Static).update(Text(activity, style=DIM))

    async def _add_turn(self, role: str, text: str = '') -> ChatTurn:
        turn = ChatTurn(role, text)
        await self.query_one('#messages', Vertical).mount(turn)
        self.query_one('#conversation', VerticalScroll).scroll_end(animate=False)
        return turn

    @on(Input.Submitted, '#prompt')
    async def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.busy:
            self.notify('A reply is still running. Press Esc to stop it first.')
            return
        event.input.value = ''
        self._input_history.append(text)
        self._history_index = len(self._input_history)
        self._draft = ''
        await self.submit_prompt(text)

    async def on_key(self, event) -> None:
        prompt = self.query_one('#prompt', Input)
        if prompt.has_focus and event.key in ('up', 'down') and self._input_history:
            event.prevent_default()
            event.stop()
            if self._history_index == len(self._input_history):
                self._draft = prompt.value
            self._history_index = max(0, min(len(self._input_history), self._history_index + (-1 if event.key == 'up' else 1)))
            prompt.value = self._draft if self._history_index == len(self._input_history) else self._input_history[self._history_index]
            prompt.cursor_position = len(prompt.value)

    @on(Button.Pressed)
    async def quick_command(self, event: Button.Pressed) -> None:
        if self.busy:
            self.notify('Wait for the current reply or press Esc to stop it.')
            return
        command = '/' + event.button.id.removeprefix('quick-')
        if self.size.width < 112:
            self.panels_override = False
            self._apply_layout()
        await self.submit_prompt(command)
        self.query_one('#prompt', Input).focus()

    async def submit_prompt(self, text: str) -> None:
        if self.busy:
            return
        if text.lower() in ('/exit', '/quit', 'exit', 'quit', ':q'):
            self.action_quit_session()
            return
        if text.lower() in ('/silent', '/plain', '/terminal', 'silent'):
            self.switch_to_silent = True
            if self.is_running:
                self.exit()
            return
        if text.lower() in ('/clear', 'clear'):
            await self.action_clear_chat()
            return
        self.dream.notify_user_activity()
        await self._add_turn('user', text)
        self.active_reply = await self._add_turn('assistant')
        self.busy = True
        self.last_error = ''
        self.generated_chars = 0
        self.task_started = time.monotonic()
        self.stop_requested.clear()
        self.activity = 'Working' if text.startswith('/') else 'Generating'
        self.task_is_chat = not text.startswith('/') or text.partition(' ')[0].lower() in ('/ask', '/code', '/think')
        for button in self.query(Button):
            button.disabled = True
        self.refresh_panels()
        self._run_request(text)

    @work(thread=True, exclusive=True, group='conversation')
    def _run_request(self, text: str) -> None:
        notes: list[str] = []
        creator = False
        error = ''
        try:
            command, _, argument = text.partition(' ')
            command = command.lower()
            chat_commands = {'/ask': '', '/code': 'Write code for the following request. State any assumptions: ',
                             '/think': 'Please reason carefully about: '}
            if text.startswith('/') and command not in chat_commands:
                result = self.command_handler(command, argument)
                self.post_message(ReplyPiece(result))
                creator = command in ('/info', '/mahakal', '/shiva')
            else:
                if command in chat_commands and not argument.strip():
                    raise ValueError(f'Usage: {command} <message>')
                prompt = chat_commands.get(command, '') + argument if command in chat_commands else text
                engine = self.engine_getter()
                if hasattr(engine.local_model, 'preserve_terminal'):
                    engine.local_model.preserve_terminal = True
                stream = engine.chat_stream(prompt)
                try:
                    for piece in stream:
                        if self.stop_requested.is_set():
                            break
                        self.post_message(ReplyPiece(piece))
                finally:
                    stream.close()
                if not self.stop_requested.is_set():
                    notes = confidence_lines(engine)
                    if not engine.last_reply_succeeded:
                        error = 'The backend could not complete this reply.'
                    elif engine.settings.backend == 'local' and engine.local_model.last_finish_reason == 'length':
                        notes.append('Response reached the token limit and may be incomplete.')
                    creator = bool(re.fullmatch(r'who (?:made|created|built) (?:you|u)[\s?!\.]*', prompt.lower()))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        self.post_message(TaskFinished(cancelled=self.stop_requested.is_set(), error=error,
                                       notes=notes, creator=creator))

    def on_reply_piece(self, event: ReplyPiece) -> None:
        if self.active_reply is None:
            return
        viewport = self.query_one('#conversation', VerticalScroll)
        follow = viewport.is_vertical_scroll_end
        self.active_reply.append(event.text)
        self.generated_chars += len(event.text)
        if follow:
            viewport.scroll_end(animate=False)

    async def on_task_finished(self, event: TaskFinished) -> None:
        if not self.query('#messages'):
            return
        viewport = self.query_one('#conversation', VerticalScroll)
        follow = viewport.is_vertical_scroll_end
        self.last_elapsed = time.monotonic() - self.task_started
        self.busy = False
        self.activity = 'Ready'
        self.last_error = event.error
        if self.active_reply:
            if event.error:
                self.active_reply.append(('\n\n' if self.active_reply.content else '') + safe_text(event.error))
            self.active_reply.annotate(['Stopped · partial output; no confidence assigned.'] if event.cancelled else event.notes)
            if not self.active_reply.content and event.cancelled:
                self.active_reply.append('Request stopped.')
        if event.creator and not event.cancelled and not event.error:
            await self.query_one('#messages', Vertical).mount(CreatorCard())
        if follow:
            viewport.scroll_end(animate=False)
        self.active_reply = None
        for button in self.query(Button):
            button.disabled = False
        self.refresh_panels()
        self.query_one('#prompt', Input).focus()
        if self.exit_requested:
            self.exit()

    async def action_clear_chat(self) -> None:
        if self.busy:
            self.notify('Stop the current reply before starting a new chat.')
            return
        self.engine_getter().clear_history()
        await self.query_one('#messages', Vertical).remove_children()
        await self._add_turn('assistant', 'New conversation. How can I help you?')
        self.last_elapsed = 0.0
        self.generated_chars = 0
        self.last_error = ''
        self.refresh_panels()

    async def action_help(self) -> None:
        if not self.busy:
            await self.submit_prompt('/help')

    def action_stop(self) -> None:
        if self.busy:
            if not self.task_is_chat:
                self.notify('This command is running; its result will appear when it finishes.')
                return
            self.stop_requested.set()
            self.activity = 'Stopping'
            self.refresh_panels()

    def action_quit_session(self) -> None:
        if self.busy:
            self.exit_requested = True
            self.action_stop()
        else:
            self.exit()

    def action_stop_or_quit(self) -> None:
        if self.busy:
            self.action_stop()
        else:
            self.action_quit_session()
