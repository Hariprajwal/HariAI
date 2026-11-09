"""HariAI Terminal-First Command Line Interface (CLI).

Provides the standalone interactive experience for HariAI:
- Full interactive REPL with streaming responses
- Dedicated subcommands: ask, think, decide, code, memory, dream, info, version
- Dynamic creator identity resolution
- Open-JEV typed decision evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

from hari_ai.dream.controller import DreamEngineController, SleepTrigger
from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager
from hari_ai.profile.creator import get_creator_profile
from hari_ai.runtime.chat import ChatEngine
from hari_ai.search.engine import SearchEngine
from hari_ai.memory.store import MemoryStore
from hari_ai.runtime.settings import load_settings, save_settings, settings_path
from hari_ai.version import __version__
from hari_ai.cli.confidence import confidence_lines


BANNER = r"""
     _   _            _    ___  
    | | | | __ _ _ __| |  / _ \ 
    | |_| |/ _` | '__| | | | | |
    |  _  | (_| | |  | | | |_| |
    |_| |_|\__,_|_|  |_|  \___/ 

        Local conversation.
        Created by K R Hari Prajwal.

""" + f"HariAI v{__version__}\n"

# Global memory, dream controller, search engine, and chat engine singletons for local session
_MEMORY_MANAGER = HierarchicalMemoryManager()
_DREAM_CONTROLLER = DreamEngineController(memory_manager=_MEMORY_MANAGER)
_SEARCH_ENGINE = SearchEngine(memory_manager=_MEMORY_MANAGER)
_CHAT_ENGINE = ChatEngine(memory_manager=_MEMORY_MANAGER)


def print_banner() -> None:
    print(BANNER)


def get_user_input() -> str:
    """Reads user input with a gray placeholder that vanishes on typing (like Ollama)."""
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.formatted_text import HTML

        placeholder = HTML('<style color="#777777">Send a message (/? for help)</style>')
        return prompt("hari> ", placeholder=placeholder).strip()
    except Exception:
        return input("hari> ").strip()


def cmd_info(*, emit=print) -> None:
    """Displays HariAI creator profile and identity details."""
    profile = get_creator_profile()

    emit("\n" + "=" * 60)
    emit(" HARIAI CREATOR PROFILE")
    emit("=" * 60)
    emit(f" Creator:        {profile.name}")
    emit(f" Origin:         {profile.origin_region}")
    emit(f" Creation Age:   {profile.creation_age} years old (At creation)")
    emit(f" Statement:      \"{profile.official_statement}\"")
    emit(f" GitHub:         {profile.github_url}")
    from hari_ai.runtime.session_status import SessionStatus
    state = SessionStatus(lambda: _CHAT_ENGINE, _DREAM_CONTROLLER).snapshot()
    emit(f" Chat Model:     {state.model_name} ({state.backend})")
    if state.model_origin:
        emit(f" Base model:     {state.model_origin}")
    if state.model_bytes is not None:
        emit(f" Model size:     {state.model_bytes / 1_000_000:.1f} MB")
    emit(" Research goal:  Under 35 MB (not achieved by this chat model)")
    emit("=" * 60 + "\n")


def cmd_version(*, emit=print) -> None:
    emit(f"HariAI version {__version__} (standalone local GGUF)")


def cmd_status(*, emit=print) -> None:
    """Show observable runtime state without implying experimental features are active."""
    settings = _CHAT_ENGINE.settings
    emit('\nHariAI status')
    emit(f'  Backend:     {settings.backend}')
    if settings.backend == 'local':
        from hari_ai.runtime.local_model import bundled_model_path
        path = _CHAT_ENGINE.local_model.path
        name = 'HariAI Chat (SmolLM-derived)' if path.resolve() == bundled_model_path().resolve() else 'Custom local model'
        emit(f'  Model:       {name}')
        if path.is_file():
            emit(f'  Model file:  {path} ({path.stat().st_size / 1_000_000:.1f} MB)')
        else:
            emit(f'  Model file:  missing ({path})')
        emit('  Inference:   CPU; no Ollama service required')
    elif settings.backend == 'nano':
        emit(f'  Checkpoint:  {settings.nano_checkpoint or "not configured"}')
    else:
        emit(f'  Model:       {settings.ollama_model or "not configured"}')
        emit('  Inference:   configured Ollama service')
    emit('  Confidence:  local token likelihood; answer accuracy is not calibrated')
    emit('  Learning:    research mode; no automatic model updates')
    emit('  Cluster:     prototype; chat has not started workers')
    emit('')


def cmd_search(query: str, *, emit=print) -> None:
    """Executes live web/local search and AI summarization."""
    emit(f"\n[HariNano Search Engine]: Gathering and synthesizing data for '{query}'...")
    summary = _SEARCH_ENGINE.search_and_summarize(query)
    emit("\n" + summary + "\n")


def _show_confidence_status() -> None:
    for message in confidence_lines(_CHAT_ENGINE):
        if sys.stdout.isatty() and 'NO_COLOR' not in os.environ:
            print(f'\033[90m{message}\033[0m')
        else:
            print(message)


def cmd_ask(prompt: str, stream: bool = False) -> None:
    """Conversational language query powered by the ChatEngine."""
    if stream:
        for chunk in _CHAT_ENGINE.chat_stream(prompt):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        response = "".join(list(_CHAT_ENGINE.chat_stream(prompt)))
        print(response)
    if (_CHAT_ENGINE.last_reply_succeeded and _CHAT_ENGINE.settings.backend == 'local'
            and getattr(_CHAT_ENGINE.local_model, 'last_finish_reason', None) == 'length'):
        print('Response reached the token limit and may be incomplete.')
    _show_confidence_status()


def cmd_think(prompt: str, llm: Optional[Any] = None) -> str:
    """Request careful reasoning; model output is not independently verified."""
    if llm is None:
        answer = ''.join(_CHAT_ENGINE.chat_stream('Please reason carefully about: ' + prompt))
    else:
        answer = llm.generate('Please reason carefully about: ' + prompt).strip()
    print(answer or 'The model produced no answer.')
    if llm is None:
        _show_confidence_status()
    return answer


def cmd_decide(situation: str, *, emit=print) -> None:
    """Open-JEV style typed decision readout: Choice, Score, Noul."""
    emit("\n" + "=" * 60)
    emit(" HARIAI TYPED DECISION ENGINE (System One Jev-Style)")
    emit("=" * 60)
    emit(f" Situation:                {situation}")

    settings = load_settings()
    if settings.openjev_checkpoint:
        from hari_ai.runtime.openjev import OpenJevAdapter
        try:
            decision = OpenJevAdapter(settings.openjev_checkpoint).decide(
                {'situation': situation}, 'What should be done?',
                ['review_situation', 'request_human_confirmation', 'stop'])
            emit(f" Choice:                   {decision['choice']}")
            emit(f" Score (raw, uncalibrated): {decision['confidence_raw']:.3f}")
            emit(" Noul (Permission Gate):   REQUIRES APPROVAL")
            emit(f" Open-JEV weights:          {decision['checkpoint_bytes']} bytes (separate from HariNano)")
            emit("=" * 60 + "\n")
            return
        except (OSError, ValueError, RuntimeError) as exc:
            emit(f" Open-JEV unavailable: {exc}")

    # Evaluate decision logic
    is_critical = any(
        w in situation.lower()
        for w in ["delete", "delet", "remove", "drop", "terminate", "format", "erase"]
    )
    selected_choice = "request_human_confirmation" if is_critical else "review_situation"
    noul_permission = True

    emit(f" Choice (Selected Action): {selected_choice}")
    emit(" Score (Confidence):       unavailable (not calibrated)")
    emit(f" Noul (Permission Gate):   {'REQUIRES APPROVAL' if noul_permission else 'AUTO-APPROVED'}")
    emit("=" * 60 + "\n")


def cmd_code(task: str) -> None:
    """Request code from the configured chat backend without claiming verification."""
    cmd_ask("Write code for the following request. State any assumptions: " + task)


def cmd_operation(value: str, *, emit=print) -> None:
    """Evaluate one bounded JSON record with the bundled Open-JEV model."""
    from hari_ai.runtime.openjev import OperationDecisionModel
    try:
        result = OperationDecisionModel().decide(json.loads(value))
        emit(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        emit(f'Operation decision unavailable: {exc}')


def cmd_memory(args: List[str], *, emit=print) -> None:
    sub = args[0] if args else "status"
    if sub == "status":
        stats = _MEMORY_MANAGER.get_status_summary()
        emit("\n--- HariAI Hierarchical Memory Status ---")
        for k, v in stats.items():
            emit(f" {k:30s}: {v}")
        emit("-----------------------------------------\n")
    elif sub == "clear":
        _MEMORY_MANAGER.working.clear()
        emit("Working memory cleared.")
    elif sub == "add":
        text = " ".join(args[1:]).strip()
        if not text:
            emit('Usage: hari memory add "Text to remember"')
            return
        store = MemoryStore()
        emit(f"Saved local memory {store.add(text)} at {store.path}. It is not used for model training.")
    elif sub == "inspect":
        memories = MemoryStore().list()
        if not memories:
            emit("No saved local memories.")
        for entry in memories:
            emit(f"{entry['id']}: {entry['content']}")
    elif sub == "delete":
        if len(args) != 2 or not args[1].isdigit():
            emit("Usage: hari memory delete <id>")
            return
        emit("Memory deleted." if MemoryStore().delete(int(args[1])) else "Memory not found.")
    else:
        emit('Usage: hari memory status|clear|add <text>|inspect|delete <id>')


def cmd_settings(args: List[str], *, emit=print) -> None:
    """Show or update chat settings; changing backends is always explicit."""
    global _CHAT_ENGINE
    settings = load_settings()
    if not args or args[0] == 'show':
        emit(f'Settings: {settings_path()}')
        emit(json.dumps(asdict(settings), indent=2))
        return
    if args[0] == 'list-models':
        models = _CHAT_ENGINE.ollama.list_models()
        emit('\n'.join(models) if models else 'No Ollama models found; check that Ollama is running.')
        return
    if args[0] == 'confidence-status':
        emit('Local model replies show average base-model token likelihood, not answer accuracy. '
              'Exact verification is available for complete single-turn integer arithmetic answers. '
              'General answer correctness is not calibrated.')
        return
    if args[0] == 'confidence-artifact':
        emit('Open-JEV confidence artifacts are no longer used by chat; '
              'exact arithmetic verification is built in.')
        return
    if args[0] == 'confidence':
        if len(args) != 2 or args[1] not in ('on', 'off'):
            emit('Usage: hari settings confidence on|off')
            return
        settings.show_confidence = args[1] == 'on'
        try:
            save_settings(settings)
        except (ValueError, OSError) as exc:
            emit(f'Settings not changed: {exc}')
            return
        _CHAT_ENGINE.settings.show_confidence = settings.show_confidence
        emit(f'Confidence status {args[1]}.')
        return
    if args[0] == 'nano-model':
        if len(args) != 3:
            emit('Usage: hari settings nano-model CHECKPOINT TOKENIZER')
            return
        checkpoint, tokenizer = (Path(value).expanduser().resolve() for value in args[1:])
        if not checkpoint.is_file() or not tokenizer.is_file():
            emit('Checkpoint and tokenizer files must both exist.')
            return
        settings.nano_checkpoint, settings.nano_tokenizer = str(checkpoint), str(tokenizer)
        try:
            new_chat = ChatEngine(memory_manager=_MEMORY_MANAGER, settings=settings)
            save_settings(settings)
        except (ValueError, OSError, RuntimeError) as exc:
            emit(f'Settings not changed: {exc}')
            return
        _CHAT_ENGINE = new_chat
        emit(f'Set Nano model to {checkpoint}')
        return
    mapping = {'backend': 'backend', 'ollama-model': 'ollama_model', 'ollama-host': 'ollama_host',
               'nano-checkpoint': 'nano_checkpoint', 'nano-tokenizer': 'nano_tokenizer',
               'openjev-checkpoint': 'openjev_checkpoint', 'model-file': 'model_file'}
    if args[0] not in mapping or len(args) < 2:
        emit('Usage: hari settings show | confidence-status | confidence on|off | list-models | backend local|nano|ollama | model-file PATH.gguf | ollama-model NAME | ollama-host URL | nano-model CHECKPOINT TOKENIZER | openjev-checkpoint PATH')
        return
    field = mapping[args[0]]
    value = ' '.join(args[1:])
    if field.endswith('checkpoint') or field in ('nano_tokenizer', 'model_file'):
        value = str(Path(value).expanduser().resolve())
        if not Path(value).is_file():
            emit(f'File not found: {value}')
            return
    setattr(settings, field, value)
    try:
        settings.validate()
        new_chat = ChatEngine(memory_manager=_MEMORY_MANAGER, settings=settings)
        save_settings(settings)
    except (ValueError, OSError, RuntimeError) as exc:
        emit(f'Settings not changed: {exc}')
        return
    _CHAT_ENGINE = new_chat
    emit(f'Set {args[0]} to {value}')


def cmd_dream(args: List[str], *, emit=print) -> None:
    sub = args[0] if args else "now"
    if sub == "now":
        emit("\nHariAI Dream Engine")
        emit("Entering controlled sleep mode...\n")
        result = _DREAM_CONTROLLER.run_sleep_cycle(trigger=SleepTrigger.MANUAL)
        emit("Analyzing recent experiences.")
        emit(f"Inspected {result.experiences_inspected} experiences.")
        emit(f"Generated {result.synthetic_exercises_generated} synthetic practice exercises.")
        emit("Status: inspection only; no training or checkpoint evaluation was run.")
        emit("\nDream cycle completed.")
        emit("No production model changes were made without passing evaluation.\n")
    elif sub == "status":
        emit(f"Dream Engine State: {_DREAM_CONTROLLER.current_state.value.upper()}")
        emit(f"Total Sleep Cycles: {len(_DREAM_CONTROLLER.sleep_cycle_history)}")
    else:
        emit('Usage: hari dream now|status (inspection only; no training)')


def repl() -> None:
    """Interactive conversational REPL with Ollama-style vanishing placeholder."""
    print_banner()
    cmd_status()

    while True:
        try:
            user_input = get_user_input()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        _DREAM_CONTROLLER.notify_user_activity()

        lower = user_input.lower()
        if lower in ["exit", "quit", "/exit", "/quit", ":q", "bye"]:
            sys.exit(0)
        elif lower in ["clear", "/clear"]:
            _CHAT_ENGINE.clear_history()
            print("Conversation history reset. Ready for a new topic!\n")
        elif lower in ["info", "/info"]:
            cmd_info()
        elif lower in ["version", "/version"]:
            cmd_version()
        elif lower in ["status", "/status"]:
            cmd_status()
        elif lower in ["/?", "/help", "help", "hari help"]:
            print("\nAvailable Commands:")
            print("  /search <query>     - Web/Local search & AI research summarization")
            print("  /think <problem>    - Deliberate reasoning mode (<|thought|>)")
            print("  /decide <situation> - Open-JEV typed single-forward-pass decision evaluation")
            print("  /code <task>        - Request code from the local model")
            print("  /dream [now|status] - Inspect Dream research status")
            print("  /memory [status]    - Inspect hierarchical memory levels")
            print("  /settings [show|backend|ollama-model|list-models|confidence] - Configure chat")
            print("  /clear              - Reset multi-turn conversation context")
            print("  /info               - Display Creator Profile (K R Hari Prajwal)")
            print("  /version            - Show the running app version")
            print("  /status             - Show actual runtime and model status")
            print("  /exit               - Exit the chat session\n")
        elif lower.startswith("/search ") or lower.startswith("search "):
            q = user_input.split(" ", 1)[1]
            cmd_search(q)
        elif lower.startswith("/think ") or lower.startswith("think "):
            p = user_input.split(" ", 1)[1]
            cmd_think(p)
        elif lower.startswith("/decide ") or lower.startswith("decide "):
            p = user_input.split(" ", 1)[1]
            cmd_decide(p)
        elif lower.startswith("/code ") or lower.startswith("code "):
            p = user_input.split(" ", 1)[1]
            cmd_code(p)
        elif lower.startswith("/dream") or lower.startswith("dream"):
            parts = user_input.split()
            cmd_dream(parts[1:] if len(parts) > 1 else [])
        elif lower.startswith("/memory") or lower.startswith("memory"):
            parts = user_input.split()
            cmd_memory(parts[1:] if len(parts) > 1 else [])
        elif lower.startswith('/settings') or lower.startswith('settings'):
            parts = user_input.split()
            cmd_settings(parts[1:])
        elif lower.startswith("/ask ") or lower.startswith("ask "):
            p = user_input.split(" ", 1)[1]
            cmd_ask(p, stream=True)
            print()
        else:
            # Full streaming conversational dialogue
            cmd_ask(user_input, stream=True)
            print()


def dashboard_command(command: str, argument: str) -> str:
    """Capture command output explicitly, without redirecting the UI's stdout."""
    lines = []
    if command in ('/help', '/?'):
        return ('/ask <message>      Chat with the selected model\n'
                '/code <task>        Request code\n'
                '/think <question>   Request careful reasoning\n'
                '/search <query>     Search the web\n'
                '/status            Runtime and model details\n'
                '/info              Creator profile\n'
                '/version           Installed app version\n'
                '/memory status     Inspect session memory\n'
                '/memory add <text>  Save a local memory explicitly\n'
                '/memory inspect    Read saved memories\n'
                '/memory delete <id> Delete a saved memory\n'
                '/dream now         Inspect experiences (no training)\n'
                '/settings show     Show model configuration\n'
                '/settings backend local|nano|ollama\n'
                '/settings model-file "C:\\path\\model.gguf"\n'
                '/settings ollama-model <name>\n'
                '/settings confidence on|off\n'
                '/decide <situation> Experimental advisory decision\n'
                '/clear             Start a new conversation\n'
                '/exit              Exit HariAI\n\n'
                'Enter: send · Up/Down: input history · Esc: stop reply\n'
                'F2: show/hide panels · Ctrl+L: new chat · Ctrl+Q: quit')
    no_args = {'/status': cmd_status, '/info': cmd_info, '/version': cmd_version}
    text_args = {'/search': cmd_search, '/decide': cmd_decide, '/check-operation': cmd_operation}
    list_args = {'/settings': cmd_settings, '/memory': cmd_memory, '/dream': cmd_dream}
    if command in no_args:
        no_args[command](emit=lines.append)
    elif command in text_args:
        text_args[command](argument, emit=lines.append)
    elif command in list_args:
        parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in '\"\'' else part
                 for part in shlex.split(argument, posix=False)]
        list_args[command](parts, emit=lines.append)
    else:
        return f'Unknown command: {command}. Enter /help for available commands.'
    return '\n'.join(lines).strip()


def launch_chat(*, plain: bool = False) -> None:
    """Use the dashboard in a real terminal and keep redirected IO line-based."""
    if plain or not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get('TERM') == 'dumb':
        repl()
        return
    try:
        from hari_ai.cli.dashboard import HariDashboard
    except ImportError:
        print('Dashboard dependency missing. Run `python -m pip install -e .` to enable it.')
        repl()
        return
    HariDashboard(lambda: _CHAT_ENGINE, _DREAM_CONTROLLER, dashboard_command).run()


def main():
    # Redirected Windows terminals can use legacy encodings that lack emoji.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description="HariAI Command Line Interface", add_help=False)
    parser.add_argument("command", nargs="?", default="run", help="Subcommand to execute")
    parser.add_argument("args", nargs="*", help="Arguments for subcommand")
    parser.add_argument('--plain', action='store_true', help='Use the line-based terminal interface')

    parsed, unknown = parser.parse_known_args()
    cmd = parsed.command.lower()
    all_args = parsed.args + unknown

    if cmd in ["run", "chat", "repl"] and not all_args:
        launch_chat(plain=parsed.plain)
    elif cmd == "ask":
        cmd_ask(" ".join(all_args))
    elif cmd == "think":
        cmd_think(" ".join(all_args))
    elif cmd == "decide":
        cmd_decide(" ".join(all_args))
    elif cmd == "code":
        cmd_code(" ".join(all_args))
    elif cmd == 'check-operation':
        cmd_operation(' '.join(all_args))
    elif cmd == "search":
        cmd_search(" ".join(all_args))
    elif cmd == "info":
        cmd_info()
    elif cmd in ["version", "-v", "--version"]:
        cmd_version()
    elif cmd == 'status':
        cmd_status()
    elif cmd == "memory":
        cmd_memory(all_args)
    elif cmd == "dream":
        cmd_dream(all_args)
    elif cmd == "settings":
        cmd_settings(all_args)
    elif cmd in ["help", "-h", "--help"]:
        print("Usage: hari [run|ask|think|decide|check-operation|code|memory|dream|settings|status|info|version|help]")
        print('Interactive launch opens the live dashboard. Use `hari run --plain` for line-based chat.')
    else:
        # Default behavior: treat unknown arguments as direct interactive prompt
        full_query = " ".join([parsed.command] + all_args)
        cmd_ask(full_query)


if __name__ == "__main__":
    main()
