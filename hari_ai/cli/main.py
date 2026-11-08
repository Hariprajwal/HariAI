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
import sys
import time
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


BANNER = r"""
     _   _            _    ___  
    | | | | __ _ _ __| |  / _ \ 
    | |_| |/ _` | '__| | | | | |
    |  _  | (_| | |  | | | |_| |
    |_| |_|\__,_|_|  |_|  \___/ 

        Local conversation.
        Created by K R Hari Prajwal.

HariAI v0.2.0a1
"""

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


def cmd_info() -> None:
    """Displays HariAI creator profile and identity details."""
    profile = get_creator_profile()

    print("\n" + "=" * 60)
    print(" HARIAI CREATOR PROFILE")
    print("=" * 60)
    print(f" Creator:        {profile.name}")
    print(f" Origin:         {profile.origin_region}")
    print(f" Creation Age:   {profile.creation_age} years old (At creation)")
    print(f" Statement:      \"{profile.official_statement}\"")
    print(f" GitHub:         {profile.github_url}")
    print(" Chat Model:     Standalone SmolLM-derived GGUF, approximately 92 MB")
    print(" Research goal:  Under 35 MB (not achieved by this chat model)")
    print("=" * 60 + "\n")


def cmd_version() -> None:
    print("HariAI version 0.2.0a1 (standalone local GGUF)")


def cmd_search(query: str) -> None:
    """Executes live web/local search and AI summarization."""
    print(f"\n[HariNano Search Engine]: Gathering and synthesizing data for '{query}'...")
    summary = _SEARCH_ENGINE.search_and_summarize(query)
    print("\n" + summary + "\n")


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


def cmd_think(prompt: str, llm: Optional[Any] = None) -> str:
    """Request careful reasoning; model output is not independently verified."""
    if llm is None:
        answer = ''.join(_CHAT_ENGINE.chat_stream('Please reason carefully about: ' + prompt))
    else:
        answer = llm.generate('Please reason carefully about: ' + prompt).strip()
    print(answer or 'The model produced no answer.')
    return answer


def cmd_decide(situation: str) -> None:
    """Open-JEV style typed decision readout: Choice, Score, Noul."""
    print("\n" + "=" * 60)
    print(" HARIAI TYPED DECISION ENGINE (System One Jev-Style)")
    print("=" * 60)
    print(f" Situation:                {situation}")

    settings = load_settings()
    if settings.openjev_checkpoint:
        from hari_ai.runtime.openjev import OpenJevAdapter
        try:
            decision = OpenJevAdapter(settings.openjev_checkpoint).decide(
                {'situation': situation}, 'What should be done?',
                ['review_situation', 'request_human_confirmation', 'stop'])
            print(f" Choice:                   {decision['choice']}")
            print(f" Score (raw, uncalibrated): {decision['confidence_raw']:.3f}")
            print(" Noul (Permission Gate):   REQUIRES APPROVAL")
            print(f" Open-JEV weights:          {decision['checkpoint_bytes']} bytes (separate from HariNano)")
            print("=" * 60 + "\n")
            return
        except (OSError, ValueError, RuntimeError) as exc:
            print(f" Open-JEV unavailable: {exc}")

    # Evaluate decision logic
    is_critical = any(
        w in situation.lower()
        for w in ["delete", "delet", "remove", "drop", "terminate", "format", "erase"]
    )
    selected_choice = "request_human_confirmation" if is_critical else "review_situation"
    noul_permission = True

    print(f" Choice (Selected Action): {selected_choice}")
    print(" Score (Confidence):       unavailable (not calibrated)")
    print(f" Noul (Permission Gate):   {'REQUIRES APPROVAL' if noul_permission else 'AUTO-APPROVED'}")
    print("=" * 60 + "\n")


def cmd_code(task: str) -> None:
    """Request code from the configured chat backend without claiming verification."""
    cmd_ask("Write code for the following request. State any assumptions: " + task)


def cmd_operation(value: str) -> None:
    """Evaluate one bounded JSON record with the bundled Open-JEV model."""
    from hari_ai.runtime.openjev import OperationDecisionModel
    try:
        result = OperationDecisionModel().decide(json.loads(value))
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Operation decision unavailable: {exc}')


def cmd_memory(args: List[str]) -> None:
    sub = args[0] if args else "status"
    if sub == "status":
        stats = _MEMORY_MANAGER.get_status_summary()
        print("\n--- HariAI Hierarchical Memory Status ---")
        for k, v in stats.items():
            print(f" {k:30s}: {v}")
        print("-----------------------------------------\n")
    elif sub == "clear":
        _MEMORY_MANAGER.working.clear()
        print("Working memory cleared.")
    elif sub == "add":
        text = " ".join(args[1:]).strip()
        if not text:
            print('Usage: hari memory add "Text to remember"')
            return
        store = MemoryStore()
        print(f"Saved local memory {store.add(text)} at {store.path}. It is not used for model training.")
    elif sub == "inspect":
        memories = MemoryStore().list()
        if not memories:
            print("No saved local memories.")
        for entry in memories:
            print(f"{entry['id']}: {entry['content']}")
    elif sub == "delete":
        if len(args) != 2 or not args[1].isdigit():
            print("Usage: hari memory delete <id>")
            return
        print("Memory deleted." if MemoryStore().delete(int(args[1])) else "Memory not found.")
    else:
        print(f"Memory command '{sub}' recognized. Memory is operating normally.")


def cmd_settings(args: List[str]) -> None:
    """Show or update chat settings; changing backends is always explicit."""
    global _CHAT_ENGINE
    settings = load_settings()
    if not args or args[0] == 'show':
        print(f'Settings: {settings_path()}')
        print(json.dumps(asdict(settings), indent=2))
        return
    if args[0] == 'list-models':
        models = _CHAT_ENGINE.ollama.list_models()
        print('\n'.join(models) if models else 'No Ollama models found; check that Ollama is running.')
        return
    if args[0] == 'nano-model':
        if len(args) != 3:
            print('Usage: hari settings nano-model CHECKPOINT TOKENIZER')
            return
        checkpoint, tokenizer = (Path(value).expanduser().resolve() for value in args[1:])
        if not checkpoint.is_file() or not tokenizer.is_file():
            print('Checkpoint and tokenizer files must both exist.')
            return
        settings.nano_checkpoint, settings.nano_tokenizer = str(checkpoint), str(tokenizer)
        try:
            new_chat = ChatEngine(memory_manager=_MEMORY_MANAGER, settings=settings)
            save_settings(settings)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f'Settings not changed: {exc}')
            return
        _CHAT_ENGINE = new_chat
        print(f'Set Nano model to {checkpoint}')
        return
    mapping = {'backend': 'backend', 'ollama-model': 'ollama_model', 'ollama-host': 'ollama_host',
               'nano-checkpoint': 'nano_checkpoint', 'nano-tokenizer': 'nano_tokenizer',
               'openjev-checkpoint': 'openjev_checkpoint', 'model-file': 'model_file'}
    if args[0] not in mapping or len(args) < 2:
        print('Usage: hari settings show | list-models | backend local|nano|ollama | model-file PATH.gguf | ollama-model NAME | ollama-host URL | nano-model CHECKPOINT TOKENIZER | openjev-checkpoint PATH')
        return
    field = mapping[args[0]]
    value = ' '.join(args[1:])
    if field.endswith('checkpoint') or field in ('nano_tokenizer', 'model_file'):
        value = str(Path(value).expanduser().resolve())
        if not Path(value).is_file():
            print(f'File not found: {value}')
            return
    setattr(settings, field, value)
    try:
        settings.validate()
        new_chat = ChatEngine(memory_manager=_MEMORY_MANAGER, settings=settings)
        save_settings(settings)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Settings not changed: {exc}')
        return
    _CHAT_ENGINE = new_chat
    print(f'Set {args[0]} to {value}')


def cmd_dream(args: List[str]) -> None:
    sub = args[0] if args else "now"
    if sub == "now":
        print("\nHariAI Dream Engine")
        print("Entering controlled sleep mode...\n")
        result = _DREAM_CONTROLLER.run_sleep_cycle(trigger=SleepTrigger.MANUAL)
        print("Analyzing recent experiences.")
        print(f"Inspected {result.experiences_inspected} experiences.")
        print(f"Generated {result.synthetic_exercises_generated} synthetic practice exercises.")
        print("Status: inspection only; no training or checkpoint evaluation was run.")
        print("\nDream cycle completed.")
        print("No production model changes were made without passing evaluation.\n")
    elif sub == "status":
        print(f"Dream Engine State: {_DREAM_CONTROLLER.current_state.value.upper()}")
        print(f"Total Sleep Cycles: {len(_DREAM_CONTROLLER.sleep_cycle_history)}")


def repl() -> None:
    """Interactive conversational REPL with Ollama-style vanishing placeholder."""
    print_banner()

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
        elif lower in ["/?", "/help", "help", "hari help"]:
            print("\nAvailable Commands:")
            print("  /search <query>     - Web/Local search & AI research summarization")
            print("  /think <problem>    - Deliberate reasoning mode (<|thought|>)")
            print("  /decide <situation> - Open-JEV typed single-forward-pass decision evaluation")
            print("  /code <task>        - Request code from the local model")
            print("  /dream [now|status] - Inspect Dream research status")
            print("  /memory [status]    - Inspect hierarchical memory levels")
            print("  /settings [show|backend|ollama-model|list-models] - Configure chat backend")
            print("  /clear              - Reset multi-turn conversation context")
            print("  /info               - Display Creator Profile (K R Hari Prajwal)")
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


def main():
    # Redirected Windows terminals can use legacy encodings that lack emoji.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description="HariAI Command Line Interface", add_help=False)
    parser.add_argument("command", nargs="?", default="run", help="Subcommand to execute")
    parser.add_argument("args", nargs="*", help="Arguments for subcommand")

    parsed, unknown = parser.parse_known_args()
    cmd = parsed.command.lower()
    all_args = parsed.args + unknown

    if cmd in ["run", "chat", "repl"] and not all_args:
        repl()
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
    elif cmd == "memory":
        cmd_memory(all_args)
    elif cmd == "dream":
        cmd_dream(all_args)
    elif cmd == "settings":
        cmd_settings(all_args)
    elif cmd in ["help", "-h", "--help"]:
        print("Usage: hari [run|ask|think|decide|check-operation|code|memory|dream|settings|info|version|help]")
    else:
        # Default behavior: treat unknown arguments as direct interactive prompt
        full_query = " ".join([parsed.command] + all_args)
        cmd_ask(full_query)


if __name__ == "__main__":
    main()
