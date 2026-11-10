"""Friendly follow-up prompts for commands that need user input."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandPrompt:
    """One concise question and the command prefix completed by its answer."""

    prefix: str
    question: str


def missing_input_prompt(text: str) -> CommandPrompt | None:
    """Return the required next question, without running the command."""
    if not isinstance(text, str):
        return None
    tokens = text.strip().split(None, 1)
    if not tokens:
        return None
    command = tokens[0].lower()
    argument = tokens[1].strip() if len(tokens) == 2 else ''

    direct_questions = {
        '/ask': 'What would you like to ask HariAI?',
        '/search': 'What should HariAI search for?',
        '/think': 'What problem should HariAI think through?',
        '/decide': 'What situation should HariAI evaluate?',
        '/code': 'What code would you like HariAI to write?',
        '/swarm': 'What topic should the HariAI swarm debate?',
        '/convert': 'Enter the source file path, for example C:\\Documents\\report.docx.',
        '/autonomous': 'Describe the desktop task for HariAI to perform.',
        '/auto': 'Describe the desktop task for HariAI to perform.',
        '/check-operation': 'Paste the JSON operation record to evaluate.',
    }
    if command in direct_questions and not argument:
        return CommandPrompt(command, direct_questions[command])

    parts = argument.split(None, 1)
    subcommand = parts[0].lower() if parts else ''
    remainder = parts[1].strip() if len(parts) > 1 else ''

    if command == '/tools' and subcommand == 'run' and not remainder:
        return CommandPrompt('/tools run', 'Describe the tool task HariAI should carry out.')
    if command == '/memory' and subcommand == 'add' and not remainder:
        return CommandPrompt('/memory add', 'What memory should HariAI save locally?')
    if command == '/memory' and subcommand == 'delete' and not remainder:
        return CommandPrompt('/memory delete', 'Enter the memory ID to delete.')
    if command == '/dream' and subcommand == 'auto' and not remainder:
        return CommandPrompt('/dream auto', 'Should automatic memory consolidation be on or off?')
    if command == '/mcp' and subcommand == 'add' and not remainder:
        return CommandPrompt('/mcp add', 'Enter: SERVER_NAME COMMAND ARGUMENTS. Example: files npx -y @modelcontextprotocol/server-filesystem C:\\Workspace')
    if command == '/cluster' and subcommand == 'debate' and not remainder:
        return CommandPrompt('/cluster debate', 'What topic should the HariAI cluster debate?')

    if command == '/settings' and subcommand in {
        'backend', 'model-file', 'ollama-model', 'ollama-host', 'confidence',
        'nano-model', 'nano-checkpoint', 'nano-tokenizer', 'openjev-checkpoint',
    } and not remainder:
        questions = {
            'backend': 'Choose a backend: local, nano, or ollama.',
            'model-file': 'Enter the full path to the GGUF model file.',
            'ollama-model': 'Enter the installed Ollama model name.',
            'ollama-host': 'Enter the Ollama URL, for example http://127.0.0.1:11434.',
            'confidence': 'Should token likelihood display be on or off?',
            'nano-model': 'Enter the checkpoint path and tokenizer path, each in quotes if it contains spaces.',
            'nano-checkpoint': 'Enter the full path to the HariNano checkpoint.',
            'nano-tokenizer': 'Enter the full path to the HariNano tokenizer.',
            'openjev-checkpoint': 'Enter the full path to the Open-JEV checkpoint.',
        }
        return CommandPrompt(f'/settings {subcommand}', questions[subcommand])

    if command == '/agent' and subcommand in {'model', 'host', 'region', 'run', 'auto', 'autonomous', 'approve', 'ocr'} and not remainder:
        questions = {
            'model': 'Enter an installed local vision model name.',
            'host': 'Enter the local Ollama URL, for example http://127.0.0.1:11434.',
            'region': 'Enter LEFT TOP WIDTH HEIGHT in screen pixels, for example 100 100 900 600.',
            'run': 'Describe the desktop task HariAI should inspect and propose.',
            'auto': 'Describe the desktop task HariAI should perform.',
            'autonomous': 'Describe the desktop task HariAI should perform.',
            'approve': 'Enter the pending approval ID shown by HariAI.',
            'ocr': 'Should optional local OCR be on or off?',
        }
        return CommandPrompt(f'/agent {subcommand}', questions[subcommand])
    return None
