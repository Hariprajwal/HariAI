"""Persistent, explicit backend selection for HariAI chat."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ChatSettings:
    schema_version: int = 2
    backend: str = 'local'
    model_file: str = ''
    nano_checkpoint: str = ''
    nano_tokenizer: str = ''
    ollama_model: str = ''
    ollama_host: str = 'http://127.0.0.1:11434'
    openjev_checkpoint: str = ''

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError('Unsupported settings version')
        if self.backend not in ('local', 'nano', 'ollama'):
            raise ValueError('backend must be local, nano (research), or ollama')
        if not self.ollama_host.startswith(('http://', 'https://')):
            raise ValueError('ollama_host must be an HTTP or HTTPS URL')


def settings_path() -> Path:
    return Path(os.environ.get('HARIAI_DATA_DIR', Path.home() / '.hari_ai')) / 'settings.json'


def load_settings(path: Optional[Path] = None) -> ChatSettings:
    path = path or settings_path()
    if not path.exists():
        return ChatSettings()
    values = json.loads(path.read_text(encoding='utf-8'))
    if 'schema_version' not in values:
        # Retire only the old implicit research default. Preserve custom Nano
        # checkpoints and explicit Ollama selection during the upgrade.
        if values.get('backend') == 'nano' and not values.get('nano_checkpoint'):
            values['backend'] = 'local'
        values['schema_version'] = 2
    settings = ChatSettings(**values)
    settings.validate()
    return settings


def save_settings(settings: ChatSettings, path: Optional[Path] = None) -> None:
    settings.validate()
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(asdict(settings), indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
