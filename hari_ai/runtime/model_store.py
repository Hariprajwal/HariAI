"""Pinned, checksum-verified model download, separate from the Python wheel."""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from hari_ai.runtime.settings import settings_path

MODEL_REPO = 'bartowski/SmolLM2-135M-Instruct-GGUF'
MODEL_REVISION = '09816acd5d99df7be770d85ea30822623dab342c'
MODEL_FILENAME = 'SmolLM2-135M-Instruct-Q4_K_M.gguf'
MODEL_BYTES = 105454432
MODEL_SHA256 = '2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d'
MODEL_URL = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILENAME}'


def default_model_path() -> Path:
    return settings_path().parent / 'models' / MODEL_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> bool:
    return (path.is_file() and path.stat().st_size == MODEL_BYTES
            and sha256_file(path) == MODEL_SHA256)


def download_model(destination: Path | None = None) -> Path:
    """Download only on explicit setup; never send conversations over the network."""
    destination = destination or default_model_path()
    if verify_model(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.partial', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'HariAI/0.2'})
            with urllib.request.urlopen(request, timeout=60) as source:
                size = 0
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > MODEL_BYTES:
                        raise ValueError('Model download exceeds the expected size.')
                    output.write(chunk)
        if not verify_model(temporary):
            raise ValueError('Model checksum or size mismatch; existing model was not replaced.')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
