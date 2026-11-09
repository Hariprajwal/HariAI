"""Read-only observations for the terminal dashboard.

Snapshots never load a model, contact a service, or start learning. Device and
readiness describe the selected backend, not hardware merely present on the PC.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
except ImportError:  # Plain installations can still expose model/session state.
    psutil = None

from hari_ai.runtime.local_model import bundled_model_path


@dataclass(frozen=True)
class RuntimeSnapshot:
    backend: str
    model_name: str
    model_path: str
    model_bytes: int | None
    model_loaded: bool
    model_available: bool
    device: str
    threads: int | None
    context_tokens: int | None
    turns: int
    working_items: int
    episodic_entries: int
    semantic_facts: int
    recipes: int
    dream_state: str
    dream_cycles: int
    last_inspected: int | None
    uptime_seconds: float
    process_mb: float | None
    available_ram_gb: float | None
    cpu_percent: float | None
    readiness: str
    model_origin: str = ''


def _file_size(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _measurement(read: Callable[[], float]) -> float | None:
    """Unavailable metrics are unknown, never misleading zero readings."""
    try:
        value = float(read())
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, RuntimeError, ValueError, AttributeError):
        return None
    except Exception as exc:
        # psutil.AccessDenied and NoSuchProcess are not OSError subclasses.
        if psutil is not None and isinstance(exc, psutil.Error):
            return None
        raise


class SessionStatus:
    def __init__(self, engine_getter: Callable[[], Any], dream_controller: Any):
        self._engine_getter = engine_getter
        self._dream = dream_controller
        self._started = time.monotonic()
        self._cpu_started: float | None = None
        if psutil is not None:
            # Prime a nonblocking system CPU sample; the first call has no
            # useful interval and must not be shown as measured idle time.
            if _measurement(lambda: psutil.cpu_percent(interval=None)) is not None:
                self._cpu_started = self._started

    def snapshot(self) -> RuntimeSnapshot:
        engine = self._engine_getter()  # Settings can replace the chat engine.
        settings = engine.settings
        backend = settings.backend
        model_path = ''
        model_bytes = None
        loaded = available = False
        threads = context_tokens = None
        model_origin = ''
        if backend == 'local':
            local = engine.local_model
            path = Path(local.path)
            model_path = str(path)
            model_bytes = _file_size(path)
            available = model_bytes is not None
            model = getattr(local, '_model', None)
            loaded = model is not None
            available = loaded or available
            try:
                bundled = path.resolve() == bundled_model_path().resolve()
            except OSError:
                bundled = False
            model_name = 'HariAI Chat' if bundled else path.name
            model_origin = 'SmolLM-135M-Instruct derived' if bundled else 'User-configured GGUF'
            metadata = getattr(model, 'metadata', None)
            if isinstance(metadata, dict) and metadata.get('general.name'):
                if bundled:
                    model_origin = str(metadata['general.name']) + ' derived'
                else:
                    model_name = str(metadata['general.name'])
            device = 'CPU'
            threads = min(local.threads, os.cpu_count() or 1)
            context_tokens = local.context_size
            readiness = 'Loaded' if loaded else ('Available · loads on first reply' if available else 'Model file missing')
        elif backend == 'nano':
            native = engine.engine
            configured_path = getattr(settings, 'nano_checkpoint', '')
            path = (Path(configured_path).expanduser() if configured_path else
                    Path(__file__).resolve().parents[2] / 'checkpoints' / 'hari_ai_nano.pt')
            model_path = str(path)
            model_bytes = _file_size(path)
            loaded = bool(getattr(native, 'checkpoint_loaded', False))
            available = loaded or model_bytes is not None
            model_name = 'HariNano · research checkpoint'
            device = str(getattr(native, 'device', 'Unknown'))
            config = getattr(getattr(native, 'model', None), 'config', None)
            context_tokens = getattr(config, 'max_seq_len', None)
            readiness = 'Research checkpoint loaded' if loaded else 'No trained checkpoint loaded'
        else:
            model_name = settings.ollama_model or 'Ollama · automatic selection'
            device = 'Ollama service · device unverified'
            readiness = 'Service unverified · checked when chatting'

        stats = engine.memory.get_status_summary()
        cycles = self._dream.sleep_cycle_history
        now = time.monotonic()
        process_mb = available_ram_gb = cpu_percent = None
        if psutil is not None:
            process_mb = _measurement(lambda: psutil.Process().memory_info().rss / 1_000_000)
            available_ram_gb = _measurement(lambda: psutil.virtual_memory().available / 1_000_000_000)
            if self._cpu_started is not None and now - self._cpu_started >= 0.1:
                cpu_percent = _measurement(lambda: psutil.cpu_percent(interval=None))
            elif self._cpu_started is None:
                if _measurement(lambda: psutil.cpu_percent(interval=None)) is not None:
                    self._cpu_started = now
        return RuntimeSnapshot(
            backend=backend, model_name=model_name, model_path=model_path,
            model_bytes=model_bytes, model_loaded=loaded, model_available=available,
            device=device, threads=threads, context_tokens=context_tokens,
            turns=sum(message.role == 'user' for message in engine.history),
            working_items=stats['working_scratchpad_items'],
            episodic_entries=stats['episodic_entries_count'],
            semantic_facts=stats['semantic_facts_count'],
            recipes=stats['procedural_recipes_count'],
            dream_state=self._dream.current_state.value,
            dream_cycles=len(cycles),
            last_inspected=cycles[-1].experiences_inspected if cycles else None,
            uptime_seconds=max(0.0, now - self._started),
            process_mb=process_mb, available_ram_gb=available_ram_gb,
            cpu_percent=cpu_percent, readiness=readiness, model_origin=model_origin,
        )
