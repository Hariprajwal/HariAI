"""Automatic setup and installation for HariAI local inference engines.

Whenever the user runs `hari` for the first time (or after a fresh install),
this module silently ensures the prebuilt llama-cpp-python binary wheel is
present before the chat session starts. No user action is required.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from typing import Callable, Iterator, Optional

CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
CUDA_WHEEL_INDEX_PREFIX = "https://abetlen.github.io/llama-cpp-python/whl/cu"

# Sentinel so we only auto-install once per process regardless of how many
# times the guard is called (e.g. from both main() and dashboard startup).
_INSTALL_DONE: bool = False
_INSTALL_LOCK = threading.Lock()


def is_engine_installed() -> bool:
    """Return True if llama-cpp-python is importable and functional."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def get_recommended_wheel_index(force_cpu: bool = False) -> str:
    """Detect GPU hardware and return the best precompiled wheel index URL."""
    if force_cpu:
        return CPU_WHEEL_INDEX

    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = getattr(torch.version, 'cuda', None)
            if cuda_ver:
                major_minor = "".join(cuda_ver.split(".")[:2])
                # Map known CUDA versions to wheel tags
                for tag in ("128", "126", "124", "122", "121", "118", "117"):
                    if major_minor.startswith(tag[:2]):
                        return f"{CUDA_WHEEL_INDEX_PREFIX}{tag}"
    except Exception:
        pass

    return CPU_WHEEL_INDEX


def _pip_install_engine(wheel_index: str, emit: Callable[[str], None]) -> bool:
    """Run pip install with --prefer-binary to fetch the prebuilt wheel.

    The critical flags:
      --prefer-binary        -> pick the binary wheel over the source tarball
                               even if the wheel is older. This prevents the
                               source .tar.gz from being downloaded and
                               extracted (which triggers the Windows MAX_PATH
                               260-character crash on deep Svelte UI paths).
      --only-binary=llama-cpp-python  -> hard-fail instead of falling back to
                               source compilation or extraction.
      --extra-index-url      -> the maintainer's official prebuilt wheel repo.
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python",
        "--prefer-binary",
        "--only-binary=llama-cpp-python",
        "--extra-index-url", wheel_index,
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        for line in iter(process.stdout.readline, ""):
            stripped = line.strip()
            if stripped:
                lower = stripped.lower()
                if any(k in lower for k in (
                    "collecting", "downloading", "installing",
                    "successfully", "requirement already", "using cached",
                )):
                    emit(f"  {stripped}")
        process.stdout.close()
        return process.wait() == 0
    except Exception as exc:
        emit(f"  pip subprocess failed: {exc}")
        return False


def install_inference_engine(
    *,
    force_cpu: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> bool:
    """Install the prebuilt binary wheel for llama-cpp-python.

    Safe on every OS:
    - Uses --prefer-binary and --only-binary so pip never downloads the
      .tar.gz source distribution (which triggers the Windows MAX_PATH
      260-char crash deep inside llama.cpp's bundled Svelte UI assets).
    - Falls back automatically from CUDA -> CPU index if the CUDA wheel
      isn't available for the user's Python version.
    """
    logger = emit or print

    if is_engine_installed():
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "unknown")
            logger(f"✓ Local GGUF engine (llama-cpp-python v{ver}) is already installed and ready.")
            return True
        except Exception:
            pass

    wheel_index = get_recommended_wheel_index(force_cpu=force_cpu)
    engine_type = "GPU (CUDA)" if "cu" in wheel_index and not force_cpu else "CPU"

    logger(f"✦ Setting up HariAI Local Engine ({engine_type} prebuilt wheel)...")
    logger(f"  Source: {wheel_index}")

    success = _pip_install_engine(wheel_index, logger)

    if not success and not force_cpu and wheel_index != CPU_WHEEL_INDEX:
        # GPU wheel not available for this Python version — fall back to CPU
        logger("  GPU wheel not available for this Python version, switching to CPU wheel...")
        wheel_index = CPU_WHEEL_INDEX
        success = _pip_install_engine(wheel_index, logger)

    if not success:
        logger("✗ Automatic install failed.")
        logger(f"  Run manually: pip install llama-cpp-python --prefer-binary --only-binary=llama-cpp-python --extra-index-url {wheel_index}")
        return False

    # Invalidate Python's import caches so the newly installed package is visible
    importlib.invalidate_caches()

    if is_engine_installed():
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "unknown")
            logger(f"✓ Engine ready (llama-cpp-python v{ver})!")
            return True
        except Exception as exc:
            logger(f"⚠ Installed but import test failed: {exc}")
            return False
    else:
        logger("⚠ Installation completed but llama_cpp could not be imported.")
        logger("  Try: pip install llama-cpp-python --prefer-binary --only-binary=llama-cpp-python "
               f"--extra-index-url {wheel_index}")
        return False


def ensure_engine_ready(
    *,
    silent: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> bool:
    """Called automatically at startup before chat begins.

    Installs llama-cpp-python once per process if not already present.
    - silent=True  : suppress all output (for background pre-load).
    - silent=False : print progress so the user knows what's happening.

    Returns True if the engine is (or becomes) available.
    """
    global _INSTALL_DONE
    with _INSTALL_LOCK:
        if _INSTALL_DONE:
            return is_engine_installed()

        if is_engine_installed():
            _INSTALL_DONE = True
            return True

        def _noop(_: str) -> None:
            pass

        logger = _noop if silent else (emit or print)

        if not silent:
            logger("")
            logger("✦ HariAI · First-time setup: installing local AI engine...")
            logger("  (This runs once and takes about 10-30 seconds)")
            logger("")

        ok = install_inference_engine(emit=logger)
        _INSTALL_DONE = True

        if not silent and not ok:
            logger("")
            logger("⚠  Could not install the local engine automatically.")
            logger("   HariAI will still launch, but chat replies will not work")
            logger("   until you run:  hari setup")
            logger("")

        return ok
