"""Automatic setup and installation for HariAI local inference engines."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Callable, Optional

CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
CUDA_WHEEL_INDEX_PREFIX = "https://abetlen.github.io/llama-cpp-python/whl/cu"


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
                if major_minor in ("121", "122", "124"):
                    return f"{CUDA_WHEEL_INDEX_PREFIX}{major_minor}"
    except Exception:
        pass

    return CPU_WHEEL_INDEX


def install_inference_engine(
    *,
    force_cpu: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> bool:
    """Installs the prebuilt binary wheel for llama-cpp-python.

    Uses the maintainer's official prebuilt wheel repository to ensure
    safe, instant binary installation without triggering C++ compilation
    or Windows MAX_PATH limits.
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
    engine_type = "GPU (CUDA)" if "cu" in wheel_index else "CPU"

    logger(f"✦ Setting up HariAI Local GGUF Engine ({engine_type})...")
    logger(f"  Fetching prebuilt binary wheel from: {wheel_index}")
    logger("  Installing llama-cpp-python (this may take a few seconds)...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "llama-cpp-python",
        "--prefer-binary",
        "--extra-index-url",
        wheel_index,
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
                if any(k in lower for k in ("collecting", "downloading", "installing", "successfully", "requirement already")):
                    logger(f"  {stripped}")

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            logger(f"✗ Installation exited with code {return_code}.")
            logger(f"  Manual fallback: pip install llama-cpp-python --extra-index-url {wheel_index}")
            return False

        # Invalidate import caches so newly installed packages are immediately visible
        importlib.invalidate_caches()

        if is_engine_installed():
            try:
                import llama_cpp
                ver = getattr(llama_cpp, "__version__", "unknown")
                logger(f"✓ Engine installed and verified successfully (v{ver})!")
                logger("  The local GGUF model is now ready for offline chat.")
                return True
            except Exception as e:
                logger(f"⚠ Package installed, but import test reported: {e}")
                return False
        else:
            logger("⚠ Installation completed, but llama_cpp could not be imported.")
            return False

    except Exception as exc:
        logger(f"✗ Setup failed with error: {exc}")
        return False
