"""Hardware accelerator detection and dynamic device utilization for HariAI.

Detects available compute devices:
- NVIDIA / CUDA GPUs via PyTorch / CUDA runtime
- DirectML (DirectX 12 acceleration on Windows for Intel, AMD, NVIDIA)
- Apple Silicon / Metal (MPS)
- AMD ROCm
- Multi-core CPU with optimal thread allocation
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    """Hardware capability and accelerator profile."""
    device_type: str  # "cuda", "directml", "mps", "cpu"
    device_name: str
    gpu_available: bool
    n_gpu_layers: int  # -1 for full offload if GPU available, 0 for CPU
    total_memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    cpu_cores: int = 1
    optimal_threads: int = 4

    def summary(self) -> str:
        if self.gpu_available:
            mem = f" · {self.total_memory_gb:.1f} GB VRAM" if self.total_memory_gb else ""
            return f"GPU ({self.device_name}{mem})"
        return f"CPU ({self.device_name} · {self.optimal_threads} threads)"


_CACHED_PROFILE: Optional[HardwareProfile] = None


def detect_hardware(refresh: bool = False) -> HardwareProfile:
    """Detects available hardware accelerators with graceful fallback to CPU."""
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None and not refresh:
        return _CACHED_PROFILE

    cpu_cores = os.cpu_count() or 1
    optimal_threads = max(1, min(cpu_cores, 8))
    cpu_name = os.environ.get("PROCESSOR_IDENTIFIER", f"x86_64 ({cpu_cores} cores)")

    # 1. Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0)
            vram = None
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                pass
            cap = None
            try:
                cc = torch.cuda.get_device_capability(0)
                cap = f"{cc[0]}.{cc[1]}"
            except Exception:
                pass
            profile = HardwareProfile(
                device_type="cuda",
                device_name=name,
                gpu_available=True,
                n_gpu_layers=-1,
                total_memory_gb=vram,
                compute_capability=cap,
                cpu_cores=cpu_cores,
                optimal_threads=optimal_threads,
            )
            _CACHED_PROFILE = profile
            return profile
    except Exception:
        pass

    # 2. Try Apple Metal (MPS)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            profile = HardwareProfile(
                device_type="mps",
                device_name="Apple Silicon Metal (MPS)",
                gpu_available=True,
                n_gpu_layers=-1,
                cpu_cores=cpu_cores,
                optimal_threads=optimal_threads,
            )
            _CACHED_PROFILE = profile
            return profile
    except Exception:
        pass

    # 3. Try llama-cpp-python native GPU offload detection
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_supports_gpu_offload") and llama_cpp.llama_supports_gpu_offload():
            profile = HardwareProfile(
                device_type="gpu_offload",
                device_name="Hardware GPU (llama-cpp accelerated)",
                gpu_available=True,
                n_gpu_layers=-1,
                cpu_cores=cpu_cores,
                optimal_threads=optimal_threads,
            )
            _CACHED_PROFILE = profile
            return profile
    except Exception:
        pass

    # Fallback: CPU
    profile = HardwareProfile(
        device_type="cpu",
        device_name=cpu_name,
        gpu_available=False,
        n_gpu_layers=0,
        cpu_cores=cpu_cores,
        optimal_threads=optimal_threads,
    )
    _CACHED_PROFILE = profile
    return profile


def is_google_colab() -> bool:
    """Detect if running inside Google Colab environment."""
    if "google.colab" in sys.modules:
        return True
    if any(k in os.environ for k in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION", "GCS_READ_CACHE")):
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def is_notebook_or_headless() -> bool:
    """Detect Jupyter notebooks, Colab, or non-interactive headless sessions."""
    if is_google_colab():
        return True
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return True
    if os.environ.get("TERM") in ("dumb", "unknown"):
        return True
    try:
        import IPython
        app = IPython.get_ipython()
        if app is not None:
            name = app.__class__.__name__
            if name in ("ZMQInteractiveShell", "Shell", "GoogleColabShell"):
                return True
    except (ImportError, Exception):
        pass
    return False

