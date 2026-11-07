"""Formal Verification Kernel for HariAI.

Integrates:
1. Lean 4 Mechanical Theorem Verification (AlphaProof / LeanDojo style)
2. Deterministic Python Assertion Sandboxing
3. Fallback routing when external compiler toolchains are absent
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """Outcome of a formal verification run."""

    passed: bool
    engine: str  # "lean4" or "python_assert"
    stdout: str
    stderr: str
    execution_time_seconds: float
    error_line: Optional[int] = None


def is_lean_available() -> bool:
    """Checks if the Lean 4 compiler ('lean') is available in the system PATH."""
    return shutil.which("lean") is not None


def verify_lean_code(code: str, timeout: float = 10.0) -> VerificationResult:
    """Verifies a Lean 4 formal theorem or proof using the trusted Lean kernel."""
    t_start = time.time()

    if not is_lean_available():
        return VerificationResult(
            passed=False,
            engine="lean4",
            stdout="",
            stderr="Lean 4 compiler ('lean') is not installed or not in PATH.",
            execution_time_seconds=0.0,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        lean_path = os.path.join(tmpdir, "verification_target.lean")
        with open(lean_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                ["lean", "--run", lean_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            t_end = time.time()
            passed = (proc.returncode == 0)
            return VerificationResult(
                passed=passed,
                engine="lean4",
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                execution_time_seconds=t_end - t_start,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                engine="lean4",
                stdout="",
                stderr=f"Verification timed out after {timeout} seconds.",
                execution_time_seconds=timeout,
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                engine="lean4",
                stdout="",
                stderr=str(e),
                execution_time_seconds=time.time() - t_start,
            )


def verify_python_asserts(code: str, tests: str, timeout: float = 5.0) -> VerificationResult:
    """Verifies Python algorithmic code against deterministic assertion suites in an isolated process."""
    t_start = time.time()
    combined_script = f"{code}\n\n# Verification Tests\n{tests}\nprint('__VERIFICATION_PASSED__')"

    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = os.path.join(tmpdir, "target.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(combined_script)

        try:
            proc = subprocess.run(
                [sys.executable, py_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            t_end = time.time()
            passed = (proc.returncode == 0 and "__VERIFICATION_PASSED__" in proc.stdout)
            return VerificationResult(
                passed=passed,
                engine="python_assert",
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                execution_time_seconds=t_end - t_start,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                engine="python_assert",
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds.",
                execution_time_seconds=timeout,
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                engine="python_assert",
                stdout="",
                stderr=str(e),
                execution_time_seconds=time.time() - t_start,
            )


def auto_verify(problem_type: str, code: str, tests: str = "") -> VerificationResult:
    """Automatically selects the best verification backend (Lean 4 or Python sandbox)."""
    if problem_type == "lean" or "<|lean|>" in code:
        return verify_lean_code(code)
    return verify_python_asserts(code, tests)
