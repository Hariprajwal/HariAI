"""Multi-Teacher Distillation and Verification Engine for HariAI.

Orchestrates diverse teacher models:
- Qwen 2.5 Coder (Coding & Algorithmic Logic)
- Gemma 2 (General Knowledge & Linguistic Nuance)
- Phi-3.5 (Structured Reasoning & Formal Logic)
- SmolLM (Small-Model Efficiency Baseline)
- High-Purity Synthetic Fallback (Deterministic Verification)

Teacher output is unverified unless independently checked by a separate evaluator.
This module does not execute teacher-generated code.
"""

from __future__ import annotations

import enum
import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hari_ai.data.synthetic import SyntheticCurriculumGenerator
from hari_ai.verification.lean_kernel import auto_verify


class TeacherModelType(str, enum.Enum):
    QWEN_CODER = "qwen2.5:7b"
    GEMMA2 = "gemma2:9b"
    PHI3 = "phi3.5:latest"
    SMOLLM = "smollm:135m"
    SYNTHETIC_FALLBACK = "synthetic_rule_generator"


class DistillationTask(str, enum.Enum):
    CODE_SYNTHESIS = "code_synthesis"
    LEAN_MATH_PROOF = "lean_math_proof"
    DECISION_TRAJECTORY = "decision_trajectory"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class VerifiedSample:
    """A distillation sample that has passed mechanical verification gates."""

    prompt: str
    response: str
    task: DistillationTask
    teacher: TeacherModelType
    verified: bool
    reward: float
    verification_engine: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTeacherDistiller:
    """Routes generation to local teacher models with automatic verification and fallback."""

    def __init__(
        self,
        ollama_endpoint: str = "http://127.0.0.1:11434",
        request_timeout: float = 5.0,
    ):
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.request_timeout = request_timeout
        self.synthetic_gen = SyntheticCurriculumGenerator(seed=int(time.time()))

    def _query_ollama(self, model_name: str, prompt: str) -> Optional[str]:
        """Attempts to query local Ollama instance via HTTP. Returns None if unreachable."""
        url = f"{self.ollama_endpoint}/api/generate"
        payload = json.dumps(
            {"model": model_name, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "").strip()
        except Exception:
            # Service not running, socket blocked, or timeout
            return None
        return None

    def route_task_to_teacher(self, task: DistillationTask) -> TeacherModelType:
        """Assigns the best specialized teacher model for each domain."""
        if task == DistillationTask.CODE_SYNTHESIS:
            return TeacherModelType.QWEN_CODER
        elif task == DistillationTask.LEAN_MATH_PROOF:
            return TeacherModelType.PHI3
        elif task == DistillationTask.DECISION_TRAJECTORY:
            return TeacherModelType.QWEN_CODER
        else:
            return TeacherModelType.GEMMA2

    def distill_sample(
        self, task: DistillationTask, custom_prompt: Optional[str] = None
    ) -> VerifiedSample:
        """Generates a sample from assigned teacher (or synthetic fallback) and verifies it."""
        teacher = self.route_task_to_teacher(task)
        raw_response: Optional[str] = None

        prompt = custom_prompt or f"Generate a verified solution for {task.value}"

        # 1. Attempt teacher model generation if available
        if teacher != TeacherModelType.SYNTHETIC_FALLBACK:
            raw_response = self._query_ollama(teacher.value, prompt)

        # 2. Seamless fallback to deterministic synthetic generator
        used_teacher = teacher
        if not raw_response:
            used_teacher = TeacherModelType.SYNTHETIC_FALLBACK
            if task == DistillationTask.CODE_SYNTHESIS:
                raw_response = self.synthetic_gen.generate_code_sample()
            elif task == DistillationTask.LEAN_MATH_PROOF:
                raw_response = self.synthetic_gen.generate_lean_math_sample()
            elif task == DistillationTask.DECISION_TRAJECTORY:
                raw_response = self.synthetic_gen.generate_decision_sample()
            else:
                raw_response = (
                    f"<|thought|>Synthesizing factual response for: {prompt}<|eos|>"
                )

        # Formatting, self-written assertions, and theorem keywords do not prove
        # correctness. Generated code must not run on the host as a sandbox.
        verification_passed = False
        verification_engine = "not_evaluated"

        reward = 1.0 if verification_passed else 0.0

        return VerifiedSample(
            prompt=prompt,
            response=raw_response,
            task=task,
            teacher=used_teacher,
            verified=verification_passed,
            reward=reward,
            verification_engine=verification_engine,
        )

    def generate_verified_batch(
        self, count: int = 10, task: Optional[DistillationTask] = None
    ) -> List[VerifiedSample]:
        """Generates a batch of verified distillation samples, keeping only verified outputs."""
        tasks = [
            DistillationTask.CODE_SYNTHESIS,
            DistillationTask.LEAN_MATH_PROOF,
            DistillationTask.DECISION_TRAJECTORY,
        ]
        results: List[VerifiedSample] = []

        for i in range(count):
            current_task = task or tasks[i % len(tasks)]
            sample = self.distill_sample(current_task)
            if sample.verified:
                results.append(sample)

        return results

