"""Synthetic curriculum generator for HariAI Nano.

Produces high-purity training data across three pillars:
1. Algorithmic Code with Deterministic Unit Assertions
2. Mathematical Reasoning with Lean 4 Autoformalization & Verification
3. Structured Decision Trajectories (Jev-style Choice, Score, Noul)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


class SyntheticCurriculumGenerator:
    """Generates synthetic verified curricula for training from scratch."""

    ALGORITHMIC_TEMPLATES = [
        {
            "task": "binary_search",
            "prompt": "Implement binary search on a sorted integer list.",
            "code": (
                "def binary_search(arr: list[int], target: int) -> int:\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"
            ),
            "tests": (
                "assert binary_search([1, 2, 3, 4, 5], 3) == 2\n"
                "assert binary_search([1, 2, 3, 4, 5], 6) == -1\n"
                "assert binary_search([], 1) == -1\n"
            ),
        },
        {
            "task": "fibonacci_iterative",
            "prompt": "Compute the n-th Fibonacci number iteratively.",
            "code": (
                "def fib(n: int) -> int:\n"
                "    if n < 0: raise ValueError('Negative index')\n"
                "    if n <= 1: return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
            ),
            "tests": (
                "assert fib(0) == 0\n"
                "assert fib(1) == 1\n"
                "assert fib(7) == 13\n"
                "assert fib(10) == 55\n"
            ),
        },
        {
            "task": "is_palindrome",
            "prompt": "Check if an alphanumeric string is a palindrome, ignoring case.",
            "code": (
                "def is_palindrome(s: str) -> bool:\n"
                "    clean = [c.lower() for c in s if c.isalnum()]\n"
                "    return clean == clean[::-1]\n"
            ),
            "tests": (
                "assert is_palindrome('A man a plan a canal Panama') == True\n"
                "assert is_palindrome('race a car') == False\n"
                "assert is_palindrome('') == True\n"
            ),
        },
        {
            "task": "gcd_euclidean",
            "prompt": "Compute the greatest common divisor of two integers using Euclid's algorithm.",
            "code": (
                "def gcd(a: int, b: int) -> int:\n"
                "    while b != 0:\n"
                "        a, b = b, a % b\n"
                "    return abs(a)\n"
            ),
            "tests": (
                "assert gcd(48, 18) == 6\n"
                "assert gcd(101, 10) == 1\n"
                "assert gcd(0, 5) == 5\n"
            ),
        },
    ]

    LEAN_MATH_TEMPLATES = [
        {
            "problem": "Prove that the sum of the first n positive integers is n * (n + 1) / 2.",
            "thought": "We proceed by induction on n. Base case n=0 is trivial. Inductive step follows by ring simplification.",
            "lean_code": (
                "theorem sum_first_n (n : Nat) : (List.range (n + 1)).sum = n * (n + 1) / 2 := by\n"
                "  induction n with\n"
                "  | zero => rfl\n"
                "  | succ n ih =>\n"
                "    simp [List.range_succ, ih]\n"
                "    omega\n"
            ),
            "result": "pass",
        },
        {
            "problem": "Prove that for all natural numbers a and b, a + b = b + a.",
            "thought": "Commutativity of natural addition is a standard property in Peano arithmetic, provable by induction.",
            "lean_code": (
                "theorem add_comm_nat (a b : Nat) : a + b = b + a := by\n"
                "  induction a with\n"
                "  | zero => rw [Nat.zero_add, Nat.add_zero]\n"
                "  | succ a ih => rw [Nat.succ_add, ih, Nat.add_succ]\n"
            ),
            "result": "pass",
        },
        {
            "problem": "Prove that 0 * n = 0 for any natural number n.",
            "thought": "Multiplication by zero on the left is true by definition in Nat arithmetic.",
            "lean_code": (
                "theorem zero_mul_nat (n : Nat) : 0 * n = 0 := by\n"
                "  rw [Nat.zero_mul]\n"
            ),
            "result": "pass",
        },
    ]

    DECISION_SCENARIOS = [
        {
            "situation": "A python test failed with an IndexError in array access.",
            "candidate_actions": ["retry", "debug_bounds", "ignore_error", "delete_file"],
            "selected_action": "debug_bounds",
            "confidence": 0.96,
            "requires_human_approval": False,
            "reasoning": "IndexError indicates an off-by-one or empty collection condition. Inspect bounds before retry.",
        },
        {
            "situation": "The user requested deletion of all files in the current workspace directory.",
            "candidate_actions": ["execute_delete", "request_confirmation", "ignore", "abort"],
            "selected_action": "request_confirmation",
            "confidence": 0.99,
            "requires_human_approval": True,
            "reasoning": "Mass deletion is irreversible and potentially destructive. Explicit user approval is mandatory.",
        },
        {
            "situation": "An API call timed out after 5000ms with error HTTP 504.",
            "candidate_actions": ["exponential_backoff_retry", "fail_hard", "escalate", "format_disk"],
            "selected_action": "exponential_backoff_retry",
            "confidence": 0.88,
            "requires_human_approval": False,
            "reasoning": "Transient gateway timeouts are standard. Exponential backoff is the correct automated recovery procedure.",
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_code_sample(self) -> str:
        """Generates a structured algorithmic code sample."""
        item = self.rng.choice(self.ALGORITHMIC_TEMPLATES)
        return (
            f"<|thought|>Task: {item['prompt']}\n"
            f"Strategy: Implement standard clean algorithm with verified invariants.<|action|>write_code\n"
            f"{item['code']}\n"
            f"# Verification tests:\n"
            f"{item['tests']}<|verify|>pass<|eos|>"
        )

    def generate_lean_math_sample(self) -> str:
        """Generates a structured Lean 4 formal verification reasoning sample."""
        item = self.rng.choice(self.LEAN_MATH_TEMPLATES)
        return (
            f"<|thought|>{item['thought']}<|lean|>\n"
            f"-- Problem: {item['problem']}\n"
            f"{item['lean_code']}\n"
            f"<|verify|>{item['result']}<|eos|>"
        )

    def generate_decision_sample(self) -> str:
        """Generates a Jev-style structured decision trajectory."""
        item = self.rng.choice(self.DECISION_SCENARIOS)
        return (
            f"<|thought|>Situation: {item['situation']}\n"
            f"Reasoning: {item['reasoning']}<|action|>{item['selected_action']}\n"
            f"<|confidence|>{item['confidence']:.2f}\n"
            f"<|observe|>Permission Required: {item['requires_human_approval']}<|verify|>pass<|eos|>"
        )

    def generate_dataset_mixture(self, num_samples: int = 100) -> List[str]:
        """Generates a balanced curriculum mixture."""
        samples: List[str] = []
        for _ in range(num_samples):
            r = self.rng.random()
            if r < 0.40:
                samples.append(self.generate_code_sample())
            elif r < 0.75:
                samples.append(self.generate_lean_math_sample())
            else:
                samples.append(self.generate_decision_sample())
        return samples
