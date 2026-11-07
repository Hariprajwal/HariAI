"""Unit tests for the HariAI Formal Verification Kernel."""

import unittest
from hari_ai.verification.lean_kernel import (
    auto_verify,
    is_lean_available,
    verify_lean_code,
    verify_python_asserts,
)


class TestVerificationKernel(unittest.TestCase):
    def test_python_assert_verification_pass(self):
        code = "def add(a, b): return a + b"
        tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        result = verify_python_asserts(code, tests)
        self.assertTrue(result.passed)
        self.assertEqual(result.engine, "python_assert")

    def test_python_assert_verification_failure(self):
        code = "def add(a, b): return a - b"  # Bug
        tests = "assert add(2, 3) == 5"
        result = verify_python_asserts(code, tests)
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.stderr)

    def test_auto_verify_routing(self):
        code = "def square(x): return x * x"
        tests = "assert square(4) == 16"
        res = auto_verify("python", code, tests)
        self.assertTrue(res.passed)

    def test_lean_availability_reporting(self):
        # Gracefully reports boolean without crashing regardless of whether Lean is on host
        available = is_lean_available()
        self.assertIsInstance(available, bool)


if __name__ == "__main__":
    unittest.main()
