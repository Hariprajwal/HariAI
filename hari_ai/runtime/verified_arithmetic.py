"""Exact verification for a deliberately narrow, unambiguous chat task.

This is a deterministic check, not a learned confidence estimate. It only
checks complete integer answers to single integer operations; it abstains on
additional claims, ambiguous prompts, truncation and conversation context.
"""
from __future__ import annotations

import re
from typing import Optional

_QUESTION = re.compile(
    r'\s*(?:(?:what\s+is|calculate|compute|solve)\s+)?'
    r'(-?\d{1,4})\s*([+*\-])\s*(-?\d{1,4})\s*'
    r'(?:=\s*\?)?\??\.?\s*'
    r'(?:answer\s+with\s+only\s+the\s+number\.)?\s*', re.I)
_ANSWER = re.compile(r'(?:(?:the\s+)?answer\s+is\s+)?([+-]?\d{1,10})[.!]?', re.I)
_EQUATION = re.compile(r'(-?\d{1,4})\s*([+*\-])\s*(-?\d{1,4})\s*=\s*([+-]?\d{1,10})[.!]?')


def verify_arithmetic(question: str, answer: str) -> Optional[bool]:
    """Return correctness if the entire question and answer can be checked."""
    match = _QUESTION.fullmatch(question)
    if not match:
        return None
    a, op, b = int(match[1]), match[2], int(match[3])
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines and lines[0].casefold() in ('here is the answer:', 'the answer is:'):
        lines = lines[1:]
    if not 1 <= len(lines) <= 3:
        return None
    stated_values = []
    for line in lines:
        number = _ANSWER.fullmatch(line)
        equation = _EQUATION.fullmatch(line)
        if number:
            stated_values.append(int(number[1]))
        elif equation and (int(equation[1]), equation[2], int(equation[3])) == (a, op, b):
            stated_values.append(int(equation[4]))
        else:
            return None
    if len(set(stated_values)) != 1:
        return None
    stated = stated_values[0]
    expected = a+b if op == '+' else a-b if op == '-' else a*b
    return stated == expected
