"""Fail-closed, task-scoped Open-JEV confidence for local chat answers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional

from hari_ai.runtime.calibration import evaluate_calibrator, platt_probability

DOMAIN = 'single_turn_integer_arithmetic_v1'
ENCODING = 'arithmetic_digits_v1'
SCORE_SEMANTICS = 'jev_choice_confidence_oriented_to_answer_correct_v1'
QUESTION = 'Is the stated numerical answer correct?'
OPTIONS = ['correct', 'incorrect']


def answer_probability_from_head(choice: str, confidence: float) -> float:
    """Interpret the Jev head as confidence in its chosen binary label."""
    if choice not in OPTIONS or not 0 <= confidence <= 1:
        raise ValueError('Invalid Open-JEV choice or confidence.')
    return confidence if choice == 'correct' else 1 - confidence


def stated_integer(question: str, answer: str, *, restrict_scope: bool = True) -> Optional[int]:
    """Read one complete integer answer, rejecting extra or unrelated claims."""
    parts = re.fullmatch(r'What is (\d{1,2}) ([+-]) (\d{1,2})\? Answer with only the number\.', question)
    if not parts:
        return None
    a, b = int(parts[1]), int(parts[3])
    if (str(a) != parts[1] or str(b) != parts[3] or
            (restrict_scope and (a > 29 or b > 29 or (parts[2] == '+' and a > b)))):
        return None
    answer = answer.strip()
    numeric = re.fullmatch(r'(?:(?:The )?answer is )?([+-]?\d+)[.!]?', answer, re.I)
    if numeric:
        value = int(numeric[1])
        return value if abs(value) <= 999 else None
    statement = re.fullmatch(r'(\d{1,2})\s*([+-])\s*(\d{1,2})\s*=\s*([+-]?\d+)[.!]?', answer)
    if statement and statement[1] == parts[1] and statement[2] == parts[2] and statement[3] == parts[3]:
        value = int(statement[4])
        return value if abs(value) <= 999 else None
    return None


def supports_pair(question: str, answer: str) -> bool:
    return stated_integer(question, answer) is not None


def arithmetic_state(question: str, answer: str) -> dict:
    """Expose input digits and stated result, without computing correctness."""
    parts = re.fullmatch(r'What is (\d{1,2}) ([+-]) (\d{1,2})\? Answer with only the number\.', question)
    value = stated_integer(question, answer, restrict_scope=False)
    if not parts or value is None:
        raise ValueError('Question or answer is not a single arithmetic statement.')
    a, b, magnitude = int(parts[1]), int(parts[3]), abs(value)
    return {'operation':'addition' if parts[2] == '+' else 'subtraction',
            'left':{'tens':str(a//10),'units':str(a%10)},
            'right':{'tens':str(b//10),'units':str(b%10)},
            'answer':{'sign':'negative' if value < 0 else 'nonnegative',
                      'hundreds':str(magnitude//100), 'tens':str((magnitude//10)%10),
                      'units':str(magnitude%10)}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class OpenJevAnswerConfidence:
    """Use a separately fitted Jev answer verifier only within its tested scope.

    Bundle files are trusted local artifacts; the metrics cannot establish that
    the source labels or claimed provenance were genuine. The release gate
    reduces accidental misuse and is not a guarantee under distribution shift.
    """
    def __init__(self, artifact_path: str, model_path: Path, system_prompt: str):
        from hari_ai.runtime.openjev import OpenJevAdapter
        path = Path(artifact_path).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if (data.get('format') != 'hariai_openjev_answer_calibration_v1' or data.get('domain') != DOMAIN
                or data.get('state_encoding') != ENCODING or data.get('score_semantics') != SCORE_SEMANTICS):
            raise ValueError('Unsupported answer-confidence artifact or task scope.')
        protocol = data.get('protocol', {})
        if protocol != {'backend':'local', 'max_tokens':192, 'temperature':0.2,
                        'top_p':0.9, 'repeat_penalty':1.1, 'history':'single_turn'}:
            raise ValueError('Calibration protocol does not match this runtime.')
        if data.get('source') != 'model_generated_independently_graded':
            raise ValueError('Calibration requires independently graded model-generated answers.')
        if data.get('problem_grouping') != 'commutative_addition_v1':
            raise ValueError('Calibration must prevent equivalent additions crossing splits.')
        if data.get('system_prompt_sha256') != hashlib.sha256(system_prompt.encode('utf-8')).hexdigest():
            raise ValueError('Calibration system prompt does not match.')
        if data.get('model_sha256') != sha256_file(model_path):
            raise ValueError('Calibration language-model weights do not match.')
        rows = data.get('rows', [])
        cal = [r for r in rows if r.get('split') == 'calibration']
        test = [r for r in rows if r.get('split') == 'test']
        if len(cal) + len(test) != len(rows):
            raise ValueError('Unsupported calibration split.')
        result = evaluate_calibrator(cal, test, predeclared=data.get('predeclared_split') is True)
        if not result['eligible']:
            raise ValueError('Calibration evaluation did not pass: ' + '; '.join(result['issues']))
        for key in ('slope', 'intercept'):
            value = data.get(key)
            if not isinstance(value, (int,float)) or not math.isclose(value,result[key],rel_tol=1e-6,abs_tol=1e-8):
                raise ValueError('Calibration parameters do not match evaluation.')
        filename = data.get('verifier_file', '')
        verifier_path = (path.parent / filename).resolve()
        if not filename or verifier_path.parent != path.parent:
            raise ValueError('Verifier must be stored beside its calibration artifact.')
        if data.get('verifier_sha256') != sha256_file(verifier_path):
            raise ValueError('Open-JEV checkpoint hash does not match the calibration artifact.')
        self.adapter = OpenJevAdapter(str(verifier_path))
        if self.adapter.checkpoint_scope != DOMAIN or self.adapter.state_encoding != ENCODING:
            raise ValueError('Open-JEV checkpoint was not trained for this answer-confidence task.')
        self.slope, self.intercept = result['slope'], result['intercept']
        self.test_count = result['test']['count']

    def estimate(self, question: str, answer: str, *, earlier_turns: int, finish_reason: str) -> Optional[float]:
        if earlier_turns or finish_reason != 'stop' or not supports_pair(question, answer):
            return None
        result = self.adapter.decide(arithmetic_state(question, answer), QUESTION, OPTIONS)
        raw = answer_probability_from_head(result['choice'], float(result['confidence_raw']))
        return platt_probability(raw, self.slope, self.intercept)
