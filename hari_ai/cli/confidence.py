"""Shared, explicitly labelled confidence text for both terminal interfaces."""
from __future__ import annotations


def confidence_lines(engine) -> list[str]:
    if not engine.settings.show_confidence or not engine.last_reply_succeeded:
        return []
    lines = []
    value = engine.last_answer_confidence
    source = getattr(engine, 'last_confidence_source', None) or 'arithmetic only'
    likelihood = getattr(engine, 'last_token_likelihood', None)
    if value is not None:
        lines.append(f'Confidence: {value:.0%} ({source})')
    if likelihood is not None:
        lines.append(f'Token likelihood: {likelihood:.0%}')
    elif value is None:
        lines.append('Token likelihood: unavailable')
    return lines
