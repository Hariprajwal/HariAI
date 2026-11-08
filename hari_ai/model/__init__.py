"""Research architectures, loaded on demand."""
from importlib import import_module

_EXPORTS = {
    **{name: ('hari_ai.model.config', name) for name in ('HariNanoConfig', 'UAIANanoConfig')},
    **{name: ('hari_ai.model.layers', name) for name in
       ('RMSNorm', 'RotaryEmbedding', 'GroupedQueryAttention', 'SwiGLU', 'apply_rotary_emb')},
    **{name: ('hari_ai.model.transformer', name) for name in
       ('StructuredDecisionHead', 'HariTransformerBlock', 'UAIATransformerBlock',
        'HariNanoModel', 'UAIANanoModel')},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

