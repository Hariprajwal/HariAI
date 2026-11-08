"""Local chat and research inference, loaded only when requested."""
from importlib import import_module

_EXPORTS = {
    'HardwareBackend': ('hari_ai.runtime.engine', 'HardwareBackend'),
    'GenerationConfig': ('hari_ai.runtime.engine', 'GenerationConfig'),
    'detect_best_hardware': ('hari_ai.runtime.engine', 'detect_best_hardware'),
    'InferenceEngine': ('hari_ai.runtime.engine', 'InferenceEngine'),
    'HariAI': ('hari_ai.runtime.engine', 'HariAI'),
    'ChatEngine': ('hari_ai.runtime.chat', 'ChatEngine'),
    'ChatMessage': ('hari_ai.runtime.chat', 'ChatMessage'),
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

