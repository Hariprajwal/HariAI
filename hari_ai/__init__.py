"""HariAI by K R Hari Prajwal: local conversation and research models.

Public objects load on first access so GGUF chat does not initialize PyTorch
training components merely by importing the package.
"""
from importlib import import_module

from hari_ai.version import __version__

__author__ = "K R Hari Prajwal"
_EXPORTS = {
    'HariAI': ('hari_ai.runtime.engine', 'HariAI'),
    'Cluster': ('hari_ai.cluster.coordinator', 'Cluster'),
    'ChatEngine': ('hari_ai.runtime.chat', 'ChatEngine'),
    'SearchEngine': ('hari_ai.search.engine', 'SearchEngine'),
    'HariNanoModel': ('hari_ai.model.transformer', 'HariNanoModel'),
    'HariNanoConfig': ('hari_ai.model.config', 'HariNanoConfig'),
}
__all__ = [*_EXPORTS, '__version__', '__author__']


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
