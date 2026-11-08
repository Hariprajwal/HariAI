"""Dream research tools, with training components loaded on demand."""
from importlib import import_module

_EXPORTS = {
    **{name: ('hari_ai.dream.controller', name) for name in
       ('SystemState', 'SleepTrigger', 'DreamCycleResult', 'DreamEngineController')},
    **{name: ('hari_ai.dream.adapter', name) for name in
       ('LoRALinear', 'inject_lora_adapters', 'remove_lora_adapters',
        'merge_lora_adapters', 'DreamLoRAEngine')},
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

