"""HariAI Dream Engine package for continual learning."""

from hari_ai.dream.controller import (
    SystemState,
    SleepTrigger,
    DreamCycleResult,
    DreamEngineController,
)
from hari_ai.dream.adapter import (
    LoRALinear,
    inject_lora_adapters,
    remove_lora_adapters,
    merge_lora_adapters,
    DreamLoRAEngine,
)

__all__ = [
    "SystemState",
    "SleepTrigger",
    "DreamCycleResult",
    "DreamEngineController",
    "LoRALinear",
    "inject_lora_adapters",
    "remove_lora_adapters",
    "merge_lora_adapters",
    "DreamLoRAEngine",
]

