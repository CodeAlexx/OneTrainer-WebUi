"""LoRA loading, merging, and online application for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.lora.adapters import (
    LoHAAdapter,
    LoKRAdapter,
    LoRAAdapter,
    OFTAdapter,
    WeightAdapterBase,
    create_adapter,
    detect_adapter_type,
)
from serenity.inference.lora.loader import detect_lora_type, load_lora
from serenity.inference.lora.merge import merge_lora_into_model, unmerge_lora_from_model
from serenity.inference.lora.online import apply_online_lora, remove_online_lora

__all__ = [
    "LoHAAdapter",
    "LoKRAdapter",
    "LoRAAdapter",
    "OFTAdapter",
    "WeightAdapterBase",
    "apply_online_lora",
    "create_adapter",
    "detect_adapter_type",
    "detect_lora_type",
    "load_lora",
    "merge_lora_into_model",
    "remove_online_lora",
    "unmerge_lora_from_model",
]
