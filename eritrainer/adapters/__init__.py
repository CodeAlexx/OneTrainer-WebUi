"""Adapter factory and registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping

from eritrainer.adapters.base import AdapterProtocol


class AdapterType(str, Enum):
    LORA = "lora"
    DORA = "dora"
    LOKR = "lokr"
    LOHA = "loha"
    LOCON = "locon"


@dataclass
class BaseAdapter(AdapterProtocol):
    adapter_type: AdapterType
    rank: int
    alpha: float
    model_type: str

    def apply(self, module):  # pragma: no cover - placeholder
        return module


class LoRAAdapter(BaseAdapter):
    pass


class DoRAAdapter(BaseAdapter):
    pass


class LoKrAdapter(BaseAdapter):
    pass


class LoHaAdapter(BaseAdapter):
    pass


class LoConAdapter(BaseAdapter):
    pass


ADAPTER_REGISTRY: Dict[AdapterType, type[BaseAdapter]] = {
    AdapterType.LORA: LoRAAdapter,
    AdapterType.DORA: DoRAAdapter,
    AdapterType.LOKR: LoKrAdapter,
    AdapterType.LOHA: LoHaAdapter,
    AdapterType.LOCON: LoConAdapter,
}


def create_adapter(
    adapter_type: AdapterType | str,
    rank: int,
    alpha: float,
    model_type: str,
) -> BaseAdapter:
    adapter_enum = AdapterType(adapter_type) if not isinstance(adapter_type, AdapterType) else adapter_type
    adapter_cls = ADAPTER_REGISTRY[adapter_enum]
    return adapter_cls(adapter_type=adapter_enum, rank=rank, alpha=alpha, model_type=model_type)


def detect_adapter_type(state_dict: Mapping[str, object]) -> AdapterType:
    keys = " ".join(state_dict.keys())
    if "lokr" in keys:
        return AdapterType.LOKR
    if "hada" in keys:
        return AdapterType.LOHA
    if "dora" in keys:
        return AdapterType.DORA
    return AdapterType.LORA


__all__ = [
    "AdapterType",
    "ADAPTER_REGISTRY",
    "create_adapter",
    "detect_adapter_type",
]
