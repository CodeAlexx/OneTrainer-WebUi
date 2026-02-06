"""Adapter factory and registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from eritrainer.adapters.base import AdapterProtocol
from eritrainer.training.lycoris_manager import AdapterConfig as LyCORISAdapterConfig
from eritrainer.training.lycoris_manager import AdapterType as LyCORISAdapterType
from eritrainer.training.lycoris_manager import LyCORISManager


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
    _manager: LyCORISManager | None = field(default=None, init=False, repr=False)

    def _ensure_manager(self) -> LyCORISManager:
        if self._manager is None:
            lycoris_type = LyCORISAdapterType(str(self.adapter_type.value))
            self._manager = LyCORISManager(
                LyCORISAdapterConfig(
                    adapter_type=lycoris_type,
                    rank=int(self.rank),
                    alpha=float(self.alpha),
                ),
                model_type=self.model_type,
            )
        return self._manager

    def apply(self, module_or_pipeline):
        manager = self._ensure_manager()
        return manager.apply(module_or_pipeline)

    def prepare_optimizer_params(self, lr: float | None = None):
        manager = self._ensure_manager()
        return manager.prepare_optimizer_params(lr)

    def save_weights(self, output_path: str, dtype=None):
        manager = self._ensure_manager()
        return manager.save_weights(output_path, dtype=dtype)


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


ADAPTER_REGISTRY: dict[AdapterType, type[BaseAdapter]] = {
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
