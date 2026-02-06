"""Adapter factory and registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

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
    dropout: float = 0.0
    _manager: LyCORISManager | None = field(default=None, init=False, repr=False)
    _target_module: object | None = field(default=None, init=False, repr=False)

    def _ensure_manager(self) -> LyCORISManager:
        if self._manager is None:
            lycoris_type = LyCORISAdapterType(str(self.adapter_type.value))
            self._manager = LyCORISManager(
                LyCORISAdapterConfig(
                    adapter_type=lycoris_type,
                    rank=int(self.rank),
                    alpha=float(self.alpha),
                    dropout=float(self.dropout),
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

    # ---------------------------------------------------------------------
    # Compatibility methods for the historical native trainer API.
    # ---------------------------------------------------------------------
    def inject(self, module):
        self._target_module = module
        self.apply(module)
        return module

    def get_trainable_params(self):
        manager = self._ensure_manager()
        param_groups = manager.prepare_optimizer_params(lr=None)
        for group in param_groups:
            for param in group.get("params", []):
                yield param

    def save(self, output_path: str, dtype=None):
        return self.save_weights(output_path, dtype=dtype)

    def load(self, input_path: str):
        manager = self._ensure_manager()
        path = Path(input_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        manager.load_state_dict(state_dict, strict=False)

    def merge(self, model=None):
        manager = self._ensure_manager()
        manager.merge_to(weight=1.0, precise=False)
        return model if model is not None else self._target_module

    def unmerge(self, model=None):
        manager = self._ensure_manager()
        manager.restore()
        return model if model is not None else self._target_module

    @property
    def is_injected(self) -> bool:
        manager = self._ensure_manager()
        return manager.is_attached()

    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.get_trainable_params())

    def memory_footprint(self) -> int:
        return sum(param.numel() * param.element_size() for param in self.get_trainable_params())

    def to(self, device, dtype=None):
        manager = self._ensure_manager()
        network = getattr(manager, "_network", None)
        if network is None:
            return self
        if dtype is not None:
            network.to(device=device, dtype=dtype)
        else:
            network.to(device=device)
        return self


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
    dropout: float = 0.0,
) -> BaseAdapter:
    adapter_enum = AdapterType(adapter_type) if not isinstance(adapter_type, AdapterType) else adapter_type
    adapter_cls = ADAPTER_REGISTRY[adapter_enum]
    return adapter_cls(
        adapter_type=adapter_enum,
        rank=rank,
        alpha=alpha,
        model_type=model_type,
        dropout=dropout,
    )


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
