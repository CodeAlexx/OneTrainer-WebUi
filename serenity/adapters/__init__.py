"""Adapter factory and registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from serenity.adapters.base import AdapterProtocol
from serenity.training.lycoris_manager import AdapterConfig as LyCORISAdapterConfig
from serenity.training.lycoris_manager import AdapterType as LyCORISAdapterType
from serenity.training.lycoris_manager import LyCORISManager


class AdapterType(str, Enum):
    LORA = "lora"
    DORA = "dora"
    LOKR = "lokr"
    LOHA = "loha"
    LOCON = "locon"
    IA3 = "ia3"
    OFT = "oft"
    BOFT = "boft"
    DIAG_OFT = "diag_oft"
    GLORA = "glora"
    DYLORA = "dylora"
    FULL = "full"


_ADAPTER_ALIASES: dict[str, str] = {
    "diag-oft": "diag_oft",
    "diagoft": "diag_oft",
    "oft_2": "oft",
    "full_finetune": "full",
    "full_fine_tune": "full",
    "fine_tune": "full",
    "finetune": "full",
}


def _normalize_adapter_type(adapter_type: AdapterType | str) -> AdapterType:
    if isinstance(adapter_type, AdapterType):
        return adapter_type
    normalized = str(adapter_type).strip().lower().replace("-", "_")
    normalized = _ADAPTER_ALIASES.get(normalized, normalized)
    return AdapterType(normalized)


@dataclass
class BaseAdapter(AdapterProtocol):
    adapter_type: AdapterType
    rank: int
    alpha: float
    model_type: str
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)
    conv_rank: int | None = None
    conv_alpha: float | None = None
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    factor: int = 2
    decompose_both: bool = False
    use_tucker: bool = False
    full_matrix: bool = False
    weight_decompose: bool = False
    dora_on_output: bool = True
    rs_lora: bool = False
    block_size: int = 4
    constraint: float = 0.0
    rescaled: bool = False
    multiplier: float = 1.0
    _manager: Any = field(default=None, init=False, repr=False)
    _target_module: object | None = field(default=None, init=False, repr=False)

    def _ensure_manager(self) -> Any:
        if self._manager is not None:
            return self._manager

        if self.adapter_type in (AdapterType.LORA, AdapterType.DORA):
            from serenity.training.lora_manager import (
                LoRAManager,
                DoRAManager,
                AdapterConfig as NativeAdapterConfig,
            )

            native_config = NativeAdapterConfig(
                rank=int(self.rank),
                alpha=float(self.alpha),
                target_modules=list(self.target_modules),
                dropout=float(self.dropout),
                rs_lora=bool(self.rs_lora),
                multiplier=float(self.multiplier),
            )
            if self.adapter_type == AdapterType.DORA:
                self._manager = DoRAManager(native_config, model_type=self.model_type)
            else:
                self._manager = LoRAManager(native_config, model_type=self.model_type)
        else:
            # Exotic adapter types use LyCORIS
            lycoris_type = LyCORISAdapterType(str(self.adapter_type.value))
            self._manager = LyCORISManager(
                LyCORISAdapterConfig(
                    adapter_type=lycoris_type,
                    rank=int(self.rank),
                    alpha=float(self.alpha),
                    target_modules=list(self.target_modules),
                    conv_rank=self.conv_rank,
                    conv_alpha=self.conv_alpha,
                    dropout=float(self.dropout),
                    rank_dropout=float(self.rank_dropout),
                    module_dropout=float(self.module_dropout),
                    factor=int(self.factor),
                    decompose_both=bool(self.decompose_both),
                    use_tucker=bool(self.use_tucker),
                    full_matrix=bool(self.full_matrix),
                    weight_decompose=bool(self.weight_decompose),
                    dora_on_output=bool(self.dora_on_output),
                    rs_lora=bool(self.rs_lora),
                    block_size=int(self.block_size),
                    constraint=float(self.constraint),
                    rescaled=bool(self.rescaled),
                    multiplier=float(self.multiplier),
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
        # load_state_dict in LoRAManager already handles diffusers↔native conversion
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


class IA3Adapter(BaseAdapter):
    pass


class OFTAdapter(BaseAdapter):
    pass


class BOFTAdapter(BaseAdapter):
    pass


class DiagOFTAdapter(BaseAdapter):
    pass


class GLoRAAdapter(BaseAdapter):
    pass


class DyLoRAAdapter(BaseAdapter):
    pass


class FullAdapter(BaseAdapter):
    pass


ADAPTER_REGISTRY: dict[AdapterType, type[BaseAdapter]] = {
    AdapterType.LORA: LoRAAdapter,
    AdapterType.DORA: DoRAAdapter,
    AdapterType.LOKR: LoKrAdapter,
    AdapterType.LOHA: LoHaAdapter,
    AdapterType.LOCON: LoConAdapter,
    AdapterType.IA3: IA3Adapter,
    AdapterType.OFT: OFTAdapter,
    AdapterType.BOFT: BOFTAdapter,
    AdapterType.DIAG_OFT: DiagOFTAdapter,
    AdapterType.GLORA: GLoRAAdapter,
    AdapterType.DYLORA: DyLoRAAdapter,
    AdapterType.FULL: FullAdapter,
}


def create_adapter(
    adapter_type: AdapterType | str,
    rank: int,
    alpha: float,
    model_type: str,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    **kwargs,
) -> BaseAdapter:
    adapter_enum = _normalize_adapter_type(adapter_type)
    adapter_cls = ADAPTER_REGISTRY[adapter_enum]
    return adapter_cls(
        adapter_type=adapter_enum,
        rank=rank,
        alpha=alpha,
        model_type=model_type,
        dropout=dropout,
        target_modules=list(target_modules or kwargs.get("target_modules") or []),
        conv_rank=kwargs.get("conv_rank"),
        conv_alpha=kwargs.get("conv_alpha"),
        rank_dropout=float(kwargs.get("rank_dropout", 0.0)),
        module_dropout=float(kwargs.get("module_dropout", 0.0)),
        factor=int(kwargs.get("factor", 2)),
        decompose_both=bool(kwargs.get("decompose_both", False)),
        use_tucker=bool(kwargs.get("use_tucker", False)),
        full_matrix=bool(kwargs.get("full_matrix", False)),
        weight_decompose=bool(kwargs.get("weight_decompose", False)),
        dora_on_output=bool(kwargs.get("dora_on_output", True)),
        rs_lora=bool(kwargs.get("rs_lora", False)),
        block_size=int(kwargs.get("block_size", 4)),
        constraint=float(kwargs.get("constraint", 0.0)),
        rescaled=bool(kwargs.get("rescaled", False)),
        multiplier=float(kwargs.get("multiplier", 1.0)),
    )


def detect_adapter_type(state_dict: Mapping[str, object]) -> AdapterType:
    keys = " ".join(state_dict.keys())
    if "lokr" in keys:
        return AdapterType.LOKR
    if "diag_oft" in keys:
        return AdapterType.DIAG_OFT
    if "boft" in keys:
        return AdapterType.BOFT
    if "oft" in keys:
        return AdapterType.OFT
    if "hada" in keys:
        return AdapterType.LOHA
    if "dora" in keys:
        return AdapterType.DORA
    return AdapterType.LORA


def detect_rank(state_dict: Mapping[str, object], default: int = 16) -> int:
    """Best-effort rank detection from adapter tensors."""

    candidate_keys = (
        "lora_a",
        "lora_down",
        "lokr_w1",
        "lokr_w1_a",
        "hada_w1_a",
        "oft",
        "boft",
        "diag_oft",
        "diag-oft",
    )

    def _is_candidate(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in candidate_keys)

    for key, value in state_dict.items():
        if not _is_candidate(str(key)):
            continue
        if not torch.is_tensor(value):
            continue
        if value.ndim >= 2:
            first = int(value.shape[0])
            second = int(value.shape[1])
            if first > 0 and second > 0:
                return min(first, second)
            if first > 0:
                return first
        elif value.ndim == 1 and int(value.shape[0]) > 0:
            return int(value.shape[0])

    for value in state_dict.values():
        if torch.is_tensor(value) and value.ndim >= 2:
            first = int(value.shape[0])
            if first > 0:
                return first

    return int(default)


__all__ = [
    "AdapterType",
    "ADAPTER_REGISTRY",
    "create_adapter",
    "detect_adapter_type",
    "detect_rank",
]
