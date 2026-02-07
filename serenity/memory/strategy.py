"""Memory strategy abstractions and native layer-offload strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from serenity.memory.checkpoint_layer import OffloadCheckpointLayer
from serenity.memory.conductor import LayerOffloadConductor


def _bool_from_config(config: Any, field_name: str, default: bool = False) -> bool:
    return bool(getattr(config, field_name, default))


def _float_from_config(config: Any, field_name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(config, field_name, default))
    except (TypeError, ValueError):
        return default


def _collect_layers(transformer: nn.Module) -> tuple[str, list[nn.Module]]:
    for attr in ("transformer_blocks", "blocks", "layers"):
        layers = getattr(transformer, attr, None)
        if isinstance(layers, nn.ModuleList):
            return attr, list(layers)
        if isinstance(layers, list) and all(isinstance(layer, nn.Module) for layer in layers):
            return attr, layers
    raise RuntimeError("Unable to locate transformer layer container for offloading.")


def _is_cpu_offloaded_checkpointing(config: Any) -> bool:
    value = str(getattr(config, "gradient_checkpointing", "off")).strip().lower()
    return value == "cpu_offloaded"


@dataclass
class MemoryConfig:
    strategy: str = "none"
    layer_offload_fraction: float = 0.0
    gradient_checkpointing: str = "off"
    enable_activation_offloading: bool = False
    enable_async_offloading: bool = False
    train_device: str = "cuda"
    temp_device: str = "cpu"


class MemoryStrategy(ABC):
    """Base strategy contract for memory optimizations."""

    @abstractmethod
    def setup(self, model: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_context(self) -> AbstractContextManager:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def estimate_vram_usage(self, batch_size: int, resolution: int) -> int:
        raise NotImplementedError


class NoOpMemoryStrategy(MemoryStrategy):
    def setup(self, model: Any) -> None:
        del model

    def forward_context(self) -> AbstractContextManager:
        return nullcontext()

    def cleanup(self) -> None:
        return

    def estimate_vram_usage(self, batch_size: int, resolution: int) -> int:
        del batch_size, resolution
        return 24 * 1024 * 1024 * 1024


class LayerOffloadStrategy(MemoryStrategy):
    """Layer offload strategy used by native Serenity paths."""

    def __init__(self, config: MemoryConfig | Any):
        self.config = config
        self.conductor: LayerOffloadConductor | None = None
        self._transformer: nn.Module | None = None
        self._layer_attr: str | None = None
        self._original_layers: list[nn.Module] = []

    def setup(self, model: Any) -> None:
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            self.conductor = None
            return

        layer_fraction = _float_from_config(self.config, "layer_offload_fraction", 0.0)
        offload_activations = (
            _bool_from_config(self.config, "enable_activation_offloading", False)
            and _is_cpu_offloaded_checkpointing(self.config)
        )
        if layer_fraction <= 0.0 and not offload_activations:
            self.conductor = None
            return

        if not 0.0 <= layer_fraction <= 1.0:
            raise ValueError("layer_offload_fraction must be in [0.0, 1.0]")

        layer_attr, layers = _collect_layers(transformer)
        self._transformer = transformer
        self._layer_attr = layer_attr
        self._original_layers = list(layers)

        conductor = LayerOffloadConductor(
            module=transformer,
            train_device=torch.device(getattr(self.config, "train_device", "cuda")),
            temp_device=torch.device(getattr(self.config, "temp_device", "cpu")),
            layer_offload_fraction=layer_fraction,
            offload_activations=offload_activations,
            enable_async=_bool_from_config(self.config, "enable_async_offloading", False),
        )
        for layer in layers:
            conductor.add_layer(layer)

        use_checkpointing = str(getattr(self.config, "gradient_checkpointing", "off")).strip().lower() != "off"
        wrapped_layers = nn.ModuleList(
            [
                OffloadCheckpointLayer(
                    layer=layer,
                    layer_idx=idx,
                    conductor=conductor,
                    use_checkpointing=use_checkpointing,
                )
                for (idx, layer) in enumerate(layers)
            ]
        )
        setattr(transformer, layer_attr, wrapped_layers)
        self.conductor = conductor

    def forward_context(self) -> AbstractContextManager:
        if self.conductor is None:
            return nullcontext()
        return self.conductor.forward_context()

    def cleanup(self) -> None:
        if self.conductor is not None:
            self.conductor.cleanup()
        if self._transformer is not None and self._layer_attr is not None and self._original_layers:
            setattr(self._transformer, self._layer_attr, nn.ModuleList(self._original_layers))
        self.conductor = None
        self._transformer = None
        self._layer_attr = None
        self._original_layers = []

    def estimate_vram_usage(self, batch_size: int, resolution: int) -> int:
        batch_component = max(batch_size, 1) * (resolution * resolution) * 8
        if self.conductor is None:
            return 24 * 1024 * 1024 * 1024 + batch_component
        offload_scale = max(0.1, 1.0 - _float_from_config(self.config, "layer_offload_fraction", 0.0))
        return int((24 * 1024 * 1024 * 1024) * offload_scale + batch_component)
