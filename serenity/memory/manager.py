"""Memory manager coordinating native serenity memory policies."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

import torch

from serenity.core.enums import GradientCheckpointingMethod
from serenity.memory.strategy import LayerOffloadStrategy, MemoryConfig, MemoryStrategy, NoOpMemoryStrategy

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from serenity.core.interfaces import BaseModel


@dataclass
class MemoryManager:
    config: Any
    model: "BaseModel"

    def __post_init__(self) -> None:
        self.train_device = getattr(self.config, "train_device", "cuda")
        self.temp_device = getattr(self.config, "temp_device", "cpu")
        self._strategy: MemoryStrategy = self._create_strategy()

    def _create_strategy(self) -> MemoryStrategy:
        strategy_name = str(getattr(self.config, "memory_strategy", "auto")).strip().lower()
        if strategy_name in {"none", "off", "noop"}:
            return NoOpMemoryStrategy()

        should_use_layer_strategy = (
            self.use_layer_offloading
            or self.use_activation_offloading
            or strategy_name in {"auto", "layer_offload", "offload"}
        )
        if not should_use_layer_strategy:
            return NoOpMemoryStrategy()

        return LayerOffloadStrategy(
            MemoryConfig(
                strategy="layer_offload",
                layer_offload_fraction=float(getattr(self.config, "layer_offload_fraction", 0.0)),
                gradient_checkpointing=str(getattr(self.config, "gradient_checkpointing", "off")),
                enable_activation_offloading=bool(getattr(self.config, "enable_activation_offloading", False)),
                enable_async_offloading=bool(getattr(self.config, "enable_async_offloading", False)),
                train_device=str(self.train_device),
                temp_device=str(self.temp_device),
            )
        )

    @property
    def use_layer_offloading(self) -> bool:
        return getattr(self.config, "layer_offload_fraction", 0.0) > 0.0

    @property
    def use_gradient_checkpointing(self) -> bool:
        method = getattr(self.config, "gradient_checkpointing", "off")
        return str(method).lower() != GradientCheckpointingMethod.OFF.value

    @property
    def use_activation_offloading(self) -> bool:
        method = getattr(self.config, "gradient_checkpointing", "off")
        return (
            bool(getattr(self.config, "enable_activation_offloading", False))
            and str(method).lower() == GradientCheckpointingMethod.CPU_OFFLOADED.value
        )

    def setup_optimizations(self) -> None:
        self._strategy.setup(self.model)

    def forward_context(self) -> AbstractContextManager:
        return self._strategy.forward_context() if self._strategy is not None else nullcontext()

    def cleanup(self) -> None:
        if self._strategy is not None:
            self._strategy.cleanup()

    @property
    def strategy(self) -> MemoryStrategy:
        return self._strategy

    def get_memory_usage(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            allocated = reserved = max_allocated = 0.0

        return {
            "allocated_gb": float(allocated),
            "cached_gb": float(reserved),
            "max_allocated_gb": float(max_allocated),
        }
