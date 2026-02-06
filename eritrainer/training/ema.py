"""Exponential moving average helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import torch


class EMAMode(str, Enum):
    OFF = "off"
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class EMAModule:
    """Tracks EMA weights for a single module."""

    module: torch.nn.Module
    decay: float = 0.999
    shadow: Optional[dict] = None

    def initialize(self) -> None:
        self.shadow = {k: v.detach().clone() for k, v in self.module.state_dict().items()}

    def update(self) -> None:
        if self.shadow is None:
            self.initialize()
        for name, param in self.module.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: torch.nn.Module) -> None:
        if self.shadow is None:
            return
        module.load_state_dict(self.shadow, strict=False)


@dataclass
class EMAModel:
    """High-level EMA container."""

    modules: Iterable[torch.nn.Module]
    decay: float = 0.999

    def __post_init__(self) -> None:
        self._ema_modules = [EMAModule(m, decay=self.decay) for m in self.modules]

    def update(self) -> None:
        for ema_module in self._ema_modules:
            ema_module.update()

    def copy_to(self, modules: Iterable[torch.nn.Module]) -> None:
        for ema_module, module in zip(self._ema_modules, modules):
            ema_module.copy_to(module)
