"""Exponential moving average helpers.

Parity with OneTrainer's EMAModuleWrapper: warmup decay, cross-device
shadow storage, temp store/restore for sampling during training.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import torch


class EMAMode(str, Enum):
    """EMA device placement mode."""

    OFF = "off"
    CPU = "cpu"
    GPU = "gpu"


# --------------------------------------------------------------------------- #
# Parameter-level EMA wrapper (OneTrainer EMAModuleWrapper parity)
# --------------------------------------------------------------------------- #


class EMAParameterWrapper:
    """Tracks EMA for a list of parameters with warmup and cross-device support.

    This matches OneTrainer's EMAModuleWrapper which operates on raw
    ``nn.Parameter`` iterables rather than full module state dicts.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ) -> None:
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters: list[torch.Tensor] | None = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step: int) -> float:
        """Warmup decay: ramp from ~0.09 to target ``decay`` over early steps."""
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay,
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int) -> None:
        """Update EMA parameters. Handles cross-device (CPU shadow, GPU params)."""
        parameters = list(parameters)
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_p, p in zip(self.ema_parameters, parameters, strict=True):
                if p.requires_grad:
                    if ema_p.device == p.device:
                        ema_p.add_(one_minus_decay * (p - ema_p))
                    else:
                        # Cross-device: in-place to save memory
                        p_copy = p.detach().to(ema_p.device)
                        p_copy.sub_(ema_p)
                        p_copy.mul_(one_minus_decay)
                        ema_p.add_(p_copy)
                        del p_copy

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Move EMA parameters to a device/dtype."""
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        """Copy EMA weights into model parameters for sampling.

        If ``store_temp`` is True, the original weights are saved to CPU
        so they can be restored after sampling via :meth:`restore_temp`.
        """
        if store_temp:
            self.temp_stored_parameters = [p.detach().cpu() for p in parameters]
        parameters = list(parameters)
        for ema_p, p in zip(self.ema_parameters, parameters, strict=True):
            p.data.copy_(ema_p.to(p.device).data)

    def restore_temp(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore original model parameters saved by :meth:`copy_ema_to`."""
        if self.temp_stored_parameters is None:
            return
        for temp_p, p in zip(self.temp_stored_parameters, parameters, strict=True):
            p.data.copy_(temp_p.data)
        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", self.ema_parameters)
        self.to(self.device)

    def state_dict(self) -> dict:
        """Save EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }


# --------------------------------------------------------------------------- #
# Module-level EMA (original Serenity API, kept for backward compat)
# --------------------------------------------------------------------------- #


@dataclass
class EMAModule:
    """Tracks EMA weights for a single module via state_dict."""

    module: torch.nn.Module
    decay: float = 0.999
    shadow: dict[str, torch.Tensor] | None = None

    def initialize(self) -> None:
        self.shadow = {k: v.detach().clone() for k, v in self.module.state_dict().items()}

    def update(self) -> None:
        if self.shadow is None:
            self.initialize()
        assert self.shadow is not None
        for name, param in self.module.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: torch.nn.Module) -> None:
        if self.shadow is None:
            return
        module.load_state_dict(self.shadow, strict=False)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.shadow is None:
            self.initialize()
        assert self.shadow is not None
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items() if torch.is_tensor(v)}


@dataclass
class EMAModel:
    """High-level EMA container for multiple modules."""

    modules: Iterable[torch.nn.Module]
    decay: float = 0.999
    _ema_modules: list[EMAModule] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ema_modules = [EMAModule(m, decay=self.decay) for m in self.modules]

    def update(self) -> None:
        for ema_module in self._ema_modules:
            ema_module.update()

    def copy_to(self, modules: Iterable[torch.nn.Module]) -> None:
        for ema_module, module in zip(self._ema_modules, modules):
            ema_module.copy_to(module)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": float(self.decay),
            "modules": [ema_module.state_dict() for ema_module in self._ema_modules],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        modules_state = state_dict.get("modules") if isinstance(state_dict, dict) else None
        if not isinstance(modules_state, list):
            return
        for ema_module, module_state in zip(self._ema_modules, modules_state):
            if not isinstance(module_state, dict):
                continue
            ema_module.load_state_dict(module_state)


__all__ = [
    "EMAMode",
    "EMAParameterWrapper",
    "EMAModule",
    "EMAModel",
]
