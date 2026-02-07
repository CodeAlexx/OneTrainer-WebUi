"""CREPA (Cross-frame REPresentation Alignment) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrepaConfig:
    enabled: bool = False
    crepa_lambda: float = 0.5
    scheduler_type: str = "constant"  # constant | linear
    warmup_steps: int = 0
    decay_steps: int = 0
    lambda_end: float = 0.0
    cutoff_step: Optional[int] = None
    threshold_mode: str = "permanent"  # permanent | temporary
    adjacent_distance: int = 1
    block_index: int = 0
    encoder_name: str = "dinov2_vitg14"
    use_backbone_features: bool = False


class CrepaScheduler:
    def __init__(self, config: CrepaConfig, max_train_steps: int) -> None:
        self.config = config
        self.max_train_steps = max_train_steps
        self._cutoff_triggered = False

    def _apply_warmup(self, step: int, value: float) -> float:
        if self.config.warmup_steps <= 0:
            return value
        if step <= 0:
            return 0.0
        if step >= self.config.warmup_steps:
            return value
        return value * (step / float(self.config.warmup_steps))

    def _apply_cutoff(self, step: int, value: float) -> float:
        if self.config.cutoff_step is None:
            return value
        if self._cutoff_triggered and self.config.threshold_mode == "permanent":
            return 0.0
        if step >= self.config.cutoff_step:
            if self.config.threshold_mode == "permanent":
                self._cutoff_triggered = True
                return 0.0
            return 0.0
        return value

    def get_weight(self, step: int) -> float:
        if not self.config.enabled:
            return 0.0

        if self.config.scheduler_type == "linear" and self.config.decay_steps > 0:
            progress = min(max(step, 0), self.config.decay_steps) / float(self.config.decay_steps)
            value = self.config.crepa_lambda + (self.config.lambda_end - self.config.crepa_lambda) * progress
        else:
            value = self.config.crepa_lambda

        value = self._apply_warmup(step, value)
        value = self._apply_cutoff(step, value)
        return float(max(value, 0.0))

    def is_cutoff(self) -> bool:
        return self._cutoff_triggered


class CrepaRegularizer:
    def __init__(
        self,
        config: CrepaConfig,
        device: torch.device,
        hidden_size: Optional[int] = None,
        max_train_steps: int = 0,
    ) -> None:
        self.config = config
        self.device = device
        self.hidden_size = hidden_size
        self.projector: Optional[nn.Module] = None
        self.encoder_dim: Optional[int] = None
        self.scheduler = CrepaScheduler(config, max_train_steps=max_train_steps or 1)

    def attach_to_model(self, model: nn.Module) -> None:
        if not self.config.enabled:
            return
        if self.hidden_size is None:
            raise ValueError("hidden_size required to attach CREPA projector")

        self.encoder_dim = self.hidden_size
        self.projector = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        model.crepa_projector = self.projector

    def wants_hidden_states(self) -> bool:
        return bool(self.config.enabled)

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            return hidden_states

        if hidden_states.ndim == 4:
            b, t, p, d = hidden_states.shape
            flat = hidden_states.reshape(b * t * p, d)
            proj = self.projector(flat)
            return proj.reshape(b, t, p, d)

        if hidden_states.ndim == 3:
            b, s, d = hidden_states.shape
            flat = hidden_states.reshape(b * s, d)
            proj = self.projector(flat)
            return proj.reshape(b, s, d)

        return self.projector(hidden_states)

    def compute_loss(
        self,
        hidden_states: Optional[torch.Tensor],
        frame_features: Optional[torch.Tensor],
        step: int,
    ) -> tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.config.enabled or hidden_states is None:
            return None, {}

        weight = self.scheduler.get_weight(step)
        if weight <= 0.0:
            return None, {"crepa_weight": weight}

        projected = self._project_hidden_states(hidden_states)

        if projected.ndim < 4:
            return None, {"crepa_weight": weight}

        # Penalize differences between adjacent frames
        diff = projected[:, 1:] - projected[:, :-1]
        loss = torch.mean(diff * diff)
        return weight * loss, {"crepa_weight": weight, "crepa_loss": float(loss.detach().cpu())}
