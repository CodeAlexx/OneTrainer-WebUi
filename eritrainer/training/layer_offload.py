"""Compatibility layer for native eritrainer layer offloading."""

from __future__ import annotations

from eritrainer.memory.conductor import LayerOffloadConductor as _BaseConductor

import torch
import torch.nn as nn


class LayerOffloadConductor(_BaseConductor):
    """Backwards-compatible conductor entrypoint used by training tests."""

    def __init__(self, module: nn.Module, config) -> None:
        self.config = config
        gradient_checkpointing = str(getattr(config, "gradient_checkpointing", "off")).strip().lower()
        activation_offload_enabled = (
            bool(getattr(config, "enable_activation_offloading", False))
            and gradient_checkpointing == "cpu_offloaded"
        )
        super().__init__(
            module=module,
            train_device=torch.device(getattr(config, "train_device", "cuda")),
            temp_device=torch.device(getattr(config, "temp_device", "cpu")),
            layer_offload_fraction=float(getattr(config, "layer_offload_fraction", 0.0)),
            offload_activations=activation_offload_enabled,
            enable_async=bool(getattr(config, "enable_async_offloading", False)),
        )
