"""Compatibility layer for native eritrainer layer offloading."""

from __future__ import annotations

import torch
import torch.nn as nn

from eritrainer.memory.conductor import LayerOffloadConductor as _BaseConductor


class LayerOffloadConductor(_BaseConductor):
    """Backwards-compatible conductor entrypoint used by training tests."""

    def __init__(self, module: nn.Module, config) -> None:
        self.config = config
        super().__init__(
            module=module,
            train_device=torch.device(getattr(config, "train_device", "cuda")),
            temp_device=torch.device(getattr(config, "temp_device", "cpu")),
            layer_offload_fraction=float(getattr(config, "layer_offload_fraction", 0.0)),
            offload_activations=bool(getattr(config, "enable_activation_offloading", False)),
            enable_async=bool(getattr(config, "enable_async_offloading", False)),
        )
