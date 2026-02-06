"""Compatibility layer for native eritrainer layer offloading."""

from __future__ import annotations

from eritrainer.memory.conductor import LayerOffloadConductor as _BaseConductor

import torch
import torch.nn as nn


class LayerOffloadConductor(_BaseConductor):
    """Backwards-compatible conductor entrypoint used by training tests."""

    def __init__(
        self,
        module: nn.Module,
        config=None,
        *,
        train_device: torch.device | str | None = None,
        temp_device: torch.device | str | None = None,
        layer_offload_fraction: float | None = None,
        offload_activations: bool | None = None,
        enable_async: bool | None = None,
    ) -> None:
        self.config = config

        if config is not None and train_device is None and temp_device is None:
            gradient_checkpointing = str(getattr(config, "gradient_checkpointing", "off")).strip().lower()
            activation_offload_enabled = (
                bool(getattr(config, "enable_activation_offloading", False))
                and gradient_checkpointing == "cpu_offloaded"
            )
            resolved_train_device = torch.device(getattr(config, "train_device", "cuda"))
            resolved_temp_device = torch.device(getattr(config, "temp_device", "cpu"))
            resolved_layer_offload_fraction = float(getattr(config, "layer_offload_fraction", 0.0))
            resolved_offload_activations = activation_offload_enabled
            resolved_enable_async = bool(getattr(config, "enable_async_offloading", False))
        else:
            resolved_train_device = torch.device(train_device or getattr(config, "train_device", "cuda"))
            resolved_temp_device = torch.device(temp_device or getattr(config, "temp_device", "cpu"))
            resolved_layer_offload_fraction = float(
                layer_offload_fraction
                if layer_offload_fraction is not None
                else getattr(config, "layer_offload_fraction", 0.0)
            )
            resolved_offload_activations = bool(
                offload_activations
                if offload_activations is not None
                else getattr(config, "enable_activation_offloading", False)
            )
            resolved_enable_async = bool(
                enable_async
                if enable_async is not None
                else getattr(config, "enable_async_offloading", False)
            )

        super().__init__(
            module=module,
            train_device=resolved_train_device,
            temp_device=resolved_temp_device,
            layer_offload_fraction=resolved_layer_offload_fraction,
            offload_activations=resolved_offload_activations,
            enable_async=resolved_enable_async,
        )
