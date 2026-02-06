"""Standalone layer-offload manager used by local training scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from eritrainer.memory.conductor import LayerOffloadConductor


@dataclass
class LayerOffloadManager:
    """High-level helper around LayerOffloadConductor for script usage."""

    layers: list[nn.Module]
    layer_offload_fraction: float = 0.0
    train_device: str = "cuda:0"
    temp_device: str = "cpu"
    use_checkpoint: bool = True
    offload_activations: bool = False
    enable_async: bool = True

    def __post_init__(self) -> None:
        wrapper = nn.Module()
        wrapper.layers = nn.ModuleList(self.layers)
        self._wrapper = wrapper
        self._conductor = LayerOffloadConductor(
            module=wrapper,
            train_device=torch.device(self.train_device),
            temp_device=torch.device(self.temp_device),
            layer_offload_fraction=self.layer_offload_fraction,
            offload_activations=self.offload_activations and self.use_checkpoint,
            enable_async=self.enable_async,
        )
        for layer in self.layers:
            self._conductor.add_layer(layer)

    def activate(self) -> None:
        self._conductor.start_forward(keep_graph=False)
        self._conductor.end_step()

    def start_forward(self, keep_graph: bool = False) -> None:
        self._conductor.start_forward(keep_graph=keep_graph)
        # In standalone mode (without wrapped layers), keep all layers loaded.
        for layer in self.layers:
            layer.to(device=torch.device(self.train_device))

    def end_step(self) -> None:
        self._conductor.end_step()

    def get_memory_stats(self) -> dict[str, float | int]:
        return self._conductor.get_memory_stats()

    def cleanup(self) -> None:
        self._conductor.cleanup()
