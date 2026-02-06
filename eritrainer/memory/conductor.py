"""Layer and activation offloading conductor for native EriTrainer."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from eritrainer.memory.sync import torch_gc


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)


def _move_structure(value: Any, device: torch.device, non_blocking: bool = False) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(_move_structure(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [_move_structure(item, device, non_blocking) for item in value]
    if isinstance(value, dict):
        return {k: _move_structure(v, device, non_blocking) for (k, v) in value.items()}
    return value


@dataclass
class _LayerState:
    layer: nn.Module
    offload_param_indices: list[int] | None
    on_train_device: bool = True


class ForwardContext(AbstractContextManager):
    """Forward/backward context for a single training step."""

    def __init__(self, conductor: "LayerOffloadConductor"):
        self.conductor = conductor

    def __enter__(self) -> "ForwardContext":
        self.conductor.start_forward(keep_graph=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.conductor.end_step()
        return None


class LayerOffloadConductor:
    """Coordinates CPU/GPU movement of layers and activations."""

    def __init__(
        self,
        module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        layer_offload_fraction: float = 0.0,
        offload_activations: bool = True,
        enable_async: bool = True,
    ) -> None:
        self.module = module
        self.train_device = torch.device(train_device)
        self.temp_device = torch.device(temp_device)
        self.layer_offload_fraction = max(0.0, min(float(layer_offload_fraction), 1.0))
        self.offload_activations = bool(offload_activations)
        self.enable_async = bool(enable_async)

        self._layers: list[_LayerState] = []
        self._call_index = 0
        self._active = False

    def __repr__(self) -> str:
        return (
            "LayerOffloadConductor("
            f"layers={len(self._layers)}, "
            f"layer_offload_fraction={self.layer_offload_fraction:.2f}, "
            f"offload_activations={self.offload_activations}, "
            f"train_device={self.train_device}, "
            f"temp_device={self.temp_device})"
        )

    def offload_activated(self) -> bool:
        return self.layer_offload_activated() or self.activation_offload_activated()

    def layer_offload_activated(self) -> bool:
        return (
            self.layer_offload_fraction > 0.0
            and self.train_device.type == "cuda"
            and self.temp_device != self.train_device
        )

    def activation_offload_activated(self) -> bool:
        return (
            self.offload_activations
            and self.train_device.type == "cuda"
            and self.temp_device != self.train_device
        )

    def add_layer(self, layer: nn.Module, included_offload_param_indices: list[int] | None = None) -> None:
        self._layers.append(
            _LayerState(
                layer=layer,
                offload_param_indices=included_offload_param_indices,
                on_train_device=True,
            )
        )

    def to(self, device: torch.device) -> None:
        self.module.to(device=device)
        target = torch.device(device)
        for state in self._layers:
            state.on_train_device = target == self.train_device

    def start_forward(self, keep_graph: bool = False) -> None:
        del keep_graph
        self._active = True
        self._call_index += 1
        if self.layer_offload_activated():
            self._load_layers_for_forward()

    def before_layer(self, layer_index: int, call_index: int, activations: Any = None) -> Any:
        del call_index
        if not self._active:
            return activations
        if not (0 <= layer_index < len(self._layers)):
            return activations

        state = self._layers[layer_index]
        if self.layer_offload_activated() and not state.on_train_device:
            state.layer.to(device=self.train_device, non_blocking=self.enable_async)
            state.on_train_device = True

        if self.activation_offload_activated() and activations is not None:
            return _move_structure(activations, self.train_device, non_blocking=self.enable_async)

        return activations

    def after_layer(self, layer_index: int, call_index: int, activations: Any = None) -> Any:
        del call_index
        if not self._active:
            return activations
        if not (0 <= layer_index < len(self._layers)):
            return activations

        if self.activation_offload_activated() and activations is not None:
            return _move_structure(activations, self.temp_device, non_blocking=self.enable_async)

        return activations

    def forward_context(self) -> AbstractContextManager:
        return ForwardContext(self)

    def end_step(self) -> None:
        self._active = False
        self._offload_deferred_layers()
        if self.activation_offload_activated():
            torch_gc()

    def cleanup(self) -> None:
        if not self._layers:
            return
        self._active = False
        for state in self._layers:
            state.layer.to(device=self.train_device)
            state.on_train_device = True

    def get_loaded_layers(self) -> list[int]:
        return [idx for (idx, state) in enumerate(self._layers) if state.on_train_device]

    def get_offloaded_layers(self) -> list[int]:
        return [idx for (idx, state) in enumerate(self._layers) if not state.on_train_device]

    def get_memory_stats(self) -> dict[str, float | int]:
        loaded_bytes = 0
        offloaded_bytes = 0
        loaded_layers = 0
        offloaded_layers = 0

        for state in self._layers:
            layer_bytes = 0
            for p in state.layer.parameters(recurse=True):
                layer_bytes += p.numel() * p.element_size()
            for b in state.layer.buffers(recurse=True):
                layer_bytes += b.numel() * b.element_size()

            if state.on_train_device:
                loaded_layers += 1
                loaded_bytes += layer_bytes
            else:
                offloaded_layers += 1
                offloaded_bytes += layer_bytes

        return {
            "total_layers": len(self._layers),
            "loaded_layers": loaded_layers,
            "offloaded_layers": offloaded_layers,
            "loaded_bytes": loaded_bytes,
            "offloaded_bytes": offloaded_bytes,
        }

    def _load_layers_for_forward(self) -> None:
        keep_on_train = max(int(round((1.0 - self.layer_offload_fraction) * len(self._layers))), 1)
        for idx, state in enumerate(self._layers):
            should_be_loaded = idx < keep_on_train
            if should_be_loaded and not state.on_train_device:
                state.layer.to(device=self.train_device, non_blocking=self.enable_async)
                state.on_train_device = True
            elif not should_be_loaded and state.on_train_device:
                state.layer.to(device=self.temp_device, non_blocking=self.enable_async)
                state.on_train_device = False

    def _offload_deferred_layers(self) -> None:
        if not self.layer_offload_activated():
            return
        keep_on_train = max(int(round((1.0 - self.layer_offload_fraction) * len(self._layers))), 1)
        for idx, state in enumerate(self._layers):
            if idx >= keep_on_train and state.on_train_device:
                state.layer.to(device=self.temp_device, non_blocking=self.enable_async)
                state.on_train_device = False
