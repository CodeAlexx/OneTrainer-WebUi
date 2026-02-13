"""Memory management utilities."""

from serenity.memory.allocators import (
    PinnedMemoryPool,
    StaticActivationAllocator,
    StaticLayerAllocator,
    StaticLayerTensorAllocator,
    SyncEvent,
    create_stream_context,
    pin_tensor_,
    unpin_tensor_,
)
from serenity.memory.checkpoint_layer import OffloadCheckpointLayer
from serenity.memory.conductor import LayerOffloadConductor
from serenity.memory.manager import MemoryManager
from serenity.memory.stagehand_strategy import StagehandStrategy, StagehandStrategyConfig
from serenity.memory.strategy import LayerOffloadStrategy, MemoryConfig, MemoryStrategy, NoOpMemoryStrategy
from serenity.memory.sync import torch_gc, device_equals, tensors_match_device

__all__ = [
    "LayerOffloadConductor",
    "OffloadCheckpointLayer",
    "MemoryManager",
    "MemoryConfig",
    "MemoryStrategy",
    "NoOpMemoryStrategy",
    "LayerOffloadStrategy",
    "StagehandStrategy",
    "StagehandStrategyConfig",
    "StaticLayerAllocator",
    "StaticLayerTensorAllocator",
    "StaticActivationAllocator",
    "PinnedMemoryPool",
    "SyncEvent",
    "create_stream_context",
    "pin_tensor_",
    "unpin_tensor_",
    "torch_gc",
    "device_equals",
    "tensors_match_device",
]
