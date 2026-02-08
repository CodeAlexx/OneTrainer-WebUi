"""Memory management subsystem for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.memory.vram import (
    VRAMBudget,
    VRAMState,
    calculate_budget,
    detect_vram_state,
    get_free_memory,
    get_total_memory,
    is_cuda_available,
    is_linux,
    is_nvidia,
    is_windows,
    minimum_inference_memory,
)
from serenity.inference.memory.streams import (
    CastBuffer,
    GatheredTransfer,
    StreamPool,
    TensorGeometry,
)
from serenity.inference.memory.pinned import (
    PinnedMemoryManager,
    pin_model_weights,
    unpin_model_weights,
)
from serenity.inference.memory.offload import (
    OffloadConv2d,
    OffloadLinear,
    OffloadMixin,
)
from serenity.inference.memory.manager import (
    LoadedModel,
    ModelManager,
)

__all__ = [
    # vram
    "VRAMBudget",
    "VRAMState",
    "calculate_budget",
    "detect_vram_state",
    "get_free_memory",
    "get_total_memory",
    "is_cuda_available",
    "is_linux",
    "is_nvidia",
    "is_windows",
    "minimum_inference_memory",
    # streams
    "CastBuffer",
    "GatheredTransfer",
    "StreamPool",
    "TensorGeometry",
    # pinned
    "PinnedMemoryManager",
    "pin_model_weights",
    "unpin_model_weights",
    # offload
    "OffloadConv2d",
    "OffloadLinear",
    "OffloadMixin",
    # manager
    "LoadedModel",
    "ModelManager",
]
