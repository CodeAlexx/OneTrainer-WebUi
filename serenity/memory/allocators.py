"""Static memory allocators, pinned memory pools, and sync event wrappers.

- ``StaticLayerAllocator`` / ``StaticLayerTensorAllocator``: pre-allocate
  GPU buffers to avoid fragmentation during layer offloading.
- ``StaticActivationAllocator``: pre-allocate activation cache buffers.
- ``PinnedMemoryPool``: pinned CPU memory for faster CPU<->GPU transfers.
- ``SyncEvent``: thin wrapper around ``torch.cuda.Event`` for overlapping
  transfers with compute.

These integrate with the existing ``LayerOffloadConductor`` via the
conductor's ``before_layer``/``after_layer`` hooks.
"""

from __future__ import annotations

import gc
import logging
import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "StaticLayerTensorAllocator",
    "StaticLayerAllocator",
    "StaticActivationAllocator",
    "PinnedMemoryPool",
    "SyncEvent",
    "create_stream_context",
    "pin_tensor_",
    "unpin_tensor_",
]


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #


def _ceil_16(n: int) -> int:
    """Round up to the next multiple of 16."""
    return n + (16 - (n % 16)) % 16


def _floor_16(n: int) -> int:
    """Round down to the nearest multiple of 16."""
    return n - (n % 16)


# --------------------------------------------------------------------------- #
# Pinning helpers (use CUDA host-register for true pinned memory)
# --------------------------------------------------------------------------- #


def pin_tensor_(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor's memory for faster transfers to GPU.

    Uses ``cudaHostRegister`` for zero-copy pinning of existing allocations.
    No-op if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        0,
    )
    if err.value != 0:
        raise RuntimeError(
            f"CUDA error pinning memory: err={err.value}, "
            f"ptr={tensor.data_ptr()}, "
            f"size={tensor.numel() * tensor.element_size()}"
        )


def unpin_tensor_(tensor: torch.Tensor) -> None:
    """Unpin a previously pinned CPU tensor.

    No-op if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostUnregister(tensor.data_ptr())
    if err.value != 0:
        raise RuntimeError(
            f"CUDA error unpinning memory: err={err.value}, ptr={tensor.data_ptr()}"
        )


# --------------------------------------------------------------------------- #
# Stream context helper
# --------------------------------------------------------------------------- #


def create_stream_context(
    stream: torch.cuda.Stream | None,
) -> torch.cuda.StreamContext | nullcontext:
    """Create a CUDA stream context, or a no-op context if stream is None."""
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.StreamContext(stream)
    return nullcontext()


# --------------------------------------------------------------------------- #
# SyncEvent
# --------------------------------------------------------------------------- #


class SyncEvent:
    """Wrapper around ``torch.cuda.Event`` for overlapping transfer and compute.

    Provides ``record()``, ``wait(stream)``, and ``synchronize()`` methods
    with optional debug logging.  A no-op event (created with no torch_event)
    is safe to call in all the same places.
    """

    def __init__(
        self,
        torch_event: torch.cuda.Event | None = None,
        log_msg: str | None = None,
    ) -> None:
        self.id = str(random.randint(0, 2 << 30)) if torch_event is not None else "-"
        self._torch_event = torch_event
        self._log_msg = log_msg

    def record(self) -> None:
        """Record the event on the current stream."""
        if self._torch_event is not None:
            self._torch_event.record()

    def wait(self, stream: torch.Stream | None, log_msg: str | None = None) -> None:
        """Make ``stream`` wait until this event has been recorded."""
        if self._torch_event is not None and stream is not None:
            stream.wait_event(self._torch_event)

    def synchronize(self, log_msg: str | None = None) -> None:
        """Block the CPU until this event has completed on the GPU."""
        if self._torch_event is not None:
            if not self._torch_event.query():
                self._torch_event.synchronize()

    @property
    def done(self) -> bool:
        """Return True if the event has already completed."""
        if self._torch_event is None:
            return True
        return self._torch_event.query()

    def __repr__(self) -> str:
        if self._torch_event is None:
            return "SyncEvent(None)"
        return f"SyncEvent({self._log_msg}, done={self.done})"


# --------------------------------------------------------------------------- #
# StaticLayerTensorAllocator
# --------------------------------------------------------------------------- #


class StaticLayerTensorAllocator:
    """Allocates tensors from a pre-allocated cache buffer for one layer.

    Supports both forward and reverse allocation (for backward-pass
    offloading). The allocator carves slices from the parent
    ``StaticLayerAllocator``'s cache tensors.
    """

    def __init__(
        self,
        layer_allocator: StaticLayerAllocator,
        allocate_forward: bool,
        layer_index: int,
    ) -> None:
        self._layer_allocator = layer_allocator
        self._allocate_forward = allocate_forward
        self._layer_index = layer_index

        if allocate_forward:
            self._allocation_start = layer_allocator.allocation_end
            self._allocation_end = layer_allocator.allocation_end
        else:
            self._allocation_start = layer_allocator.allocation_start
            self._allocation_end = layer_allocator.allocation_start

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        """Allocate a tensor with the same shape/dtype from the cache buffer."""
        num_bytes = source_tensor.numel() * source_tensor.element_size()
        cache_tensor_size = self._layer_allocator.cache_tensor_size
        total_cache_bytes = cache_tensor_size * len(self._layer_allocator.cache_tensors)

        if self._allocate_forward:
            ct_index = self._allocation_end // cache_tensor_size
            ct_offset = _ceil_16(self._allocation_end % cache_tensor_size)

            if ct_offset + num_bytes > cache_tensor_size:
                ct_index += 1
                ct_offset = 0
            if ct_index * cache_tensor_size + ct_offset + num_bytes > total_cache_bytes:
                ct_index = 0
                ct_offset = 0

            self._allocation_end = ct_index * cache_tensor_size + ct_offset
            self._layer_allocator.ensure_allocation(ct_index)
            cache_tensor = self._layer_allocator.cache_tensors[ct_index]
            allocated = cache_tensor[ct_offset:ct_offset + num_bytes]
            self._allocation_end += num_bytes
            self._layer_allocator.allocation_end = self._allocation_end
        else:
            ct_index = self._allocation_start // cache_tensor_size
            ct_offset = self._allocation_start % cache_tensor_size

            if ct_offset - num_bytes < 0:
                ct_index -= 1
                ct_offset = cache_tensor_size
            if ct_index < 0:
                ct_index = len(self._layer_allocator.cache_tensors) - 1
                ct_offset = cache_tensor_size

            new_start = _floor_16(ct_offset - num_bytes)
            self._layer_allocator.ensure_allocation(ct_index)
            cache_tensor = self._layer_allocator.cache_tensors[ct_index]
            allocated = cache_tensor[new_start:new_start + num_bytes]
            self._allocation_start = ct_index * cache_tensor_size + new_start
            self._layer_allocator.allocation_start = self._allocation_start

        return allocated.view(dtype=source_tensor.dtype).view(size=source_tensor.shape)

    def deallocate(self, deallocate_forward: bool) -> None:
        """Release this layer's allocation region."""
        if deallocate_forward:
            self._layer_allocator.allocation_start = self._allocation_end
        else:
            self._layer_allocator.allocation_end = self._allocation_start


# --------------------------------------------------------------------------- #
# StaticLayerAllocator
# --------------------------------------------------------------------------- #


class StaticLayerAllocator:
    """Pre-allocates GPU (or CPU) buffers to avoid memory fragmentation.

    Manages a set of large cache tensors that ``StaticLayerTensorAllocator``
    instances carve sub-allocations from.  CPU allocations are automatically
    pinned for faster GPU transfers.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._is_pinned = device.type == "cpu"

        self._layer_bytes: list[int] = []
        self._max_tensor_bytes: int = 0
        self.cache_tensors: list[torch.Tensor | None] = []
        self.cache_tensor_size: int = 0

        self.allocation_start: int = 0
        self.allocation_end: int = 0

        self._tensor_allocators: list[StaticLayerTensorAllocator | None] = []

    def allocate_cache(self, layers: list[nn.Module], target_bytes: int) -> None:
        """Pre-allocate the cache tensor pool for the given layers.

        ``target_bytes`` should be the maximum bytes that need to be
        resident on this device at once.
        """
        if any(x is not None for x in self.cache_tensors):
            return

        self._max_tensor_bytes = 0
        self._layer_bytes = []
        for layer in layers:
            all_bytes: list[int] = []
            for p in layer.parameters(recurse=True):
                all_bytes.append(p.numel() * p.element_size())
            for b in layer.buffers(recurse=True):
                all_bytes.append(b.numel() * b.element_size())
            self._max_tensor_bytes = max(self._max_tensor_bytes, *(all_bytes or [0]))
            self._layer_bytes.append(sum(all_bytes))

        if self._max_tensor_bytes == 0:
            return

        cache_bytes = target_bytes
        num_cache_tensors = min(
            math.ceil(int(cache_bytes * 0.10) / max(self._max_tensor_bytes, 1)),
            math.ceil(cache_bytes / max(self._max_tensor_bytes * 2, 1)),
            10,
        )
        num_cache_tensors = max(num_cache_tensors, 1)
        self.cache_tensor_size = (
            math.ceil(cache_bytes / num_cache_tensors)
            + self._max_tensor_bytes
            + 4096
        )

        self._tensor_allocators = [None] * len(layers)
        self.cache_tensors = [None] * num_cache_tensors
        self.allocation_start = 0
        self.allocation_end = 0

    def ensure_allocation(self, cache_tensor_index: int) -> None:
        """Lazily allocate a cache tensor slab, pinning CPU memory."""
        if self.cache_tensors[cache_tensor_index] is not None:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.cache_tensors[cache_tensor_index] = torch.zeros(
            (self.cache_tensor_size,), dtype=torch.int8, device=self.device,
        )
        if self._is_pinned:
            pin_tensor_(self.cache_tensors[cache_tensor_index])

    def deallocate_cache(self) -> None:
        """Free all cache tensors, unpinning CPU memory."""
        for ct in self.cache_tensors:
            if ct is not None and self._is_pinned:
                unpin_tensor_(ct)
        self.cache_tensors = [None] * len(self.cache_tensors)
        self._tensor_allocators = [None] * len(self._tensor_allocators)

    def get_allocator(
        self, layer_index: int, allocate_forward: bool,
    ) -> StaticLayerTensorAllocator:
        """Get a tensor allocator for the given layer."""
        allocator = StaticLayerTensorAllocator(self, allocate_forward, layer_index)
        self._tensor_allocators[layer_index] = allocator
        return allocator

    def deallocate_layer(self, layer_index: int, deallocate_forward: bool) -> None:
        """Release the allocation for a specific layer."""
        if self._tensor_allocators[layer_index] is not None:
            self._tensor_allocators[layer_index].deallocate(deallocate_forward)
            self._tensor_allocators[layer_index] = None


# --------------------------------------------------------------------------- #
# StaticActivationAllocator
# --------------------------------------------------------------------------- #


class StaticActivationAllocator:
    """Pre-allocates cache buffers for activation offloading.

    Accumulates activation tensors into a pool of cache slabs.
    CPU allocations are automatically pinned.  After each forward/backward
    pass, call ``deallocate()`` to reset the pointer without freeing memory
    (the cache is reused across steps).
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._is_pinned = device.type == "cpu"

        self._cache_tensors: list[torch.Tensor] = []
        self._current_ct: int = 0
        self._current_ct_offset: int = 0
        self._allocated_bytes: int = 0
        self._max_allocated_bytes: int = 0

    def reserve_cache(self, tensors: list[torch.Tensor]) -> None:
        """Ensure enough cache space for the given list of tensors."""
        num_bytes = sum(t.element_size() * t.numel() for t in tensors)
        num_bytes += len(tensors) * 4  # alignment padding

        if num_bytes == 0:
            return

        if len(self._cache_tensors) == 0:
            num_bytes = max(num_bytes, self._max_allocated_bytes)

        # Try to fit into an existing cache tensor
        found = False
        while self._current_ct < len(self._cache_tensors):
            remaining = self._cache_tensors[self._current_ct].shape[0] - self._current_ct_offset
            if remaining >= num_bytes:
                found = True
                break
            self._current_ct += 1
            self._current_ct_offset = 0

        if not found:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cache_tensor = torch.zeros(
                (num_bytes,), dtype=torch.int8, device=self._device,
            )
            if self._is_pinned:
                pin_tensor_(cache_tensor)

            self._cache_tensors.append(cache_tensor)
            self._allocated_bytes += num_bytes

        self._max_allocated_bytes = max(self._max_allocated_bytes, self._allocated_bytes)

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        """Carve out a buffer matching the source tensor's shape and dtype."""
        num_bytes = source_tensor.element_size() * source_tensor.numel()
        cache_tensor = self._cache_tensors[self._current_ct]
        allocated = cache_tensor[self._current_ct_offset:self._current_ct_offset + num_bytes]
        self._current_ct_offset += _ceil_16(num_bytes)
        return allocated.view(dtype=source_tensor.dtype).view(size=source_tensor.shape)

    def deallocate(self) -> None:
        """Reset allocation pointers for reuse. Condenses fragmented slabs."""
        if len(self._cache_tensors) > 1:
            if self._is_pinned:
                for ct in self._cache_tensors:
                    unpin_tensor_(ct)
            self._cache_tensors = []

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            num_bytes = self._allocated_bytes + 4096
            cache_tensor = torch.zeros(
                (num_bytes,), dtype=torch.int8, device=self._device,
            )
            if self._is_pinned:
                pin_tensor_(cache_tensor)
            self._cache_tensors = [cache_tensor]

        self._current_ct = 0
        self._current_ct_offset = 0
        self._allocated_bytes = sum(ct.shape[0] for ct in self._cache_tensors)

    def deallocate_cache(self) -> None:
        """Free all cache tensors entirely."""
        if self._is_pinned:
            for ct in self._cache_tensors:
                unpin_tensor_(ct)
        self._cache_tensors = []


# --------------------------------------------------------------------------- #
# PinnedMemoryPool
# --------------------------------------------------------------------------- #


class PinnedMemoryPool:
    """Pool of pinned CPU tensors for fast GPU<->CPU transfers.

    Pre-allocates a set of pinned CPU buffers that can be borrowed and
    returned.  Avoids the overhead of pinning/unpinning on every transfer.

    Usage::

        pool = PinnedMemoryPool(num_buffers=4, buffer_size=64 * 1024 * 1024)
        buf = pool.acquire(needed_bytes=1024 * 1024)
        # ... use buf for CPU<->GPU copy ...
        pool.release(buf)
    """

    def __init__(
        self,
        num_buffers: int = 4,
        buffer_size: int = 64 * 1024 * 1024,
    ) -> None:
        self._buffer_size = buffer_size
        self._all_buffers: list[torch.Tensor] = []
        self._available: list[torch.Tensor] = []

        for _ in range(num_buffers):
            buf = torch.zeros((buffer_size,), dtype=torch.uint8, device="cpu")
            if torch.cuda.is_available():
                pin_tensor_(buf)
            self._all_buffers.append(buf)
            self._available.append(buf)

    @property
    def buffer_size(self) -> int:
        """Size of each buffer in bytes."""
        return self._buffer_size

    @property
    def num_buffers(self) -> int:
        """Total number of buffers in the pool."""
        return len(self._all_buffers)

    @property
    def num_available(self) -> int:
        """Number of buffers currently available."""
        return len(self._available)

    def acquire(self, needed_bytes: int | None = None) -> torch.Tensor | None:
        """Borrow a pinned buffer from the pool.

        Returns None if no buffers are available or ``needed_bytes`` exceeds
        buffer size.
        """
        if needed_bytes is not None and needed_bytes > self._buffer_size:
            return None
        if not self._available:
            return None
        return self._available.pop()

    def release(self, buffer: torch.Tensor) -> None:
        """Return a borrowed buffer to the pool."""
        if buffer in self._all_buffers and buffer not in self._available:
            self._available.append(buffer)

    def cleanup(self) -> None:
        """Unpin and release all buffers."""
        for buf in self._all_buffers:
            if torch.cuda.is_available():
                try:
                    unpin_tensor_(buf)
                except RuntimeError:
                    pass  # already unpinned
        self._all_buffers.clear()
        self._available.clear()

    def __del__(self) -> None:
        self.cleanup()
