"""Pinned (page-locked) host memory management for fast CPU-to-GPU transfers."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.memory.vram import is_cuda_available, is_windows

__all__ = [
    "PinnedMemoryManager",
    "pin_model_weights",
    "unpin_model_weights",
]

logger = logging.getLogger(__name__)


def _system_ram_bytes() -> int:
    """Best-effort total system RAM in bytes."""
    try:
        import psutil  # noqa: F811
        return psutil.virtual_memory().total
    except ImportError:
        # psutil unavailable — use a conservative 8 GB fallback.
        return 8 * 1024 * 1024 * 1024


def _discard_cuda_async_error() -> None:
    """Drain a stale async CUDA error so subsequent ops don't fail.

    Pinning sometimes leaves a deferred error on the GPU stream.  Executing
    a trivial op + sync clears it.
    """
    if not is_cuda_available():
        return
    try:
        a = torch.tensor([1], dtype=torch.uint8, device="cuda")
        b = torch.tensor([1], dtype=torch.uint8, device="cuda")
        _ = a + b
        torch.cuda.synchronize()
    except RuntimeError:
        # CUDA async error or synchronization failed
        pass


class PinnedMemoryManager:
    """Track and limit pinned (page-locked) host memory.

    Pinned memory enables DMA transfers between CPU and GPU, roughly doubling
    transfer bandwidth.  However, the OS imposes hard limits on how much
    memory a process can pin — exceeding the limit on Windows can destabilise
    the system.

    Budget defaults:
    * **Linux**: 95 % of system RAM
    * **Windows**: 45 % of system RAM (the OS limit is ~50 %)
    * **No CUDA**: 0 (pinning is disabled)
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        self._pinned: dict[int, int] = {}  # data_ptr -> nbytes
        self._total_pinned: int = 0

        if max_bytes is not None:
            self._max_bytes = max_bytes
        elif not is_cuda_available():
            self._max_bytes = 0
        elif is_windows():
            self._max_bytes = int(_system_ram_bytes() * 0.45)
        else:
            # Linux / other POSIX
            self._max_bytes = int(_system_ram_bytes() * 0.95)

    # -- public API ---------------------------------------------------------

    @property
    def total_pinned(self) -> int:
        """Total bytes currently pinned."""
        return self._total_pinned

    @property
    def max_bytes(self) -> int:
        """Configured ceiling for pinned memory."""
        return self._max_bytes

    def can_pin(self, size: int) -> bool:
        """Return ``True`` if *size* bytes can be pinned within the budget."""
        if self._max_bytes <= 0:
            return False
        return (self._total_pinned + size) <= self._max_bytes

    def pin(self, tensor: torch.Tensor) -> bool:
        """Pin a CPU tensor using ``cudaHostRegister``.

        Returns ``True`` on success.  Silently returns ``False`` when:
        * CUDA is unavailable
        * The tensor is already pinned
        * The tensor is not contiguous
        * The budget would be exceeded
        """
        if self._max_bytes <= 0:
            return False

        if not is_cuda_available():
            return False

        if tensor.device.type != "cpu":
            return False

        if tensor.is_pinned():
            return False

        if not tensor.is_contiguous():
            return False

        size = tensor.nbytes
        if not self.can_pin(size):
            return False

        ptr = tensor.data_ptr()
        if ptr == 0:
            return False

        result = torch.cuda.cudart().cudaHostRegister(ptr, size, 1)
        if result == 0:
            self._pinned[ptr] = size
            self._total_pinned += size
            return True

        logger.warning("cudaHostRegister failed (error %s)", result)
        _discard_cuda_async_error()
        return False

    def unpin(self, tensor: torch.Tensor) -> bool:
        """Unpin a previously pinned tensor.

        Returns ``True`` on success, ``False`` if the tensor was not tracked.
        """
        if self._max_bytes <= 0:
            return False

        if not is_cuda_available():
            return False

        if tensor.device.type != "cpu":
            return False

        ptr = tensor.data_ptr()
        stored = self._pinned.get(ptr)
        if stored is None:
            return False

        if tensor.nbytes != stored:
            logger.warning("Pinned tensor size changed (%d -> %d)", stored, tensor.nbytes)
            return False

        result = torch.cuda.cudart().cudaHostUnregister(ptr)
        if result == 0:
            self._total_pinned -= self._pinned.pop(ptr)
            if not self._pinned:
                self._total_pinned = 0  # reset drift
            return True

        logger.warning("cudaHostUnregister failed (error %s)", result)
        _discard_cuda_async_error()
        return False


# ---------------------------------------------------------------------------
# Convenience helpers for entire models
# ---------------------------------------------------------------------------

def pin_model_weights(
    model: nn.Module,
    manager: PinnedMemoryManager | None = None,
) -> PinnedMemoryManager:
    """Pin all CPU parameters of *model*.

    If no *manager* is given, a new one is created with default budget.
    Returns the manager so the caller can later :func:`unpin_model_weights`.
    """
    if manager is None:
        manager = PinnedMemoryManager()
    for param in model.parameters():
        if param.device.type == "cpu":
            manager.pin(param.data)
    return manager


def unpin_model_weights(model: nn.Module, manager: PinnedMemoryManager) -> None:
    """Unpin all CPU parameters that *manager* pinned."""
    for param in model.parameters():
        if param.device.type == "cpu":
            manager.unpin(param.data)
