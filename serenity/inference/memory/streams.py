"""CUDA stream pool and gathered weight transfer utilities."""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch

from serenity.inference.memory.vram import is_cuda_available

__all__ = [
    "StreamPool",
    "CastBuffer",
    "TensorGeometry",
    "GatheredTransfer",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alignment constant — 1 KiB aligned, matching ComfyUI convention.
# ---------------------------------------------------------------------------
_ALIGNMENT = 1024


# ---------------------------------------------------------------------------
# StreamPool
# ---------------------------------------------------------------------------

class StreamPool:
    """Round-robin pool of CUDA streams for async weight transfers.

    On CPU-only machines the pool is empty and :meth:`get_stream` returns
    ``None``, so callers can always use::

        stream = pool.get_stream()
        if stream is not None:
            ...
    """

    def __init__(self, num_streams: int = 2, device: torch.device | None = None) -> None:
        self._streams: list[torch.cuda.Stream] = []
        self._index: int = 0
        self._device = device

        if not is_cuda_available():
            return

        if device is not None and device.type != "cuda":
            return

        if device is None:
            device = torch.device("cuda")
        self._device = device

        for _ in range(num_streams):
            self._streams.append(torch.cuda.Stream(device=device, priority=0))

    # -- public API ---------------------------------------------------------

    @property
    def num_streams(self) -> int:
        return len(self._streams)

    def get_stream(self) -> torch.cuda.Stream | None:
        """Return the next stream in the round-robin.

        Before returning a stream, it synchronises the *oldest* stream with
        the current default stream so that any prior transfer on that stream
        has completed before the default stream tries to use the result. This
        gives pipeline overlap: layer *N* computes on the default stream while
        layer *N+1* transfers on a different stream.
        """
        if not self._streams:
            return None

        # Sync the oldest stream with the current default stream.
        oldest = self._streams[self._index]
        oldest.wait_stream(torch.cuda.current_stream(self._device))

        # Advance round-robin.
        self._index = (self._index + 1) % len(self._streams)
        return self._streams[self._index]

    def sync_all(self) -> None:
        """Wait for every stream in the pool to finish."""
        if not self._streams:
            return
        current = torch.cuda.current_stream(self._device)
        for s in self._streams:
            current.wait_stream(s)


# ---------------------------------------------------------------------------
# CastBuffer — re-usable GPU scratch buffer for weight transfers.
# ---------------------------------------------------------------------------

class CastBuffer:
    """Reusable GPU buffer for transferring weights.

    Grows lazily to accommodate the largest weight seen.  If a *single*
    oversized weight is encountered (identified by *ref* identity), the buffer
    is **not** duplicated — callers should handle that edge-case by falling
    back to a synchronous copy.
    """

    def __init__(self) -> None:
        self._buffer: torch.Tensor | None = None
        self._largest_ref: object | None = None
        self._largest_size: int = 0

    def get_buffer(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        ref: object | None = None,
    ) -> torch.Tensor | None:
        """Return a buffer with at least *size* elements of *dtype* on *device*.

        Returns ``None`` if the request matches the single-oversized-weight
        edge-case so the caller can fall back.
        """
        byte_size = size  # we allocate raw int8 bytes
        if self._buffer is not None and self._buffer.numel() >= byte_size and self._buffer.device == device:
            return self._buffer[:byte_size]

        # Guard against two streams both allocating for the same huge weight.
        if ref is not None and ref is self._largest_ref:
            return None

        self._buffer = torch.empty(byte_size, dtype=torch.int8, device=device)
        if byte_size > self._largest_size:
            self._largest_ref = ref
            self._largest_size = byte_size

        return self._buffer[:byte_size]

    def reset(self) -> None:
        """Release the buffer."""
        self._buffer = None
        self._largest_ref = None
        self._largest_size = 0


# ---------------------------------------------------------------------------
# TensorGeometry — lightweight shape+dtype descriptor.
# ---------------------------------------------------------------------------

class TensorGeometry(NamedTuple):
    """Describes a tensor's shape and dtype without holding data."""

    shape: tuple[int, ...]
    dtype: torch.dtype

    def element_size(self) -> int:
        if self.dtype.is_floating_point:
            return torch.finfo(self.dtype).bits // 8
        return torch.iinfo(self.dtype).bits // 8

    def numel(self) -> int:
        return math.prod(self.shape)

    def nbytes(self) -> int:
        return self.numel() * self.element_size()


# ---------------------------------------------------------------------------
# GatheredTransfer — pack weight+bias into a single contiguous buffer.
# ---------------------------------------------------------------------------

def _aligned_size(nbytes: int) -> int:
    """Round *nbytes* up to the next multiple of ``_ALIGNMENT``."""
    return (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _tensor_aligned_size(t: torch.Tensor | None) -> int:
    """Aligned byte size of a single tensor (or 0 for None)."""
    if t is None:
        return 0
    return _aligned_size(t.numel() * t.element_size())


class GatheredTransfer:
    """Pack weight + bias into one contiguous byte buffer for a single DMA.

    This mirrors ComfyUI's ``interpret_gathered_like`` pattern: we allocate a
    flat ``uint8`` buffer, then use ``torch.Tensor.view`` to hand out
    correctly-typed, correctly-shaped views into it.
    """

    @staticmethod
    def packed_size(weight: torch.Tensor, bias: torch.Tensor | None = None) -> int:
        """Return the total byte size needed for the packed buffer."""
        total = _tensor_aligned_size(weight)
        if bias is not None:
            total += _tensor_aligned_size(bias)
        return total

    @staticmethod
    def pack_weight_bias(
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Pack *weight* (and optionally *bias*) into one contiguous buffer.

        The returned tensor is a flat ``uint8`` buffer on *device* (defaults to
        weight's device).  Use :meth:`unpack_weight_bias` to recover views.
        """
        if device is None:
            device = weight.device

        total = GatheredTransfer.packed_size(weight, bias)
        buf = torch.empty(total, dtype=torch.uint8, device=device)

        offset = 0
        # -- weight --
        w_bytes = weight.numel() * weight.element_size()
        buf[offset:offset + w_bytes].view(dtype=weight.dtype).view(weight.shape).copy_(
            weight, non_blocking=non_blocking,
        )
        offset += _aligned_size(w_bytes)

        # -- bias --
        if bias is not None:
            b_bytes = bias.numel() * bias.element_size()
            buf[offset:offset + b_bytes].view(dtype=bias.dtype).view(bias.shape).copy_(
                bias, non_blocking=non_blocking,
            )

        return buf

    @staticmethod
    def unpack_weight_bias(
        buffer: torch.Tensor,
        weight_shape: tuple[int, ...],
        weight_dtype: torch.dtype,
        bias_shape: tuple[int, ...] | None = None,
        bias_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Recover weight (and optional bias) views from a packed buffer.

        The returned tensors are **views** into *buffer* — no copy is made.
        """
        offset = 0
        w_numel = math.prod(weight_shape)
        w_bytes = w_numel * torch.tensor([], dtype=weight_dtype).element_size()
        weight = buffer[offset:offset + w_bytes].view(dtype=weight_dtype).view(weight_shape)
        offset += _aligned_size(w_bytes)

        bias: torch.Tensor | None = None
        if bias_shape is not None and bias_dtype is not None:
            b_numel = math.prod(bias_shape)
            b_bytes = b_numel * torch.tensor([], dtype=bias_dtype).element_size()
            bias = buffer[offset:offset + b_bytes].view(dtype=bias_dtype).view(bias_shape)

        return weight, bias
