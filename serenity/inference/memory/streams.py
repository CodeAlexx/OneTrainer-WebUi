"""CUDA/XPU stream pool and gathered weight transfer utilities."""

from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

import torch

from serenity.inference.memory.vram import is_cuda_available, is_xpu_available

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
# Device-agnostic stream helpers (CUDA + XPU)
# ---------------------------------------------------------------------------

def _create_stream(device: torch.device) -> Any:
    """Create a device stream for CUDA or XPU."""
    if device.type == "cuda":
        return torch.cuda.Stream(device=device, priority=0)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.Stream(device)
    raise ValueError(f"Unsupported device type for streams: {device.type}")


def _current_stream(device: torch.device) -> Any:
    """Get current stream for device."""
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.current_stream(device)
    raise ValueError(f"Unsupported device type: {device.type}")


def _synchronize(device: torch.device) -> None:
    """Synchronize device."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


# ---------------------------------------------------------------------------
# StreamPool
# ---------------------------------------------------------------------------

_STREAM_DEVICE_TYPES = {"cuda", "xpu"}


class StreamPool:
    """Round-robin pool of device streams for async weight transfers.

    Supports both CUDA and Intel XPU devices.  On CPU-only machines the pool
    is empty and :meth:`get_stream` returns ``None``, so callers can always
    use::

        stream = pool.get_stream()
        if stream is not None:
            ...
    """

    def __init__(self, num_streams: int = 2, device: torch.device | None = None) -> None:
        self._streams: list[Any] = []
        self._index: int = 0
        self._device = device

        # Determine whether we can create streams for this device.
        if device is not None and device.type not in _STREAM_DEVICE_TYPES:
            return

        if device is None:
            if is_cuda_available():
                device = torch.device("cuda")
            elif is_xpu_available():
                device = torch.device("xpu")
            else:
                return
        self._device = device

        # Verify the backend is actually available.
        if device.type == "cuda" and not is_cuda_available():
            return
        if device.type == "xpu" and not is_xpu_available():
            return

        for _ in range(num_streams):
            self._streams.append(_create_stream(device))

    # -- public API ---------------------------------------------------------

    @property
    def num_streams(self) -> int:
        return len(self._streams)

    def get_stream(self) -> Any | None:
        """Return the next stream in the round-robin.

        Before returning a stream, it synchronises the *oldest* stream with
        the current default stream so that any prior transfer on that stream
        has completed before the default stream tries to use the result. This
        gives pipeline overlap: layer *N* computes on the default stream while
        layer *N+1* transfers on a different stream.
        """
        if not self._streams:
            return None

        # Ensure the default stream waits for the oldest transfer to complete.
        oldest = self._streams[self._index]
        _current_stream(self._device).wait_stream(oldest)

        # Advance round-robin.
        self._index = (self._index + 1) % len(self._streams)
        return self._streams[self._index]

    def sync_all(self) -> None:
        """Wait for every stream in the pool to finish."""
        if not self._streams:
            return
        current = _current_stream(self._device)
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

    QuantizedTensor support: when a weight implements the
    ``__tensor_flatten__`` / ``__tensor_unflatten__`` protocol (duck-typed),
    the weight is decomposed into its inner tensors for transfer.  The main
    (first) inner tensor is packed into the contiguous buffer; metadata and
    any secondary inner tensors are stored so that :meth:`unpack_weight_bias`
    can reconstruct the original QuantizedTensor.
    """

    @staticmethod
    def packed_size(weight: torch.Tensor, bias: torch.Tensor | None = None) -> int:
        """Return the total byte size needed for the packed buffer."""
        w = weight
        # For QuantizedTensors, measure the first inner tensor.
        if hasattr(w, "__tensor_flatten__"):
            try:
                inner_names, _meta = w.__tensor_flatten__()
                w = getattr(w, inner_names[0])
            except (RuntimeError, AttributeError):
                # __tensor_flatten__ protocol failed or tensor structure unexpected
                pass  # Fall through to normal measurement.
        total = _tensor_aligned_size(w)
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

        When *weight* is a QuantizedTensor (detected via ``__tensor_flatten__``),
        the decomposed inner tensors and reconstruction metadata are stored on
        the returned buffer as ``_quant_state`` so that :meth:`unpack_weight_bias`
        can reconstruct the original type transparently.
        """
        quant_state: dict[str, object] = {}

        # -- QuantizedTensor detection (duck-typed) --------------------------
        if hasattr(weight, "__tensor_flatten__"):
            try:
                inner_names, metadata = weight.__tensor_flatten__()
                inner_tensors = {n: getattr(weight, n) for n in inner_names}
                quant_state = {
                    "quantized_cls": type(weight),
                    "inner_names": inner_names,
                    "metadata": metadata,
                    "extra_tensors": {
                        n: inner_tensors[n] for n in inner_names[1:]
                    },
                }
                # Use the primary inner tensor for the packed buffer.
                weight = inner_tensors[inner_names[0]]
            except (RuntimeError, AttributeError):
                logger.debug("QuantizedTensor flatten failed, falling back to plain pack")
                quant_state = {}

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

        # Attach quant metadata to the buffer for unpack_weight_bias.
        if quant_state:
            buf._quant_state = quant_state  # type: ignore[attr-defined]

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

        When *buffer* carries ``_quant_state`` (set by :meth:`pack_weight_bias`
        for QuantizedTensors), the primary inner tensor is extracted from the
        buffer and the original QuantizedTensor is reconstructed via
        ``__tensor_unflatten__``.

        The returned tensors are **views** into *buffer* — no copy is made
        (except for the QuantizedTensor wrapper reconstruction).
        """
        offset = 0
        w_numel = math.prod(weight_shape)
        w_bytes = w_numel * TensorGeometry(shape=(), dtype=weight_dtype).element_size()
        weight = buffer[offset:offset + w_bytes].view(dtype=weight_dtype).view(weight_shape)
        offset += _aligned_size(w_bytes)

        bias: torch.Tensor | None = None
        if bias_shape is not None and bias_dtype is not None:
            b_numel = math.prod(bias_shape)
            b_bytes = b_numel * TensorGeometry(shape=(), dtype=bias_dtype).element_size()
            bias = buffer[offset:offset + b_bytes].view(dtype=bias_dtype).view(bias_shape)

        # -- QuantizedTensor reconstruction ----------------------------------
        quant_state: dict[str, object] | None = getattr(buffer, "_quant_state", None)
        if quant_state:
            try:
                cls = quant_state["quantized_cls"]
                inner_names: list[str] = quant_state["inner_names"]  # type: ignore[assignment]
                metadata = quant_state["metadata"]
                extra: dict[str, torch.Tensor] = quant_state.get("extra_tensors", {})  # type: ignore[assignment]

                inner_tensors = {inner_names[0]: weight}
                inner_tensors.update(extra)
                weight = cls.__tensor_unflatten__(  # type: ignore[union-attr]
                    inner_tensors, metadata, weight.size(), weight.stride(),
                )
            except (RuntimeError, AttributeError):
                logger.debug("QuantizedTensor unflatten failed, returning plain tensor")

        return weight, bias
