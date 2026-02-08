"""GGUF quantized operations."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from serenity.inference.quantization.ops import QuantizedLinear

__all__ = [
    "GGUFLinear",
    "dequantize_tensor",
    "bake_gguf_model",
    "is_available",
]

logger = logging.getLogger(__name__)

# Try to import the gguf library; gracefully handle absence.
try:
    import gguf as _gguf_lib
    import gguf.quants as _gguf_quants  # noqa: F401

    _GGUF_AVAILABLE = True
except ImportError:
    _gguf_lib = None  # type: ignore[assignment]
    _gguf_quants = None  # type: ignore[assignment]
    _GGUF_AVAILABLE = False


def is_available() -> bool:
    """Return True when the gguf library is installed."""
    return _GGUF_AVAILABLE


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------


def dequantize_tensor(
    tensor: torch.Tensor | Any, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a GGUF-quantized tensor to *dtype*.

    If the tensor does not carry ``gguf_cls`` metadata it is returned as-is
    (optionally cast to *dtype*).
    """
    if tensor is None:
        raise ValueError("Cannot dequantize None")

    gguf_cls = getattr(tensor, "gguf_cls", None)
    if gguf_cls is not None:
        # Use the quant class's own dequantize method
        result = gguf_cls.dequantize_pytorch(tensor)
        if dtype is not None:
            result = result.to(dtype)
        return result

    # Not a GGUF tensor -- just return (optionally cast)
    if dtype is not None:
        return tensor.to(dtype)
    return tensor


# ---------------------------------------------------------------------------
# GGUF Linear layer
# ---------------------------------------------------------------------------


class GGUFLinear(QuantizedLinear):
    """Linear layer that stores weights in GGUF quantized format.

    On each forward call the weight is dequantized to the input dtype,
    the linear is computed, and the result is returned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            compute_dtype=compute_dtype,
        )

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize GGUF weight to compute_dtype."""
        return dequantize_tensor(self.weight, self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight().to(x.dtype)
        bias = self.bias
        if bias is not None:
            # Bias may also be GGUF-quantized
            if hasattr(bias, "gguf_cls") and bias.gguf_cls is not None:
                bias = dequantize_tensor(bias, x.dtype)
            else:
                bias = bias.to(x.dtype)
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Model-level GGUF baking
# ---------------------------------------------------------------------------


def bake_gguf_model(model: nn.Module) -> None:
    """One-time preprocessing pass for GGUF models.

    Walks every parameter and, if it carries a ``gguf_cls`` with a ``bake``
    method, calls that method to prepare the tensor for fast dequantization.
    Derived from Forge's ``memory_management.py:397-410``.
    """
    for name, param in model.named_parameters():
        gguf_cls = getattr(param, "gguf_cls", None)
        if gguf_cls is not None and hasattr(gguf_cls, "bake"):
            if not getattr(param, "baked", False):
                gguf_cls.bake(param)
                param.baked = True  # type: ignore[attr-defined]
                logger.debug("Baked GGUF parameter: %s", name)
