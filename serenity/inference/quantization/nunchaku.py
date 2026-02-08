"""Nunchaku/SVDQ passthrough quantization for pre-quantized models."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

__all__ = ["NunchakuLinear", "is_available", "detect_nunchaku_weights"]

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if nunchaku runtime is available."""
    try:
        import nunchaku  # noqa: F401

        return True
    except ImportError:
        return False


def detect_nunchaku_weights(state_dict: dict[str, Any]) -> bool:
    """Detect if a state dict contains Nunchaku-quantized weights.

    Nunchaku models typically have keys with '.qweight', '.qzeros', '.scales'
    suffixes indicating pre-quantized GPTQ/AWQ-style weights.
    """
    nunchaku_keys = {".qweight", ".qzeros", ".scales", ".g_idx"}
    for key in state_dict:
        if any(key.endswith(suffix) for suffix in nunchaku_keys):
            return True
    return False


class NunchakuLinear(nn.Module):
    """Passthrough linear layer for pre-quantized Nunchaku weights.

    Stores weights in their original quantized format without re-quantization.
    Forward pass dequantizes on-the-fly for computation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._qweight: torch.Tensor | None = None
        self._scales: torch.Tensor | None = None
        self._qzeros: torch.Tensor | None = None
        self._g_idx: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        self._inner: nn.Module | None = None  # Optional nunchaku runtime module

    def load_quantized(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None = None,
        g_idx: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Load pre-quantized weight components."""
        self._qweight = qweight
        self._scales = scales
        self._qzeros = qzeros
        self._g_idx = g_idx
        self._bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        if self._inner is not None:
            return self._inner(x)

        # Fallback: dequantize and compute
        weight = self._dequantize()
        out = torch.nn.functional.linear(x, weight, self._bias)
        return out

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weight for computation."""
        if self._qweight is None:
            raise RuntimeError("No quantized weights loaded")
        # Simple dequantization: qweight * scales
        # Full implementation depends on the specific quantization format
        weight = self._qweight.float() * self._scales.float()
        return weight.to(self._qweight.device)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Intercept state dict loading to handle pre-quantized weights."""
        qweight_key = f"{prefix}qweight"
        scales_key = f"{prefix}scales"
        qzeros_key = f"{prefix}qzeros"
        g_idx_key = f"{prefix}g_idx"
        bias_key = f"{prefix}bias"

        if qweight_key in state_dict:
            self._qweight = state_dict.pop(qweight_key)
            self._scales = state_dict.pop(scales_key, None)
            self._qzeros = state_dict.pop(qzeros_key, None)
            self._g_idx = state_dict.pop(g_idx_key, None)
            self._bias = state_dict.pop(bias_key, None)
        else:
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
