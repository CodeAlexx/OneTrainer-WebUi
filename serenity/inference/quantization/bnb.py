"""BitsAndBytes 4-bit quantized operations."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from serenity.inference.quantization.ops import QuantizedLinear

__all__ = ["BnbLinear4bit", "is_available"]

logger = logging.getLogger(__name__)

# Try to import bitsandbytes; gracefully handle absence.
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit

    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    bnb = None  # type: ignore[assignment]
    Linear4bit = None  # type: ignore[assignment,misc]


def is_available() -> bool:
    """Return True when bitsandbytes is installed and importable."""
    return _BNB_AVAILABLE


class BnbLinear4bit(QuantizedLinear):
    """4-bit quantized linear layer backed by ``bitsandbytes.nn.Linear4bit``.

    Supports NF4 and FP4 quantisation types.  When bitsandbytes is not
    installed, construction raises ``RuntimeError``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        if not _BNB_AVAILABLE:
            raise RuntimeError(
                "bitsandbytes is required for BnbLinear4bit but is not installed"
            )
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            compute_dtype=compute_dtype,
        )
        self.quant_type = quant_type
        self._inner: Linear4bit | None = None

    def init_inner(self, device: torch.device | None = None) -> None:
        """Lazily initialise the underlying ``Linear4bit`` layer."""
        if self._inner is not None:
            return
        self._inner = Linear4bit(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            compute_dtype=self.compute_dtype,
            quant_type=self.quant_type,
            device=device,
        )

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize the inner 4-bit weight to compute_dtype."""
        if self._inner is not None and hasattr(self._inner, "weight"):
            w = self._inner.weight
            if hasattr(w, "bnb_quantized") and w.bnb_quantized:
                return bnb.functional.dequantize_4bit(
                    w.data, w.quant_state
                ).to(self.compute_dtype)
        return self.weight.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inner is not None:
            return self._inner(x)
        # Fallback: dequantize and compute
        weight = self.dequantize_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any,
    ) -> None:
        """Detect and handle pre-quantized BnB weights."""
        quant_state_key = f"{prefix}weight.quant_state"
        absmax_key = f"{prefix}weight.absmax"

        # Check for pre-quantized BnB format
        if quant_state_key in state_dict or absmax_key in state_dict:
            # Pre-quantized: load directly into inner module
            if self._inner is not None:
                self._inner._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            return

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
