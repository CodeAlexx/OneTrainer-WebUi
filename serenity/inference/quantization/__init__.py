"""Quantization backends for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.quantization.nunchaku import (
    NunchakuLinear,
    detect_nunchaku_weights,
)
from serenity.inference.quantization.nunchaku import is_available as nunchaku_available
from serenity.inference.quantization.ops import (
    ManualCastLinear,
    OperationContext,
    QuantizedLinear,
    disable_weight_init,
    get_weight_and_bias,
)

__all__ = [
    "ManualCastLinear",
    "NunchakuLinear",
    "OperationContext",
    "QuantizedLinear",
    "detect_nunchaku_weights",
    "disable_weight_init",
    "get_weight_and_bias",
    "nunchaku_available",
]
