"""Base quantized operations for the inference engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantizedLinear",
    "ManualCastLinear",
    "OperationContext",
    "get_weight_and_bias",
]

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """Base class for all quantized linear layers.

    Subclasses implement specific quantization formats (INT8, FP8, BnB, GGUF).
    The forward pass dequantizes the weight, computes the linear, and returns.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias: nn.Parameter | None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight to compute_dtype. Override in subclasses."""
        return self.weight.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def get_weight_and_bias(
    layer: nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Get weight and bias from a layer, applying online LoRA if present.

    Mirrors Forge's get_weight_and_bias (operations.py:46-65). When a layer
    has ``forge_online_loras`` attached, the stored patches are applied to the
    weight/bias before returning.
    """
    scale_weight: torch.Tensor | None = getattr(layer, "scale_weight", None)
    loras: dict[str, list[Any]] = getattr(layer, "forge_online_loras", {})

    weight_patches = loras.get("weight", None)
    bias_patches = loras.get("bias", None)

    weight: torch.Tensor | None = getattr(layer, "weight", None)
    if weight is not None:
        if scale_weight is not None:
            weight = weight * scale_weight.to(device=weight.device, dtype=weight.dtype)
        if weight_patches is not None:
            for patch in weight_patches:
                weight = weight + patch.to(device=weight.device, dtype=weight.dtype)

    bias: torch.Tensor | None = getattr(layer, "bias", None)
    if bias is not None:
        if bias_patches is not None:
            for patch in bias_patches:
                bias = bias + patch.to(device=bias.device, dtype=bias.dtype)

    return weight, bias


class ManualCastLinear(nn.Linear):
    """Linear layer that casts weight to input dtype on forward.

    Derived from Forge's ForgeOperations.Linear. When ``parameters_manual_cast``
    is enabled, weight/bias are cast to the input tensor's dtype before the
    matmul.  Online LoRA patches are applied via ``get_weight_and_bias``.
    """

    parameters_manual_cast: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        manual_cast: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.parameters_manual_cast = manual_cast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parameters_manual_cast:
            weight, bias = get_weight_and_bias(self)
            if weight is not None:
                weight = weight.to(dtype=x.dtype, device=x.device)
            if bias is not None:
                bias = bias.to(dtype=x.dtype, device=x.device)
            return F.linear(x, weight, bias)
        else:
            weight, bias = get_weight_and_bias(self)
            return F.linear(x, weight, bias)


@dataclass
class OperationContext:
    """Replaces Forge's monkey-patching with a clean context object.

    Holds the classes to use for model construction so that model code can
    call ``ctx.create_linear(...)`` instead of relying on global state.
    """

    linear_cls: type = field(default_factory=lambda: nn.Linear)
    conv2d_cls: type = field(default_factory=lambda: nn.Conv2d)
    dtype: torch.dtype = torch.float32
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def create_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> nn.Module:
        """Create a linear layer using the configured class."""
        if self.linear_cls is ManualCastLinear:
            return ManualCastLinear(
                in_features, out_features, bias=bias, manual_cast=True
            )
        return self.linear_cls(in_features, out_features, bias=bias)

    def create_conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        **kwargs: Any,
    ) -> nn.Module:
        """Create a Conv2d layer using the configured class."""
        return self.conv2d_cls(in_channels, out_channels, kernel_size, **kwargs)
