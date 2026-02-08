"""Per-layer weight offloading — transfer weights to GPU on demand during forward."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "OffloadMixin",
    "OffloadLinear",
    "OffloadConv2d",
]

logger = logging.getLogger(__name__)


class OffloadMixin:
    """Mixin that adds on-demand weight casting to any ``nn.Module`` layer.

    When :attr:`offload_enabled` is ``True`` the layer's ``forward()``
    will:

    1. Transfer ``weight`` (and ``bias``) to the input tensor's device/dtype
       via :meth:`cast_weight`.
    2. Run the actual compute.
    3. Release the GPU copy via :meth:`release_weight` so VRAM can be reused
       by the next layer.

    The optional ``weight_function`` / ``bias_function`` lists hold callables
    that are applied to the weight/bias *after* the device transfer — this is
    the hook-point for model patching (LoRA, etc.).
    """

    offload_enabled: bool = False
    weight_function: list[Callable[[torch.Tensor], torch.Tensor]]
    bias_function: list[Callable[[torch.Tensor], torch.Tensor]]

    def _init_offload(self) -> None:
        """Initialise mixin state (call from ``__init__``)."""
        self.offload_enabled = False
        self.weight_function = []
        self.bias_function = []
        self._cached_weight: torch.Tensor | None = None
        self._cached_bias: torch.Tensor | None = None

    def cast_weight(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transfer weight (+ bias) to *device*/*dtype* and apply transforms.

        The transferred tensors are cached in ``_cached_weight`` /
        ``_cached_bias`` so :meth:`release_weight` can free them.
        """
        # Transfer weight.
        weight: torch.Tensor = self.weight  # type: ignore[attr-defined]
        if weight.device != device or weight.dtype != dtype:
            weight = weight.to(device=device, dtype=dtype, non_blocking=True)
        else:
            weight = weight.clone() if self.weight_function else weight

        for fn in self.weight_function:
            weight = fn(weight)

        self._cached_weight = weight

        # Transfer bias.
        bias: torch.Tensor | None = getattr(self, "bias", None)
        if bias is not None:
            if bias.device != device or bias.dtype != dtype:
                bias = bias.to(device=device, dtype=dtype, non_blocking=True)
            else:
                bias = bias.clone() if self.bias_function else bias
            for fn in self.bias_function:
                bias = fn(bias)

        self._cached_bias = bias
        return weight, bias

    def release_weight(self) -> None:
        """Release cached GPU copies of weight and bias."""
        self._cached_weight = None
        self._cached_bias = None


class OffloadLinear(nn.Linear, OffloadMixin):
    """``nn.Linear`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = F.linear(x, weight, bias)
            self.release_weight()
            return result
        return super().forward(x)


class OffloadConv2d(nn.Conv2d, OffloadMixin):
    """``nn.Conv2d`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype,
        )
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = self._conv_forward(x, weight, bias)
            self.release_weight()
            return result
        return super().forward(x)
