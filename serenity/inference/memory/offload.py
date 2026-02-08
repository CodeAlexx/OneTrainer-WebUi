"""Per-layer weight offloading — transfer weights to GPU on demand during forward."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from serenity.inference.memory.streams import CastBuffer, StreamPool

__all__ = [
    "OffloadMixin",
    "OffloadLinear",
    "OffloadConv1d",
    "OffloadConv2d",
    "OffloadConv3d",
    "OffloadConvTranspose1d",
    "OffloadConvTranspose2d",
    "OffloadGroupNorm",
    "OffloadLayerNorm",
    "OffloadRMSNorm",
    "OffloadEmbedding",
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
        self._stream_pool: StreamPool | None = None
        self._cast_buffer: CastBuffer | None = None

    def set_stream_pool(self, pool: StreamPool, buffer: CastBuffer | None = None) -> None:
        """Attach a stream pool (and optional cast buffer) for async transfers."""
        self._stream_pool = pool
        self._cast_buffer = buffer

    def cast_weight(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transfer weight (+ bias) to *device*/*dtype* and apply transforms.

        When a :class:`StreamPool` is attached (via :meth:`set_stream_pool`),
        the device transfer is executed on a dedicated CUDA stream so
        computation of layer *N* can overlap with the transfer of layer *N+1*.

        The transferred tensors are cached in ``_cached_weight`` /
        ``_cached_bias`` so :meth:`release_weight` can free them.
        """
        weight: torch.Tensor = self.weight  # type: ignore[attr-defined]
        bias: torch.Tensor | None = getattr(self, "bias", None)

        needs_transfer = (weight.device != device or weight.dtype != dtype)

        if needs_transfer and self._stream_pool is not None:
            stream = self._stream_pool.get_stream()
            if stream is not None:
                with torch.cuda.stream(stream):
                    weight = weight.to(device=device, dtype=dtype, non_blocking=True)
                    if bias is not None:
                        bias = bias.to(device=device, dtype=dtype, non_blocking=True)
            else:
                weight = weight.to(device=device, dtype=dtype, non_blocking=True)
                if bias is not None:
                    bias = bias.to(device=device, dtype=dtype, non_blocking=True)
        elif needs_transfer:
            weight = weight.to(device=device, dtype=dtype, non_blocking=True)
            if bias is not None:
                bias = bias.to(device=device, dtype=dtype, non_blocking=True)
        else:
            weight = weight.clone() if self.weight_function else weight
            if bias is not None:
                bias = bias.clone() if self.bias_function else bias

        for fn in self.weight_function:
            weight = fn(weight)
        if bias is not None:
            for fn in self.bias_function:
                bias = fn(bias)

        self._cached_weight = weight
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


class OffloadConv1d(nn.Conv1d, OffloadMixin):
    """``nn.Conv1d`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
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
            result = F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            self.release_weight()
            return result
        return super().forward(x)


class OffloadConv3d(nn.Conv3d, OffloadMixin):
    """``nn.Conv3d`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        dilation: int | tuple[int, int, int] = 1,
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
            result = F.conv3d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            self.release_weight()
            return result
        return super().forward(x)


class OffloadConvTranspose1d(nn.ConvTranspose1d, OffloadMixin):
    """``nn.ConvTranspose1d`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation,
            padding_mode=padding_mode, device=device, dtype=dtype,
        )
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = F.conv_transpose1d(
                x, weight, bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation,
            )
            self.release_weight()
            return result
        return super().forward(x)


class OffloadConvTranspose2d(nn.ConvTranspose2d, OffloadMixin):
    """``nn.ConvTranspose2d`` with optional per-forward weight offloading."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation,
            padding_mode=padding_mode, device=device, dtype=dtype,
        )
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = F.conv_transpose2d(
                x, weight, bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation,
            )
            self.release_weight()
            return result
        return super().forward(x)


class OffloadGroupNorm(nn.GroupNorm, OffloadMixin):
    """``nn.GroupNorm`` with optional per-forward weight offloading."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine, device=device, dtype=dtype)
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = F.group_norm(x, self.num_groups, weight, bias, self.eps)
            self.release_weight()
            return result
        return super().forward(x)


class OffloadLayerNorm(nn.LayerNorm, OffloadMixin):
    """``nn.LayerNorm`` with optional per-forward weight offloading."""

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
            bias=bias, device=device, dtype=dtype,
        )
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function or self.bias_function:
            weight, bias = self.cast_weight(x.device, x.dtype)
            result = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            self.release_weight()
            return result
        return super().forward(x)


class OffloadRMSNorm(nn.Module, OffloadMixin):
    """RMSNorm with optional per-forward weight offloading."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.eps = eps
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function:
            weight, _bias = self.cast_weight(x.device, x.dtype)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class OffloadEmbedding(nn.Embedding, OffloadMixin):
    """``nn.Embedding`` with optional per-forward weight offloading."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx,
            max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
            _weight=_weight, _freeze=_freeze, device=device, dtype=dtype,
        )
        self._init_offload()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.offload_enabled or self.weight_function:
            weight, _bias = self.cast_weight(x.device, self.weight.dtype)
            result = F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse,
            )
            self.release_weight()
            return result
        return super().forward(x)
