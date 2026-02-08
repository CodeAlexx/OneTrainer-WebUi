"""INT8 quantized operations using torch._int_mm."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.quantization.ops import QuantizedLinear

__all__ = [
    "Int8Linear",
    "INT8_EXCLUDED_BY_ARCH",
    "is_available",
    "quantize_int8_tensorwise",
    "dequantize_int8",
    "stochastic_round_int8_delta",
]

logger = logging.getLogger(__name__)

# Layer name fragments that should remain in full precision.
DEFAULT_DISABLE_FP: list[str] = [
    "img_in",
    "time_in",
    "guidance_in",
    "txt_in",
    "final_layer",
    "patch_embedding",
    "time_embedding",
]

# Per-model exclusion lists for INT8 quantization
# These layers produce quality degradation when quantized
FLUX_INT8_EXCLUDED = [
    "proj_out",
    "x_embedder",
    "t_embedder",
    "y_embedder",
    "context_embedder",
    "final_layer",
]

SDXL_INT8_EXCLUDED = [
    "proj_out",
    "proj_in",
    "time_embed",
    "label_emb",
]

SD3_INT8_EXCLUDED = [
    "proj_out",
    "x_embedder",
    "t_embedder",
    "y_embedder",
    "context_embedder",
]

# Combined map for easy lookup
INT8_EXCLUDED_BY_ARCH: dict[str, list[str]] = {
    "flux_dev": FLUX_INT8_EXCLUDED,
    "flux_schnell": FLUX_INT8_EXCLUDED,
    "sdxl": SDXL_INT8_EXCLUDED,
    "sdxl_refiner": SDXL_INT8_EXCLUDED,
    "sd3": SD3_INT8_EXCLUDED,
}


def is_available() -> bool:
    """Check whether INT8 matmul is supported."""
    return hasattr(torch, "_int_mm")


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def quantize_int8(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """Quantize a float tensor to INT8 given a pre-computed scale."""
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_tensorwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize *x* to INT8 with a single per-tensor scale factor."""
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def quantize_int8_axiswise(
    x: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis (row/column) INT8 quantization."""
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize_int8(
    q: torch.Tensor, scale: float | torch.Tensor
) -> torch.Tensor:
    """Dequantize an INT8 tensor back to float."""
    return q.float() * scale


def stochastic_round_int8_delta(
    x: torch.Tensor,
    scale: float | torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Quantize a delta tensor to INT8 using stochastic rounding.

    Used when merging LoRA into already-quantized weights so that rounding
    error is unbiased.  Results are intentionally non-deterministic across
    different seeds.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    x_scaled = x / scale
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor

    random_vals = torch.rand(
        x_scaled.shape,
        generator=generator,
        device=x.device,
        dtype=x_scaled.dtype,
    )
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# ---------------------------------------------------------------------------
# INT8 Linear layer
# ---------------------------------------------------------------------------


class Int8Linear(QuantizedLinear):
    """Linear layer that stores weights in INT8 and uses ``torch._int_mm``.

    For batch sizes > 16 the fast ``_int_mm`` path is used.  Smaller batches
    fall back to dequantize-then-matmul to avoid poor GPU utilisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, compute_dtype=compute_dtype)
        self.weight_scale: torch.Tensor | float | None = None
        self._is_quantized: bool = False

    # -- quantization helpers ------------------------------------------------

    def quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize *weight* to INT8 and store the result."""
        q_weight, scale = quantize_int8_tensorwise(weight)
        self.weight = nn.Parameter(q_weight, requires_grad=False)
        self.weight_scale = scale
        self._is_quantized = True
        return q_weight, scale

    def dequantize_weight(self) -> torch.Tensor:
        if self._is_quantized and self.weight_scale is not None:
            return dequantize_int8(self.weight, self.weight_scale).to(self.compute_dtype)
        return self.weight.to(self.compute_dtype)

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            return super().forward(x)

        device = x.device
        weight = self.weight.to(device, non_blocking=True)
        bias = self.bias.to(device, non_blocking=True) if self.bias is not None else None

        w_scale = self.weight_scale
        if isinstance(w_scale, torch.Tensor):
            w_scale = w_scale.to(device, non_blocking=True)

        compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])

        if x_2d.shape[0] > 16 and is_available():
            # Fast INT8 path
            x_8, x_scale = quantize_int8_axiswise(x_2d, dim=-1)
            res = torch._int_mm(x_8, weight.T)  # type: ignore[attr-defined]
            y = res.float().mul_(w_scale * x_scale).to(compute_dtype)
        else:
            # Small-batch fallback
            w_float = dequantize_int8(weight, w_scale).to(x.dtype)
            y = torch.nn.functional.linear(x_2d, w_float)

        if bias is not None:
            y = y + bias.to(compute_dtype)

        return y.reshape(*x_shape[:-1], y.shape[-1])
