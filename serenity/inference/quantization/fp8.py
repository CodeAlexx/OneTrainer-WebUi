"""FP8 quantized operations using torch._scaled_mm."""

from __future__ import annotations

import logging

import torch

from serenity.inference.quantization.ops import QuantizedLinear

__all__ = ["Fp8Linear", "is_available"]

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check whether FP8 scaled matmul is supported."""
    return hasattr(torch, "_scaled_mm")


class Fp8Linear(QuantizedLinear):
    """Linear layer that uses ``torch._scaled_mm`` for FP8 matmul.

    Weights are expected to be stored in ``torch.float8_e4m3fn``.  Inputs
    are dynamically cast and clamped to the FP8 range before the scaled
    matmul.
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
        self.scale_weight: torch.Tensor | None = None

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize FP8 weight to compute_dtype."""
        w = self.weight
        if w.dtype == torch.float8_e4m3fn:
            scale = self.scale_weight
            if scale is not None:
                return w.to(torch.float32) * scale.to(w.device)
            return w.to(self.compute_dtype)
        return w.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not is_available() or self.weight.dtype != torch.float8_e4m3fn:
            return super().forward(x)

        input_dtype = x.dtype
        fp8_dtype = torch.float8_e4m3fn

        # Handle 2D and 3D inputs
        tensor_2d = False
        if x.ndim == 2:
            tensor_2d = True
            x = x.unsqueeze(1)

        input_shape = x.shape

        # Prepare weight (transposed for _scaled_mm)
        w = self.weight.to(x.device).t().contiguous()

        # Prepare scales
        if self.scale_weight is not None:
            scale_weight = self.scale_weight.to(x.device)
        else:
            scale_weight = torch.ones((), device=x.device, dtype=torch.float32)
        scale_input = torch.ones((), device=x.device, dtype=torch.float32)

        # Clamp and cast input to FP8
        x = torch.clamp(x, min=-448, max=448)
        x = x.reshape(-1, input_shape[-1]).to(fp8_dtype).contiguous()

        # Scaled matmul
        out = torch._scaled_mm(  # type: ignore[attr-defined]
            x,
            w,
            out_dtype=input_dtype,
            bias=self.bias.to(input_dtype) if self.bias is not None else None,
            scale_a=scale_input,
            scale_b=scale_weight,
        )

        if isinstance(out, tuple):
            out = out[0]

        if tensor_2d:
            return out.reshape(input_shape[0], -1)

        return out.reshape(-1, input_shape[1], self.weight.shape[0])
