"""Flux2 transformer wrapper for FLUX.2 Klein models."""

from __future__ import annotations

from typing import Optional

import torch

try:
    from diffusers import FluxTransformer2DModel as _DiffusersFluxTransformer2DModel
except Exception:  # pragma: no cover - optional dependency
    _DiffusersFluxTransformer2DModel = None


if _DiffusersFluxTransformer2DModel is not None:

    class Flux2Transformer2DModel(_DiffusersFluxTransformer2DModel):
        """Compatibility wrapper for FLUX.2 Klein weights."""

        @classmethod
        def from_pretrained_klein(
            cls,
            model_path: str,
            torch_dtype: torch.dtype = torch.bfloat16,
            device: str | torch.device = "cpu",
        ) -> "Flux2Transformer2DModel":
            model = cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            return model.to(device)

else:

    class Flux2Transformer2DModel:  # type: ignore
        """Fallback stub when diffusers is unavailable."""

        @classmethod
        def from_pretrained_klein(
            cls,
            model_path: str,
            torch_dtype: torch.dtype = torch.bfloat16,
            device: str | torch.device = "cpu",
        ) -> "Flux2Transformer2DModel":
            raise ImportError("diffusers is required to load Flux2Transformer2DModel")
