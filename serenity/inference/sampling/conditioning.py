"""Conditioning format and noise creation for inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "Conditioning",
    "prepare_conditioning",
    "create_noise",
]


@dataclass
class Conditioning:
    """Container for model conditioning tensors.

    Attributes
    ----------
    cond : Tensor
        Positive conditioning (text embeddings).
    uncond : Tensor or None
        Negative conditioning. ``None`` when cfg_scale == 1 (skip uncond).
    pooled : Tensor or None
        Pooled text output for models that need it (SDXL, SD3).
    extra : dict
        Model-specific extras (timestep_cond, image_cond, etc.).
    """

    cond: Tensor
    uncond: Tensor | None = None
    pooled: Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def prepare_conditioning(
    text_embeddings: Tensor,
    negative_embeddings: Tensor | None = None,
    model_type: str = "sd15",
    **kwargs: Any,
) -> Conditioning:
    """Format conditioning for the given model type.

    Parameters
    ----------
    text_embeddings : Tensor
        Positive text encoder output.
    negative_embeddings : Tensor or None
        Negative text encoder output. Pass ``None`` when using cfg_scale=1.
    model_type : str
        Model identifier for type-specific handling.
    **kwargs
        Extra conditioning fields (pooled, timestep_cond, etc.).
    """
    pooled = kwargs.pop("pooled", None)
    extra = dict(kwargs)

    return Conditioning(
        cond=text_embeddings,
        uncond=negative_embeddings,
        pooled=pooled,
        extra=extra,
    )


def create_noise(
    seed: int,
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a reproducible noise tensor from a seed.

    Uses ``torch.Generator`` for deterministic results across runs.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
    return noise.to(device)
