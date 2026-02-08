"""Area composition for regional prompting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["AreaRegion", "compose_regional_predictions"]

logger = logging.getLogger(__name__)


@dataclass
class AreaRegion:
    """A prompted region within the latent space.

    Attributes:
        mask: Binary or soft mask (B, 1, H, W) defining the region.
        prediction: Model prediction for this region's prompt.
        strength: Blending strength (0.0 to 1.0).
    """

    mask: Any  # Tensor at runtime
    prediction: Any  # Tensor at runtime
    strength: float = 1.0


def compose_regional_predictions(
    regions: list[AreaRegion],
    base_prediction: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Compose multiple regional predictions into a single output.

    Args:
        regions: List of prompted regions with masks and predictions.
        base_prediction: Default prediction for unmasked areas.
        normalize: If True, normalize mask weights to sum to 1.0.

    Returns:
        Composed prediction tensor.
    """
    if not regions:
        return base_prediction

    result = torch.zeros_like(base_prediction)
    total_weight = torch.zeros(
        base_prediction.shape[0],
        1,
        *base_prediction.shape[2:],
        device=base_prediction.device,
        dtype=base_prediction.dtype,
    )

    for region in regions:
        mask = region.mask.to(base_prediction.device, dtype=base_prediction.dtype)
        # Broadcast mask to match prediction spatial dims
        if mask.shape[2:] != base_prediction.shape[2:]:
            mask = torch.nn.functional.interpolate(
                mask,
                size=base_prediction.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        weighted_mask = mask * region.strength
        result += weighted_mask * region.prediction
        total_weight += weighted_mask

    # Fill unmasked areas with base prediction
    if normalize:
        # Where total_weight > 0, divide to normalize; elsewhere use base
        covered = total_weight > 1e-6
        result = torch.where(
            covered, result / total_weight.clamp(min=1e-6), base_prediction
        )
    else:
        uncovered = total_weight < 1e-6
        result = result + uncovered.float() * base_prediction

    return result
