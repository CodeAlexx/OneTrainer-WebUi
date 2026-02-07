"""Inpainting training utilities: masked loss, conditioning input, prior preservation.

Parity with OneTrainer's inpainting training for SD1.5 (9-channel) and SDXL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class InpaintingConfig:
    """Configuration for inpainting training."""

    enabled: bool = False
    unmasked_weight: float = 0.0
    normalize_masked_area_loss: bool = True
    prior_preservation_weight: float = 0.0
    conditioning_dropout_probability: float = 0.0


def prepare_inpainting_input(
    noisy_latents: Tensor,
    mask: Tensor,
    conditioning_latents: Tensor,
    *,
    mask_channel_dim: int = 1,
) -> Tensor:
    """Concatenate noisy latents, downscaled mask, and masked conditioning image.

    For SD1.5 inpainting this produces a 9-channel input:
    4 channels (noisy latents) + 1 channel (mask) + 4 channels (masked image).

    For SDXL inpainting this also produces a 9-channel input with the same
    structure.

    Parameters
    ----------
    noisy_latents : Tensor
        Noisy latent image, shape ``(B, C, H, W)``.
    mask : Tensor
        Binary or soft mask in latent space, shape ``(B, 1, H, W)``
        or ``(B, H, W)``.
    conditioning_latents : Tensor
        Latent-space encoding of the conditioning (masked) image,
        shape ``(B, C, H, W)``.
    mask_channel_dim : int
        Channel dimension for the mask. If mask is 3D, it is unsqueezed.

    Returns
    -------
    Tensor
        Concatenated input of shape ``(B, C + 1 + C, H, W)``.
    """
    # Ensure mask has channel dimension
    if mask.ndim == 3:
        mask = mask.unsqueeze(mask_channel_dim)

    # Resize mask to match latent spatial dims if needed
    if mask.shape[-2:] != noisy_latents.shape[-2:]:
        mask = F.interpolate(
            mask.float(),
            size=noisy_latents.shape[-2:],
            mode="nearest",
        )

    # Ensure conditioning latents match spatial dims
    if conditioning_latents.shape[-2:] != noisy_latents.shape[-2:]:
        conditioning_latents = F.interpolate(
            conditioning_latents,
            size=noisy_latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return torch.cat([noisy_latents, mask, conditioning_latents], dim=1)


def create_masked_conditioning_image(
    pixel_values: Tensor,
    mask: Tensor,
) -> Tensor:
    """Create the masked conditioning image for inpainting.

    Zero out masked regions so the model learns to inpaint them.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    # Resize mask to match image spatial dims
    if mask.shape[-2:] != pixel_values.shape[-2:]:
        mask = F.interpolate(
            mask.float(),
            size=pixel_values.shape[-2:],
            mode="nearest",
        )

    # Invert mask: 1 = keep, 0 = inpaint
    inv_mask = 1.0 - mask
    return pixel_values * inv_mask


def inpainting_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    config: InpaintingConfig | None = None,
    prior_predicted: Tensor | None = None,
    prior_target: Tensor | None = None,
) -> Tensor:
    """Compute inpainting-aware loss with optional masked weighting.

    If no mask is provided, returns standard MSE loss.  With a mask,
    applies weighted loss where masked regions get full weight and
    unmasked regions get *config.unmasked_weight*.
    """
    if config is None:
        config = InpaintingConfig()

    if mask is None:
        return F.mse_loss(
            predicted.to(dtype=torch.float32),
            target.to(dtype=torch.float32),
        )

    # Ensure mask has correct shape
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != predicted.shape[-2:]:
        mask = F.interpolate(
            mask.float(),
            size=predicted.shape[-2:],
            mode="nearest",
        )

    # Element-wise loss
    losses = F.mse_loss(
        predicted.to(dtype=torch.float32),
        target.to(dtype=torch.float32),
        reduction="none",
    )

    # Apply mask weighting
    clamped_mask = torch.clamp(mask.to(dtype=torch.float32), config.unmasked_weight, 1.0)
    losses = losses * clamped_mask

    if config.normalize_masked_area_loss:
        mean_dims = list(range(1, clamped_mask.ndim))
        losses = losses / clamped_mask.mean(dim=mean_dims, keepdim=True)

    # Prior preservation
    if config.prior_preservation_weight > 0 and prior_predicted is not None and prior_target is not None:
        prior_losses = F.mse_loss(
            prior_predicted.to(dtype=torch.float32),
            prior_target.to(dtype=torch.float32),
            reduction="none",
        )
        inv_mask = 1.0 - clamped_mask
        prior_losses = prior_losses * inv_mask * config.prior_preservation_weight
        if config.normalize_masked_area_loss:
            mean_dims = list(range(1, inv_mask.ndim))
            prior_losses = prior_losses / inv_mask.mean(dim=mean_dims, keepdim=True).clamp(min=1e-8)
        losses = losses + prior_losses

    return losses.mean()


def prepare_latent_mask(
    mask: Tensor,
    latent_shape: tuple[int, ...],
    vae_scale_factor: int = 8,
) -> Tensor:
    """Downscale a pixel-space mask to latent resolution.

    Parameters
    ----------
    mask : Tensor
        Pixel-space mask, shape ``(B, 1, H, W)`` or ``(B, H, W)``.
    latent_shape : tuple
        Target latent shape ``(B, C, H_lat, W_lat)``.
    vae_scale_factor : int
        VAE spatial downscale factor (default 8 for SD1.5/SDXL).

    Returns
    -------
    Tensor
        Mask in latent space, shape ``(B, 1, H_lat, W_lat)``.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    target_h, target_w = latent_shape[-2], latent_shape[-1]
    if mask.shape[-2:] != (target_h, target_w):
        mask = F.interpolate(
            mask.float(),
            size=(target_h, target_w),
            mode="nearest",
        )
    return mask


__all__ = [
    "InpaintingConfig",
    "prepare_inpainting_input",
    "create_masked_conditioning_image",
    "inpainting_loss",
    "prepare_latent_mask",
]
