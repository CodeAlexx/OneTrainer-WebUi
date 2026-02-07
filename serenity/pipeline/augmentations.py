"""Image augmentation pipeline for training datasets.

Provides per-concept configurable augmentations matching OneTrainer's
augmentation module set: flip, rotate, crop jitter, brightness, contrast,
saturation, and hue adjustments.  Each augmentation can operate in random
(stochastic per sample) or fixed (deterministic per sample) mode.

All transforms operate on [C, H, W] float tensors in [0, 1] or [-1, 1].
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from serenity.core.concept_config import ConceptImageConfig


# ---------------------------------------------------------------------------
# Augmentation config (runtime, not stored)
# ---------------------------------------------------------------------------

@dataclass
class AugmentationResult:
    """Holds the augmented tensor and any metadata produced by augmentation."""

    tensor: torch.Tensor
    crop_offset: tuple[int, int] = (0, 0)
    was_flipped: bool = False


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def apply_random_flip(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
) -> tuple[torch.Tensor, bool]:
    """Horizontal flip: random (50% chance) or fixed (always).

    Returns (tensor, was_flipped).
    """
    if enable_fixed:
        return torch.flip(tensor, dims=[-1]), True
    if enable_random and random.random() < 0.5:
        return torch.flip(tensor, dims=[-1]), True
    return tensor, False


def apply_random_rotate(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
    max_angle: float = 0.0,
) -> torch.Tensor:
    """Rotate the image tensor by a random angle within [-max_angle, max_angle].

    Fixed mode uses max_angle directly.  Random mode samples uniformly.
    Uses affine_grid + grid_sample for differentiable rotation.
    """
    if not enable_random and not enable_fixed:
        return tensor
    if max_angle <= 0.0:
        return tensor

    if enable_fixed:
        angle = max_angle
    else:
        angle = random.uniform(-max_angle, max_angle)

    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Build 2x3 affine matrix for rotation about center
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        dtype=tensor.dtype,
        device=tensor.device,
    ).unsqueeze(0)

    # Need batch dim for grid_sample
    batched = tensor.unsqueeze(0)
    grid = torch.nn.functional.affine_grid(
        theta, batched.size(), align_corners=False
    )
    rotated = torch.nn.functional.grid_sample(
        batched, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )
    return rotated.squeeze(0)


def apply_random_brightness(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
    max_strength: float = 0.0,
) -> torch.Tensor:
    """Adjust brightness by adding a uniform offset.

    Fixed mode uses +max_strength, random mode samples [-max_strength, +max_strength].
    """
    if not enable_random and not enable_fixed:
        return tensor
    if max_strength <= 0.0:
        return tensor

    if enable_fixed:
        offset = max_strength
    else:
        offset = random.uniform(-max_strength, max_strength)

    return tensor + offset


def apply_random_contrast(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
    max_strength: float = 0.0,
) -> torch.Tensor:
    """Adjust contrast by scaling distance from the mean.

    factor = 1 + strength.  Fixed uses +max, random samples [-max, +max].
    """
    if not enable_random and not enable_fixed:
        return tensor
    if max_strength <= 0.0:
        return tensor

    if enable_fixed:
        strength = max_strength
    else:
        strength = random.uniform(-max_strength, max_strength)

    factor = 1.0 + strength
    mean = tensor.mean()
    return (tensor - mean) * factor + mean


def apply_random_saturation(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
    max_strength: float = 0.0,
) -> torch.Tensor:
    """Adjust saturation by interpolating toward grayscale.

    factor = 1 + strength.  Tensor must be [C, H, W] with C=3 (RGB).
    """
    if not enable_random and not enable_fixed:
        return tensor
    if max_strength <= 0.0:
        return tensor
    if tensor.shape[0] != 3:
        return tensor

    if enable_fixed:
        strength = max_strength
    else:
        strength = random.uniform(-max_strength, max_strength)

    factor = 1.0 + strength
    # Luminance weights (ITU-R BT.601)
    gray = (
        0.2989 * tensor[0:1]
        + 0.5870 * tensor[1:2]
        + 0.1140 * tensor[2:3]
    )
    return gray + factor * (tensor - gray)


def apply_random_hue(
    tensor: torch.Tensor,
    enable_random: bool = False,
    enable_fixed: bool = False,
    max_strength: float = 0.0,
) -> torch.Tensor:
    """Shift hue by rotating the RGB color plane.

    max_strength is in [0, 1] range (fraction of full hue cycle).
    Fixed mode uses +max_strength, random samples [-max, +max].
    """
    if not enable_random and not enable_fixed:
        return tensor
    if max_strength <= 0.0:
        return tensor
    if tensor.shape[0] != 3:
        return tensor

    if enable_fixed:
        shift = max_strength
    else:
        shift = random.uniform(-max_strength, max_strength)

    # Convert shift to radians (full cycle = 2*pi)
    angle = shift * 2.0 * math.pi

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rotation matrix in YIQ-like space for hue shift
    # This is a simplified hue rotation that works on RGB directly
    k = 1.0 / 3.0
    sqrt_k = math.sqrt(1.0 / 3.0)

    mat = torch.tensor([
        [cos_a + (1.0 - cos_a) * k, k * (1.0 - cos_a) - sqrt_k * sin_a, k * (1.0 - cos_a) + sqrt_k * sin_a],
        [k * (1.0 - cos_a) + sqrt_k * sin_a, cos_a + (1.0 - cos_a) * k, k * (1.0 - cos_a) - sqrt_k * sin_a],
        [k * (1.0 - cos_a) - sqrt_k * sin_a, k * (1.0 - cos_a) + sqrt_k * sin_a, cos_a + (1.0 - cos_a) * k],
    ], dtype=tensor.dtype, device=tensor.device)

    # Reshape for matrix multiply: [3, H*W]
    c, h, w = tensor.shape
    flat = tensor.reshape(3, -1)
    rotated = mat @ flat
    return rotated.reshape(c, h, w)


def apply_crop_jitter(
    tensor: torch.Tensor,
    crop_h: int,
    crop_w: int,
    enable: bool = True,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Random crop with jitter, or center crop if disabled.

    Expects tensor already scaled to at least crop_h x crop_w.
    Returns (cropped_tensor, (offset_y, offset_x)).
    """
    _, h, w = tensor.shape

    # Clamp to available space
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    if enable and (h > crop_h or w > crop_w):
        offset_y = random.randint(0, h - crop_h)
        offset_x = random.randint(0, w - crop_w)
    else:
        offset_y = (h - crop_h) // 2
        offset_x = (w - crop_w) // 2

    cropped = tensor[:, offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    return cropped, (offset_y, offset_x)


# ---------------------------------------------------------------------------
# Composite augmentation
# ---------------------------------------------------------------------------

def apply_augmentations(
    tensor: torch.Tensor,
    config: ConceptImageConfig,
) -> AugmentationResult:
    """Apply the full augmentation pipeline for a concept's image config.

    Applies augmentations in OneTrainer order:
    1. Random flip
    2. Random rotate
    3. Random brightness
    4. Random contrast
    5. Random saturation
    6. Random hue

    Crop jitter is handled separately (before augmentations) by the
    dataset/bucketing system.
    """
    result = AugmentationResult(tensor=tensor)

    # 1. Flip
    result.tensor, result.was_flipped = apply_random_flip(
        result.tensor,
        enable_random=config.enable_random_flip,
        enable_fixed=config.enable_fixed_flip,
    )

    # 2. Rotate
    result.tensor = apply_random_rotate(
        result.tensor,
        enable_random=config.enable_random_rotate,
        enable_fixed=config.enable_fixed_rotate,
        max_angle=config.random_rotate_max_angle,
    )

    # 3. Brightness
    result.tensor = apply_random_brightness(
        result.tensor,
        enable_random=config.enable_random_brightness,
        enable_fixed=config.enable_fixed_brightness,
        max_strength=config.random_brightness_max_strength,
    )

    # 4. Contrast
    result.tensor = apply_random_contrast(
        result.tensor,
        enable_random=config.enable_random_contrast,
        enable_fixed=config.enable_fixed_contrast,
        max_strength=config.random_contrast_max_strength,
    )

    # 5. Saturation
    result.tensor = apply_random_saturation(
        result.tensor,
        enable_random=config.enable_random_saturation,
        enable_fixed=config.enable_fixed_saturation,
        max_strength=config.random_saturation_max_strength,
    )

    # 6. Hue
    result.tensor = apply_random_hue(
        result.tensor,
        enable_random=config.enable_random_hue,
        enable_fixed=config.enable_fixed_hue,
        max_strength=config.random_hue_max_strength,
    )

    return result


__all__ = [
    "AugmentationResult",
    "apply_augmentations",
    "apply_random_flip",
    "apply_random_rotate",
    "apply_random_brightness",
    "apply_random_contrast",
    "apply_random_saturation",
    "apply_random_hue",
    "apply_crop_jitter",
]
