"""Mask loading, generation, and augmentation for training datasets.

Provides mask utilities for loading masks from files, generating default
masks, random circular mask shrinking, and mask-aware rotate/crop
augmentations.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask(
    path: str | Path,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Load a mask image from disk as a [1, H, W] float tensor in [0, 1].

    Supports grayscale and RGBA (uses alpha channel). If height/width are
    provided, the mask is resized to match.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    img = Image.open(path)

    # Use alpha channel if RGBA, otherwise convert to grayscale
    if img.mode == "RGBA":
        mask_img = img.split()[-1]  # alpha channel
    else:
        mask_img = img.convert("L")

    if height is not None and width is not None:
        mask_img = mask_img.resize((width, height), Image.BILINEAR)

    tensor = torch.from_numpy(
        __import__("numpy").array(mask_img, dtype="float32")
    ) / 255.0

    # Ensure [1, H, W]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    return tensor


def load_mask_for_image(
    image_path: Path,
    mask_suffix: str = "-masklabel",
    mask_extension: str = ".png",
) -> torch.Tensor | None:
    """Load the mask file associated with an image, if it exists.

    Convention: mask file is ``{image_stem}{suffix}{ext}``
    in the same directory as the image.
    """
    mask_path = image_path.parent / f"{image_path.stem}{mask_suffix}{mask_extension}"
    if mask_path.exists():
        return load_mask(mask_path)
    return None


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_full_mask(
    height: int,
    width: int,
    value: float = 1.0,
) -> torch.Tensor:
    """Generate a uniform mask of given size.  Returns [1, H, W] float tensor."""
    return torch.full((1, height, width), value, dtype=torch.float32)


def generate_random_mask(
    height: int,
    width: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Generate a random binary mask.

    Each pixel is 1.0 with probability ``threshold``, else 0.0.
    Returns [1, H, W] float tensor.
    """
    return (torch.rand(1, height, width) < threshold).float()


def generate_center_mask(
    height: int,
    width: int,
    fraction: float = 0.5,
) -> torch.Tensor:
    """Generate a centered rectangular mask.

    The center ``fraction`` of each dimension is 1.0, the rest 0.0.
    Returns [1, H, W] float tensor.
    """
    mask = torch.zeros(1, height, width)
    pad_h = int(height * (1.0 - fraction) / 2.0)
    pad_w = int(width * (1.0 - fraction) / 2.0)
    mask[:, pad_h : height - pad_h, pad_w : width - pad_w] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Mask augmentations
# ---------------------------------------------------------------------------

def apply_circular_mask_shrink(
    mask: torch.Tensor,
    shrink_factor_min: float = 0.2,
    shrink_factor_max: float = 1.0,
    probability: float = 1.0,
) -> torch.Tensor:
    """Shrink a mask toward its center of mass using a circular falloff.

    ``mask`` should be [1, H, W] or [H, W].
    """
    if random.random() > probability:
        return mask

    squeeze = False
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        squeeze = True

    _, h, w = mask.shape
    factor = random.uniform(shrink_factor_min, shrink_factor_max)

    if factor >= 1.0:
        return mask.squeeze(0) if squeeze else mask

    # Find center of mass of the mask
    ys = torch.arange(h, dtype=torch.float32, device=mask.device)
    xs = torch.arange(w, dtype=torch.float32, device=mask.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    mask_sum = mask[0].sum().clamp(min=1e-6)
    cy = (mask[0] * grid_y).sum() / mask_sum
    cx = (mask[0] * grid_x).sum() / mask_sum

    # Distance from center of mass, normalized
    dist = torch.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)
    max_dist = dist.max().clamp(min=1e-6)
    norm_dist = dist / max_dist

    # Circular shrink: pixels outside the shrink radius are zeroed
    shrunk = mask.clone()
    shrunk[0] = mask[0] * (norm_dist <= factor).float()

    return shrunk.squeeze(0) if squeeze else shrunk


def apply_mask_rotate_crop(
    mask: torch.Tensor,
    image: torch.Tensor,
    min_size: int = 64,
    max_rotate_angle: float = 20.0,
    min_padding_percent: float = 10.0,
    max_padding_percent: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a random rotation and crop centered on the mask region.

    Performs focused augmentation on masked areas.
    Operates on [1, H, W] mask and [C, H, W] image.

    Returns (cropped_mask, cropped_image).
    """
    _, mh, mw = mask.shape
    c, ih, iw = image.shape
    assert mh == ih and mw == iw, "Mask and image must have the same spatial dimensions"

    # Find bounding box of mask
    nonzero = mask[0].nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return mask, image

    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values

    # Add random padding
    padding = random.uniform(min_padding_percent, max_padding_percent) / 100.0
    box_h = (y_max - y_min).item()
    box_w = (x_max - x_min).item()
    pad_h = int(box_h * padding)
    pad_w = int(box_w * padding)

    crop_y0 = max(0, y_min.item() - pad_h)
    crop_x0 = max(0, x_min.item() - pad_w)
    crop_y1 = min(ih, y_max.item() + 1 + pad_h)
    crop_x1 = min(iw, x_max.item() + 1 + pad_w)

    # Ensure minimum crop size
    crop_h = max(min_size, crop_y1 - crop_y0)
    crop_w = max(min_size, crop_x1 - crop_x0)

    # Re-center if min_size forced expansion
    cy = (crop_y0 + crop_y1) // 2
    cx = (crop_x0 + crop_x1) // 2
    crop_y0 = max(0, cy - crop_h // 2)
    crop_x0 = max(0, cx - crop_w // 2)
    crop_y1 = min(ih, crop_y0 + crop_h)
    crop_x1 = min(iw, crop_x0 + crop_w)

    cropped_mask = mask[:, crop_y0:crop_y1, crop_x0:crop_x1]
    cropped_image = image[:, crop_y0:crop_y1, crop_x0:crop_x1]

    # Apply random rotation
    if max_rotate_angle > 0:
        angle = random.uniform(-max_rotate_angle, max_rotate_angle)
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0)

        for tensor in [cropped_image, cropped_mask]:
            batched = tensor.unsqueeze(0)
            grid = torch.nn.functional.affine_grid(
                theta, batched.size(), align_corners=False
            )
            rotated = torch.nn.functional.grid_sample(
                batched, grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )
            tensor.copy_(rotated.squeeze(0))

    return cropped_mask, cropped_image


def resize_mask_to_latent(
    mask: torch.Tensor,
    scale_factor: float = 0.125,
) -> torch.Tensor:
    """Downscale a mask to match latent space dimensions.

    Default scale_factor of 0.125 (1/8) matches the typical VAE
    downsampling ratio used by SD and Flux models.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
        squeeze = 2
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
        squeeze = 1
    else:
        squeeze = 0

    result = torch.nn.functional.interpolate(
        mask, scale_factor=scale_factor, mode="nearest",
    )

    if squeeze == 2:
        result = result.squeeze(0).squeeze(0)
    elif squeeze == 1:
        result = result.squeeze(0)

    return result


__all__ = [
    "apply_circular_mask_shrink",
    "apply_mask_rotate_crop",
    "generate_center_mask",
    "generate_full_mask",
    "generate_random_mask",
    "load_mask",
    "load_mask_for_image",
    "resize_mask_to_latent",
]
