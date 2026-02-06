"""LTX2 model utilities and helpers."""

from __future__ import annotations

from typing import List

import torch

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


def normalize_ltx2_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float,
    reverse: bool = False,
) -> torch.Tensor:
    """Normalize or denormalize LTX2 latents."""

    mean = latents_mean.view(1, -1, 1, 1, 1)
    std = latents_std.view(1, -1, 1, 1, 1)

    if reverse:
        return (latents / scaling_factor) * std + mean

    return ((latents - mean) / std) * scaling_factor


def pack_ltx2_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack latents from [B, C, T, H, W] to [B, S, C*ps*ps*pt]."""

    b, c, t, h, w = latents.shape
    if t % patch_size_t != 0 or h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Latent dimensions must be divisible by patch sizes")

    t2 = t // patch_size_t
    h2 = h // patch_size
    w2 = w // patch_size

    latents = latents.view(b, c, t2, patch_size_t, h2, patch_size, w2, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    latents = latents.reshape(b, t2 * h2 * w2, c * patch_size_t * patch_size * patch_size)
    return latents


def unpack_ltx2_latents(
    packed: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """Unpack latents from [B, S, C*ps*ps*pt] to [B, C, T, H, W]."""

    b, seq_len, hidden = packed.shape
    t2, h2, w2 = frames, height, width

    c = hidden // (patch_size_t * patch_size * patch_size)
    if c <= 0:
        raise ValueError("Invalid hidden dimension for unpacking")

    packed = packed.view(b, t2, h2, w2, c, patch_size_t, patch_size, patch_size)
    packed = packed.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = packed.reshape(b, c, t2 * patch_size_t, h2 * patch_size, w2 * patch_size)
    return latents


def adjust_video_frames(frame_count: int) -> int:
    """Adjust frame count to satisfy LTX2 constraint (frames % 8 == 1)."""

    if frame_count <= 1:
        return 1
    remainder = (frame_count - 1) % 8
    return frame_count - remainder


class LTX2Model(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.LTX2)

    @staticmethod
    def validate_frame_count(frame_count: int) -> int:
        return adjust_video_frames(frame_count)

    @staticmethod
    def adjust_video_frames(frame_count: int) -> int:
        return adjust_video_frames(frame_count)

    @staticmethod
    def get_valid_frame_counts(max_frames: int) -> List[int]:
        return [f for f in range(1, max_frames + 1, 8)]
