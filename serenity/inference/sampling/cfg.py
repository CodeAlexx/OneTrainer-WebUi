"""Classifier-free guidance computation with RescaleCFG and MaHiRo corrections."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "compute_cfg",
    "rescale_cfg",
    "mahiro_correction",
    "apply_cfg",
]


# --------------------------------------------------------------------------- #
# Core CFG
# --------------------------------------------------------------------------- #


def compute_cfg(cond_pred: Tensor, uncond_pred: Tensor, cfg_scale: float) -> Tensor:
    """Standard classifier-free guidance.

    Returns ``uncond + cfg_scale * (cond - uncond)``.
    When *cfg_scale* is 1.0 the unconditional prediction is irrelevant,
    so we short-circuit and return *cond_pred* directly (2x speedup).
    """
    if cfg_scale == 1.0:
        return cond_pred
    return uncond_pred + cfg_scale * (cond_pred - uncond_pred)


# --------------------------------------------------------------------------- #
# RescaleCFG  (arXiv:2305.08891 — Common Diffusion Noise Schedules)
# --------------------------------------------------------------------------- #


def rescale_cfg(
    denoised: Tensor,
    cond_pred: Tensor,
    cfg_scale: float,
    rescale_phi: float,
) -> Tensor:
    """Rescale CFG output to prevent over-saturation.

    *rescale_phi* in [0, 1]:
      - 0.0 = no rescaling (return *denoised* unchanged)
      - 1.0 = fully rescale to match conditional prediction std

    Algorithm from Forge ``rescale_cfg.py`` and the RescaleCFG paper:
    ``x_rescaled = x_cfg * (std_cond / std_cfg)``
    ``result = phi * x_rescaled + (1 - phi) * x_cfg``
    """
    if rescale_phi <= 0.0:
        return denoised

    # Per-sample std over spatial dims (keep batch dim)
    std_cond = torch.std(cond_pred, dim=tuple(range(1, cond_pred.ndim)), keepdim=True)
    std_cfg = torch.std(denoised, dim=tuple(range(1, denoised.ndim)), keepdim=True)

    # Avoid division by zero
    std_cfg = torch.clamp(std_cfg, min=1e-8)

    x_rescaled = denoised * (std_cond / std_cfg)
    return rescale_phi * x_rescaled + (1.0 - rescale_phi) * denoised


# --------------------------------------------------------------------------- #
# MaHiRo post-CFG normalization
# --------------------------------------------------------------------------- #


def mahiro_correction(denoised: Tensor, uncond_pred: Tensor, cfg_scale: float = 1.0) -> Tensor:
    """MaHiRo post-CFG normalization — prevents color shift via cosine similarity.

    Based on Forge ``mahiro.py``:
    1. Compute ``leap = uncond_pred * scale``
    2. Merge ``(leap + cfg) / 2``
    3. Soft-normalize both, compute cosine similarity
    4. Blend ``cfg`` and ``leap`` based on similarity score
    """
    leap = uncond_pred * cfg_scale
    merge = (leap + denoised) / 2.0

    # Soft sqrt normalization (sign-preserving)
    norm_leap = torch.sqrt(leap.abs()) * leap.sign()
    norm_merge = torch.sqrt(merge.abs()) * merge.sign()

    # Global cosine similarity
    sim = F.cosine_similarity(
        norm_leap.flatten(start_dim=1),
        norm_merge.flatten(start_dim=1),
        dim=1,
    ).mean()

    # Scale factor from similarity
    simsc = 2.0 * (sim + 1.0)
    result = (simsc * denoised + (4.0 - simsc) * leap) / 4.0
    return result


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def apply_cfg(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg_scale: float,
    rescale_phi: float = 0.0,
    mahiro: bool = False,
) -> Tensor:
    """Full CFG pipeline: compute_cfg -> optional rescale -> optional mahiro."""
    denoised = compute_cfg(cond_pred, uncond_pred, cfg_scale)

    if rescale_phi > 0.0:
        denoised = rescale_cfg(denoised, cond_pred, cfg_scale, rescale_phi)

    if mahiro:
        denoised = mahiro_correction(denoised, uncond_pred, cfg_scale)

    return denoised
