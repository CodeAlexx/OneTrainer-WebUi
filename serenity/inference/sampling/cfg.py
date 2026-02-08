"""Classifier-free guidance computation with RescaleCFG and MaHiRo corrections."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "CFGHookRegistry",
    "CFGHookType",
    "CfgHookFn",
    "apply_cfg",
    "compute_cfg",
    "epsilon_scaling",
    "mahiro_correction",
    "rescale_cfg",
]


# --------------------------------------------------------------------------- #
# Hook system (P4.5)
# --------------------------------------------------------------------------- #


class CFGHookType(str, Enum):
    """Injection points for CFG hook functions."""

    PRE_CFG = "pre_cfg"      # Before CFG computation
    CFG = "cfg"              # After CFG, before post-processing
    POST_CFG = "post_cfg"    # After all processing


# Hook signature: (prediction, sigma, cfg_scale) -> modified_prediction
CfgHookFn = Callable[[Tensor, Tensor, float], Tensor]


class CFGHookRegistry:
    """Registry for pre/post CFG hook functions.

    Hooks are called in registration order at each injection point.
    """

    def __init__(self) -> None:
        self._hooks: dict[CFGHookType, list[CfgHookFn]] = {
            t: [] for t in CFGHookType
        }

    def register(self, hook_type: CFGHookType, fn: CfgHookFn) -> None:
        """Register a hook function at the given injection point."""
        self._hooks[hook_type].append(fn)

    def remove(self, hook_type: CFGHookType, fn: CfgHookFn) -> None:
        """Remove a previously registered hook."""
        self._hooks[hook_type].remove(fn)

    def clear(self) -> None:
        """Remove all registered hooks."""
        for hook_list in self._hooks.values():
            hook_list.clear()

    def run(
        self,
        hook_type: CFGHookType,
        pred: Tensor,
        sigma: Tensor,
        cfg_scale: float,
    ) -> Tensor:
        """Run all hooks for *hook_type*, chaining their outputs."""
        for fn in self._hooks[hook_type]:
            pred = fn(pred, sigma, cfg_scale)
        return pred


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
# Epsilon scaling (P4.2)
# --------------------------------------------------------------------------- #


def epsilon_scaling(noise_pred: Tensor, sigma: Tensor) -> Tensor:
    """Scale noise prediction to prevent color drift.

    Applies ``noise_pred * (1 + sigma^2)^0.5`` which counteracts
    the signal-to-noise ratio change at each timestep.
    """
    scale = (1.0 + sigma ** 2) ** 0.5
    return noise_pred * scale


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
    eps_scaling: bool = False,
    sigma: Tensor | None = None,
    hooks: CFGHookRegistry | None = None,
    distilled_cfg_scale: float = 0.0,
) -> Tensor:
    """Full CFG pipeline: hooks -> compute_cfg -> rescale -> mahiro -> distilled -> eps_scaling.

    Parameters
    ----------
    cond_pred : Tensor
        Conditional model prediction.
    uncond_pred : Tensor
        Unconditional model prediction.
    cfg_scale : float
        Guidance scale.
    rescale_phi : float
        RescaleCFG strength (0 = disabled).
    mahiro : bool
        Enable MaHiRo post-CFG correction.
    eps_scaling : bool
        Apply epsilon scaling post-processing.
    sigma : Tensor or None
        Current noise level, required when *eps_scaling* is True.
    hooks : CFGHookRegistry or None
        Optional hook registry for pre/post CFG injection.
    distilled_cfg_scale : float
        Scaling factor for distilled models (0.0 = disabled).
    """
    # Provide a zero sigma fallback for hook calls when sigma is not given
    _sigma = sigma if sigma is not None else torch.zeros(1, device=cond_pred.device)

    # PRE_CFG hooks
    if hooks is not None:
        cond_pred = hooks.run(CFGHookType.PRE_CFG, cond_pred, _sigma, cfg_scale)

    denoised = compute_cfg(cond_pred, uncond_pred, cfg_scale)

    # CFG hooks (after cfg, before post-processing)
    if hooks is not None:
        denoised = hooks.run(CFGHookType.CFG, denoised, _sigma, cfg_scale)

    if rescale_phi > 0.0:
        denoised = rescale_cfg(denoised, cond_pred, cfg_scale, rescale_phi)

    if mahiro:
        denoised = mahiro_correction(denoised, uncond_pred, cfg_scale)

    if distilled_cfg_scale > 0.0:
        denoised = denoised * distilled_cfg_scale

    if eps_scaling and sigma is not None:
        denoised = epsilon_scaling(denoised, sigma)

    # POST_CFG hooks
    if hooks is not None:
        denoised = hooks.run(CFGHookType.POST_CFG, denoised, _sigma, cfg_scale)

    return denoised
