"""Sampler wrapper — k-diffusion integration with built-in samplers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from serenity.inference.sampling.cfg import apply_cfg
from serenity.inference.sampling.prediction import Prediction

logger = logging.getLogger(__name__)

__all__ = [
    "SamplerType",
    "DenoiseFn",
    "sample",
    "euler_sample",
    "euler_ancestral_sample",
    "dpm_pp_2m_sample",
    "dpm_pp_2m_sde_sample",
    "create_model_fn",
]

# Type alias: (noisy_input, sigma) -> denoised prediction
DenoiseFn = Callable[[Tensor, Tensor], Tensor]

# Step callback: (step, total_steps, sigma, denoised) -> None
StepCallback = Callable[[int, int, Tensor, Tensor], None] | None


class SamplerType(str, Enum):
    """Supported sampler algorithms."""

    EULER = "euler"
    EULER_A = "euler_a"
    DPM_2M = "dpm_2m"
    DPM_2M_SDE = "dpm_2m_sde"
    DPM_PP_2M = "dpm_pp_2m"
    DPM_PP_2M_SDE = "dpm_pp_2m_sde"
    LCM = "lcm"
    HEUN = "heun"
    DEIS = "deis"
    UNIPC = "unipc"


# --------------------------------------------------------------------------- #
# k-diffusion sampler name mapping
# --------------------------------------------------------------------------- #

_KDIFFUSION_MAP: dict[SamplerType, str] = {
    SamplerType.EULER: "sample_euler",
    SamplerType.EULER_A: "sample_euler_ancestral",
    SamplerType.DPM_2M: "sample_dpm_2m",
    SamplerType.DPM_2M_SDE: "sample_dpm_2m_sde",
    SamplerType.DPM_PP_2M: "sample_dpmpp_2m",
    SamplerType.DPM_PP_2M_SDE: "sample_dpmpp_2m_sde",
    SamplerType.LCM: "sample_lcm",
    SamplerType.HEUN: "sample_heun",
    SamplerType.DEIS: "sample_deis",
    SamplerType.UNIPC: "sample_unipc",
}


def _try_kdiffusion_sample(
    sampler_type: SamplerType,
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    callback: StepCallback = None,
    extra_args: dict[str, Any] | None = None,
) -> Tensor | None:
    """Attempt to use k-diffusion. Returns None if unavailable."""
    try:
        import k_diffusion.sampling as k_sampling
    except ImportError:
        return None

    func_name = _KDIFFUSION_MAP.get(sampler_type)
    if func_name is None:
        return None

    fn = getattr(k_sampling, func_name, None)
    if fn is None:
        return None

    # k-diffusion callback adapter
    k_callback = None
    if callback is not None:
        total = len(sigmas) - 1

        def k_callback(info: dict) -> None:
            step = info.get("i", 0)
            sigma = info.get("sigma", sigmas[step] if step < len(sigmas) else sigmas[-1])
            denoised = info.get("denoised", noise)
            callback(step, total, sigma, denoised)

    kwargs: dict[str, Any] = {}
    if extra_args:
        kwargs["extra_args"] = extra_args
    if k_callback is not None:
        kwargs["callback"] = k_callback

    return fn(model_fn, noise, sigmas, **kwargs)


# --------------------------------------------------------------------------- #
# Built-in Euler samplers (fallback)
# --------------------------------------------------------------------------- #


def euler_sample(
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    callback: StepCallback = None,
) -> Tensor:
    """Simple Euler ODE sampler — deterministic, first-order.

    ``x_{i+1} = x_i + (denoised - x_i) / sigma_i * (sigma_{i+1} - sigma_i)``
    """
    x = noise
    total_steps = len(sigmas) - 1

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Ensure sigma is a tensor for model_fn
        sigma_t = sigma.unsqueeze(0) if sigma.ndim == 0 else sigma
        denoised = model_fn(x, sigma_t)

        if callback is not None:
            callback(i, total_steps, sigma, denoised)

        # ODE step: dx = (denoised - x) / sigma * dsigma
        d = (x - denoised) / sigma
        dt = sigma_next - sigma
        x = x + d * dt

    return x


def euler_ancestral_sample(
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    callback: StepCallback = None,
    eta: float = 1.0,
) -> Tensor:
    """Euler ancestral sampler — adds noise at each step for stochastic sampling.

    Uses the SDE formulation with noise injection controlled by *eta*.
    """
    x = noise
    total_steps = len(sigmas) - 1

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_t = sigma.unsqueeze(0) if sigma.ndim == 0 else sigma
        denoised = model_fn(x, sigma_t)

        if callback is not None:
            callback(i, total_steps, sigma, denoised)

        # Ancestral step with noise injection
        sigma_down = sigma_next
        sigma_up = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if sigma_next > 0 and eta > 0:
            # Compute noise injection amount
            sigma_up = (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt() * eta
            sigma_down = (sigma_next**2 - sigma_up**2).sqrt()

        # ODE step to sigma_down
        d = (x - denoised) / sigma
        dt = sigma_down - sigma
        x = x + d * dt

        # Noise injection
        if sigma_up > 0:
            x = x + torch.randn_like(x) * sigma_up

    return x


# --------------------------------------------------------------------------- #
# Built-in DPM++ 2M samplers
# --------------------------------------------------------------------------- #


def dpm_pp_2m_sample(
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    callback: StepCallback = None,
) -> Tensor:
    """DPM++ 2M sampler (Lu et al. 2022) -- deterministic, second-order multistep.

    Uses log-sigma space for the time variable: ``t = -log(sigma)``.
    First step and last step use first-order (Euler-like) updates; subsequent
    steps use second-order corrections from the previous denoised prediction.
    """
    x = noise
    total_steps = len(sigmas) - 1
    old_denoised: Tensor | None = None

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_t = sigma.unsqueeze(0) if sigma.ndim == 0 else sigma
        denoised = model_fn(x, sigma_t)

        if callback is not None:
            callback(i, total_steps, sigma, denoised)

        # Convert to log-sigma time: t = -log(sigma)
        t = -sigma.log()
        t_next = -sigma_next.log() if sigma_next > 0 else t + 1.0  # safe fallback
        h = t_next - t

        if old_denoised is None or sigma_next == 0:
            # First step or final step: first-order update
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised
        else:
            # Second-order multistep correction
            h_last = t - (-sigmas[i - 1].log())
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        old_denoised = denoised

    return x


def dpm_pp_2m_sde_sample(
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    callback: StepCallback = None,
    eta: float = 1.0,
) -> Tensor:
    """DPM++ 2M SDE sampler -- stochastic variant with noise injection.

    Same multistep structure as :func:`dpm_pp_2m_sample` but injects noise
    at each step proportional to the step size, controlled by *eta*.
    """
    x = noise
    total_steps = len(sigmas) - 1
    old_denoised: Tensor | None = None

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_t = sigma.unsqueeze(0) if sigma.ndim == 0 else sigma
        denoised = model_fn(x, sigma_t)

        if callback is not None:
            callback(i, total_steps, sigma, denoised)

        # Convert to log-sigma time
        t = -sigma.log()
        t_next = -sigma_next.log() if sigma_next > 0 else t + 1.0
        h = t_next - t

        if old_denoised is None or sigma_next == 0:
            # First-order update
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised
        else:
            # Second-order multistep correction
            h_last = t - (-sigmas[i - 1].log())
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        # SDE noise injection (skip at final step)
        if sigma_next > 0 and eta > 0:
            # Noise magnitude: based on the change in sigma^2 over the step
            noise_amount = (sigma_next**2 - sigma_next**2 * (-2.0 * h).exp()).clamp(min=0.0).sqrt()
            x = x + torch.randn_like(x) * noise_amount * eta

        old_denoised = denoised

    return x


# --------------------------------------------------------------------------- #
# Samplers that require k-diffusion
# --------------------------------------------------------------------------- #

_BUILTIN_SAMPLERS: set[SamplerType] = {
    SamplerType.EULER,
    SamplerType.EULER_A,
    SamplerType.DPM_PP_2M,
    SamplerType.DPM_PP_2M_SDE,
}


# --------------------------------------------------------------------------- #
# Main sample dispatcher
# --------------------------------------------------------------------------- #


def sample(
    model_fn: DenoiseFn,
    noise: Tensor,
    sigmas: Tensor,
    sampler_type: SamplerType | str = SamplerType.EULER,
    callback: StepCallback = None,
    extra_args: dict[str, Any] | None = None,
) -> Tensor:
    """Run the denoising loop with the specified sampler.

    Tries k-diffusion first for all samplers. For samplers with built-in
    implementations (Euler, Euler_A, DPM++ 2M, DPM++ 2M SDE), falls back
    to the built-in version. For samplers that only exist in k-diffusion,
    raises ``ImportError`` with install instructions.
    """
    sampler_type = SamplerType(sampler_type)

    # Try k-diffusion first for all sampler types
    result = _try_kdiffusion_sample(
        sampler_type, model_fn, noise, sigmas, callback, extra_args
    )
    if result is not None:
        logger.debug("Used k-diffusion sampler: %s", sampler_type.value)
        return result

    # Built-in implementations
    if sampler_type == SamplerType.EULER:
        logger.debug("Using built-in euler_sample")
        return euler_sample(model_fn, noise, sigmas, callback)

    if sampler_type == SamplerType.EULER_A:
        logger.debug("Using built-in euler_ancestral_sample")
        return euler_ancestral_sample(model_fn, noise, sigmas, callback)

    if sampler_type == SamplerType.DPM_PP_2M:
        logger.debug("Using built-in dpm_pp_2m_sample")
        return dpm_pp_2m_sample(model_fn, noise, sigmas, callback)

    if sampler_type == SamplerType.DPM_PP_2M_SDE:
        logger.debug("Using built-in dpm_pp_2m_sde_sample")
        return dpm_pp_2m_sde_sample(model_fn, noise, sigmas, callback)

    # No built-in fallback -- k-diffusion is required
    raise ImportError(
        f"Sampler '{sampler_type.value}' requires the k-diffusion package. "
        f"Install it with: pip install k-diffusion"
    )


# --------------------------------------------------------------------------- #
# Model function wrapper
# --------------------------------------------------------------------------- #


def create_model_fn(
    model: Callable[..., Tensor],
    prediction: Prediction,
    cfg_fn: Callable[..., Tensor] | None = None,
    cond: Tensor | None = None,
    uncond: Tensor | None = None,
    cfg_scale: float = 1.0,
    rescale_phi: float = 0.0,
    mahiro: bool = False,
) -> DenoiseFn:
    """Wrap a model with prediction type and CFG into a single DenoiseFn.

    The returned function has signature ``(noisy_input, sigma) -> denoised``.

    Parameters
    ----------
    model : callable
        The neural network. Called as ``model(x, timestep, cond=cond)``.
    prediction : Prediction
        Prediction type for input scaling and denoised conversion.
    cfg_fn : callable or None
        Custom CFG function. If None, uses ``apply_cfg``.
    cond : Tensor or None
        Positive conditioning.
    uncond : Tensor or None
        Negative conditioning. None skips CFG.
    cfg_scale : float
        CFG guidance scale.
    rescale_phi : float
        RescaleCFG strength.
    mahiro : bool
        Enable MaHiRo post-CFG correction.
    """

    def denoise_fn(x: Tensor, sigma: Tensor) -> Tensor:
        # Scale input for the model
        model_input = prediction.calculate_input(sigma, x)

        # Convert sigma to model timestep
        timestep = prediction.sigma_to_timestep(sigma)

        # Run conditional prediction
        cond_output = model(model_input, timestep, cond=cond)
        cond_denoised = prediction.calculate_denoised(sigma, cond_output, model_input)

        if uncond is None or cfg_scale == 1.0:
            return cond_denoised

        # Run unconditional prediction
        uncond_output = model(model_input, timestep, cond=uncond)
        uncond_denoised = prediction.calculate_denoised(sigma, uncond_output, model_input)

        # Apply CFG
        if cfg_fn is not None:
            return cfg_fn(cond_denoised, uncond_denoised)

        return apply_cfg(
            cond_denoised,
            uncond_denoised,
            cfg_scale,
            rescale_phi=rescale_phi,
            mahiro=mahiro,
        )

    return denoise_fn
