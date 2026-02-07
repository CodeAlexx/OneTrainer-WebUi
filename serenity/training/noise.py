"""Noise creation and timestep sampling distributions for diffusion training.

Parity with OneTrainer's ModelSetupNoiseMixin.
"""

from __future__ import annotations

import math

import torch
from torch import Generator, Tensor


# --------------------------------------------------------------------------- #
# 2.5 - Noise Creation
# --------------------------------------------------------------------------- #


def _compute_offset_noise_psi_schedule(betas: Tensor) -> Tensor:
    """Compute time-dependent psi_t for generalized offset noise.

    Follows "Generalized Diffusion Model with Adjusted Offset Noise"
    Equation (34) / Algorithm 1 (balanced-phi_t, psi_t strategy).
    """
    betas = betas.to(torch.float64)
    T = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
    )

    gammas = torch.zeros(T, device=betas.device, dtype=betas.dtype)
    gammas[0] = 1.0
    cumulative_sum_term = gammas[0] / torch.sqrt(alphas_cumprod_prev[0])

    for t in range(1, T):
        alpha_t = alphas[t]
        alpha_cumprod_tm1 = alphas_cumprod_prev[t]
        c_t_denom = alpha_t * (1 - alpha_cumprod_tm1)
        c_t = (1 - alpha_t) * torch.sqrt(alpha_cumprod_tm1) / c_t_denom
        gammas[t] = c_t * cumulative_sum_term
        cumulative_sum_term += gammas[t] / torch.sqrt(alphas_cumprod_prev[t])

    psi_T_denom = torch.sqrt(1 - alphas_cumprod[-1])
    psi_T = cumulative_sum_term / psi_T_denom
    gammas_normalized = gammas / psi_T

    terms = gammas_normalized / torch.sqrt(alphas_cumprod_prev)
    s_cumulative = torch.cumsum(terms, dim=0)
    psi_schedule = s_cumulative / torch.sqrt(1 - alphas_cumprod)
    return psi_schedule


def create_noise(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
    *,
    offset_noise_weight: float = 0.0,
    perturbation_noise_weight: float = 0.0,
    generalized_offset_noise: bool = False,
    timestep: Tensor | None = None,
    betas: Tensor | None = None,
    psi_schedule: Tensor | None = None,
) -> Tensor:
    """Create noise tensor with optional offset and perturbation noise.

    Args:
        shape: Shape of the noise tensor (B, C, ...).
        device: Target device.
        dtype: Target dtype.
        generator: Optional RNG for reproducibility.
        offset_noise_weight: Strength of channel-wise offset noise.
        perturbation_noise_weight: Strength of full-shape perturbation noise.
        generalized_offset_noise: Use time-dependent psi_t scaling.
        timestep: Required when generalized_offset_noise=True.
        betas: Beta schedule; required to compute psi_schedule if not provided.
        psi_schedule: Pre-computed psi schedule (avoids recomputation).

    Returns:
        Noise tensor of the requested shape.
    """
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    if offset_noise_weight > 0:
        # Channel-only noise: (B, C, 1, 1, ...)
        offset_shape = (shape[0], shape[1], *([1] * (len(shape) - 2)))
        offset_noise = torch.randn(offset_shape, generator=generator, device=device, dtype=dtype)

        if generalized_offset_noise and timestep is not None and betas is not None:
            if psi_schedule is None:
                psi_schedule = _compute_offset_noise_psi_schedule(betas).to(timestep.device)
            psi_t = psi_schedule[timestep]
            psi_t = psi_t.view(psi_t.shape[0], *([1] * (len(shape) - 1)))
            noise = noise + (psi_t * offset_noise_weight * offset_noise)
        else:
            noise = noise + (offset_noise_weight * offset_noise)

    if perturbation_noise_weight > 0:
        perturbation = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        noise = noise + (perturbation_noise_weight * perturbation)

    return noise


# --------------------------------------------------------------------------- #
# 2.6 - Timestep Distributions
# --------------------------------------------------------------------------- #


def sample_timesteps(
    distribution_type: str,
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    generator: Generator | None = None,
    *,
    min_noising_strength: float = 0.0,
    max_noising_strength: float = 1.0,
    noising_bias: float = 0.0,
    noising_weight: float = 0.0,
    timestep_shift: float = 1.0,
) -> Tensor:
    """Sample discrete timesteps from a specified distribution.

    Args:
        distribution_type: One of UNIFORM, SIGMOID, LOGIT_NORMAL, HEAVY_TAIL,
            COS_MAP, INVERTED_PARABOLA (matching TimestepDistribution enum).
        batch_size: Number of timesteps to sample.
        num_timesteps: Total training timesteps (e.g. 1000).
        device: Target device.
        generator: Optional RNG.
        min_noising_strength: Minimum timestep fraction (0-1).
        max_noising_strength: Maximum timestep fraction (0-1).
        noising_bias: Distribution bias parameter.
        noising_weight: Distribution weight/scale parameter.
        timestep_shift: Shift factor for timestep warping.

    Returns:
        Integer timestep tensor of shape (batch_size,).
    """
    dist = distribution_type.upper()
    shift = timestep_shift

    min_t = int(num_timesteps * min_noising_strength)
    max_t = int(num_timesteps * max_noising_strength)
    num_t = max_t - min_t

    if dist in ("UNIFORM", "LOGIT_NORMAL", "HEAVY_TAIL"):
        # Continuous implementations with shift applied after
        if dist == "UNIFORM":
            timestep = min_t + (max_t - min_t) * torch.rand(
                batch_size, generator=generator, device=device,
            )
        elif dist == "LOGIT_NORMAL":
            bias = noising_bias
            scale = noising_weight + 1.0
            normal = torch.normal(
                bias, scale, size=(batch_size,), generator=generator, device=device,
            )
            logit_normal = normal.sigmoid()
            timestep = logit_normal * num_t + min_t
        else:  # HEAVY_TAIL
            scale = noising_weight
            u = torch.rand(size=(batch_size,), generator=generator, device=device)
            u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
            timestep = u * num_t + min_t

        # Apply shift
        timestep = num_timesteps * shift * timestep / ((shift - 1) * timestep + num_timesteps)

    elif dist in ("SIGMOID", "COS_MAP", "INVERTED_PARABOLA"):
        # Discrete: build weight arrays with shift, then multinomial sample
        linspace = torch.linspace(0, 1, num_t)
        linspace_shifted = linspace / (shift - shift * linspace + linspace)

        linspace_derivative = torch.linspace(0, 1, num_t)
        linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

        if dist == "COS_MAP":
            weights = 2.0 / (math.pi - 2.0 * math.pi * linspace_shifted + 2.0 * math.pi * linspace_shifted ** 2.0)
            weights *= linspace_derivative
        elif dist == "SIGMOID":
            bias = noising_bias + 0.5
            w = noising_weight
            vals = linspace_shifted
            weights = 1 / (1 + torch.exp(-w * (vals - bias)))
            weights *= linspace_derivative
        elif dist == "INVERTED_PARABOLA":
            bias = noising_bias + 0.5
            w = noising_weight
            weights = torch.clamp(-w * ((linspace_shifted - bias) ** 2) + 2, min=0.0)
            weights *= linspace_derivative
        else:
            weights = torch.ones(num_t)

        weights = weights.to(device=device)
        samples = torch.multinomial(weights, num_samples=batch_size, replacement=True, generator=generator)
        timestep = (samples + min_t).to(dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown timestep distribution: {distribution_type}")

    return timestep.int()


def sample_continuous_timesteps(
    distribution_type: str,
    batch_size: int,
    device: torch.device,
    generator: Generator | None = None,
    *,
    min_noising_strength: float = 0.0,
    max_noising_strength: float = 1.0,
    noising_bias: float = 0.0,
    noising_weight: float = 0.0,
    timestep_shift: float = 1.0,
) -> Tensor:
    """Sample continuous timesteps in [0, 1] for flow matching models.

    Internally discretizes to 10000 steps then normalizes.
    """
    discrete_steps = 10000
    discrete = sample_timesteps(
        distribution_type=distribution_type,
        batch_size=batch_size,
        num_timesteps=discrete_steps,
        device=device,
        generator=generator,
        min_noising_strength=min_noising_strength,
        max_noising_strength=max_noising_strength,
        noising_bias=noising_bias,
        noising_weight=noising_weight,
        timestep_shift=timestep_shift,
    ) + 1
    return discrete.float() / discrete_steps


__all__ = [
    "create_noise",
    "sample_timesteps",
    "sample_continuous_timesteps",
]
