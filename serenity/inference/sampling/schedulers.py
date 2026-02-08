"""Noise sigma schedulers for diffusion sampling — Karras, exponential, AYS, and more."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from enum import Enum

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "SchedulerType",
    "compute_sigmas",
    "normal_scheduler",
    "karras_scheduler",
    "exponential_scheduler",
    "sgm_uniform_scheduler",
    "simple_scheduler",
    "ddim_uniform_scheduler",
    "beta_scheduler",
    "linear_quadratic_scheduler",
    "ays_scheduler",
]


class SchedulerType(str, Enum):
    """Supported noise schedule types."""

    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    BETA = "beta"
    LINEAR_QUADRATIC = "linear_quadratic"
    AYS = "ays"


# --------------------------------------------------------------------------- #
# Individual schedulers — each returns a 1-D tensor of length (n + 1),
# monotonically decreasing with sigmas[-1] == 0.
# --------------------------------------------------------------------------- #


def normal_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """Linear interpolation in log-sigma space (the 'normal' schedule)."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    # n evenly spaced timesteps in log space, then convert back
    log_sigmas = torch.linspace(log_max, log_min, n)
    sigmas = log_sigmas.exp()
    # Append zero
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def karras_scheduler(
    n: int, sigma_min: float, sigma_max: float, rho: float = 7.0
) -> Tensor:
    """Karras et al. 2022 — power-function schedule.

    ``sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho``
    """
    ramp = torch.linspace(0, 1, n)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def exponential_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """Exponential schedule — evenly spaced in log space."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def sgm_uniform_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """SGM Uniform — linear in log space, excludes the endpoint (then adds zero)."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    # n+1 points, drop the last (sigma_min end), then append 0
    log_sigmas = torch.linspace(log_max, log_min, n + 1)[:-1]
    sigmas = log_sigmas.exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def simple_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """Simple linear interpolation between sigma_max and sigma_min."""
    sigmas = torch.linspace(sigma_max, sigma_min, n)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def ddim_uniform_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """DDIM-style uniform spacing — evenly spaced in sigma space, reversed."""
    sigmas = torch.linspace(sigma_max, sigma_min, n)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def beta_scheduler(
    n: int,
    sigma_min: float,
    sigma_max: float,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> Tensor:
    """Beta distribution schedule (Lee et al. 2024 — 'Beta Sampling is All You Need').

    Uses the beta CDF to concentrate steps where they matter most.
    """
    try:
        from scipy import stats as sp_stats
    except ImportError as exc:
        raise ImportError(
            "beta_scheduler requires scipy. Install with: pip install scipy"
        ) from exc

    # Quantile positions from the beta distribution
    ts = 1.0 - np.linspace(0, 1, n, endpoint=False)
    # Map through beta CDF inverse (PPF)
    betas_ppf = sp_stats.beta.ppf(ts, alpha, beta)
    # Scale to sigma range in log space
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    log_sigmas = log_max + betas_ppf * (log_min - log_max)
    sigmas = np.exp(log_sigmas)
    result = torch.from_numpy(sigmas.copy()).float()
    return torch.cat([result, result.new_zeros(1)])


def linear_quadratic_scheduler(
    n: int,
    sigma_min: float,
    sigma_max: float,
    threshold_noise: float = 0.025,
) -> Tensor:
    """Linear-quadratic schedule — linear near zero, quadratic for higher noise.

    Based on Forge ``linear_quadratic`` scheduler.
    """
    if n == 1:
        sigma_schedule = torch.tensor([1.0, 0.0])
    else:
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i**2) + linear_coef * i + const
            for i in range(linear_steps, n)
        ]
        sigma_schedule_list = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule_list = [1.0 - x for x in sigma_schedule_list]
        sigma_schedule = torch.tensor(sigma_schedule_list, dtype=torch.float32)

    return sigma_schedule * sigma_max


def _loglinear_interp(t_steps: list[float], num_steps: int) -> np.ndarray:
    """Log-linear interpolation of a decreasing sequence."""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(np.array(t_steps[::-1]))
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    return np.exp(new_ys)[::-1].copy()


def ays_scheduler(
    n: int,
    sigma_min: float,
    sigma_max: float,
    model_type: str = "sd15",
) -> Tensor:
    """Align Your Steps (NVIDIA 2024) — optimized sigma schedules per model type.

    *model_type*: ``'sd15'`` (default) or ``'sdxl'``.
    """
    if model_type == "sdxl":
        ref_sigmas: list[float] = [
            sigma_max,
            sigma_max / 2.314,
            sigma_max / 3.875,
            sigma_max / 6.701,
            sigma_max / 10.89,
            sigma_max / 16.954,
            sigma_max / 26.333,
            sigma_max / 38.46,
            sigma_max / 62.457,
            sigma_max / 129.336,
            0.029,
        ]
    else:
        # SD 1.5 reference sigmas
        ref_sigmas = [
            sigma_max,
            sigma_max / 2.257,
            sigma_max / 3.785,
            sigma_max / 5.418,
            sigma_max / 7.749,
            sigma_max / 10.469,
            sigma_max / 15.176,
            sigma_max / 22.415,
            sigma_max / 36.629,
            sigma_max / 96.151,
            0.029,
        ]

    if n != len(ref_sigmas):
        sigmas_np = np.append(_loglinear_interp(ref_sigmas, n), [0.0])
    else:
        sigmas_np = np.array(ref_sigmas + [0.0])

    return torch.from_numpy(sigmas_np).float()


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

_SCHEDULER_MAP: dict[SchedulerType, Callable[..., Tensor]] = {
    SchedulerType.NORMAL: normal_scheduler,
    SchedulerType.KARRAS: karras_scheduler,
    SchedulerType.EXPONENTIAL: exponential_scheduler,
    SchedulerType.SGM_UNIFORM: sgm_uniform_scheduler,
    SchedulerType.SIMPLE: simple_scheduler,
    SchedulerType.DDIM_UNIFORM: ddim_uniform_scheduler,
    SchedulerType.BETA: beta_scheduler,
    SchedulerType.LINEAR_QUADRATIC: linear_quadratic_scheduler,
    SchedulerType.AYS: ays_scheduler,
}


def compute_sigmas(
    scheduler: SchedulerType | str,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    **kwargs,
) -> Tensor:
    """Compute a sigma schedule of length ``num_steps + 1`` (final element is 0).

    Parameters
    ----------
    scheduler : SchedulerType or str
        Which noise schedule to use.
    num_steps : int
        Number of denoising steps.
    sigma_min, sigma_max : float
        Noise level bounds.
    **kwargs
        Extra arguments forwarded to the scheduler function
        (e.g. ``rho`` for Karras, ``model_type`` for AYS).
    """
    scheduler = SchedulerType(scheduler)
    fn = _SCHEDULER_MAP[scheduler]
    return fn(num_steps, sigma_min, sigma_max, **kwargs)
