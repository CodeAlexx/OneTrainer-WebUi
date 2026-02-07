"""Loss functions, weighting, scaling, and masking for diffusion training.

Parity with OneTrainer's ModelSetupDiffusionLossMixin, masked_loss, and vb_loss.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# 2.1 - Loss Function Enum & Primitives
# --------------------------------------------------------------------------- #


class LossFunction(str, Enum):
    """Supported loss function types."""

    MSE = "MSE"
    MAE = "MAE"
    HUBER = "HUBER"
    LOG_COSH = "LOG_COSH"
    VB = "VB"

    def __str__(self) -> str:
        return self.value


# --- individual loss functions (element-wise, reduction='none') ---


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error (L2), element-wise."""
    return F.mse_loss(
        pred.to(dtype=torch.float32),
        target.to(dtype=torch.float32),
        reduction="none",
    )


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error (L1), element-wise."""
    return F.l1_loss(
        pred.to(dtype=torch.float32),
        target.to(dtype=torch.float32),
        reduction="none",
    )


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss, element-wise."""
    return F.huber_loss(
        pred.to(dtype=torch.float32),
        target.to(dtype=torch.float32),
        reduction="none",
        delta=delta,
    )


def log_cosh_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Log-cosh loss, element-wise. Numerically stable formulation."""
    diff = pred.to(dtype=torch.float32) - target.to(dtype=torch.float32)
    two = torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device)
    return diff + F.softplus(-2.0 * diff) - torch.log(two)


def create_loss_function(loss_type: LossFunction | str) -> callable:
    """Factory returning an element-wise loss callable (pred, target) -> Tensor.

    For HUBER, the returned callable accepts an optional ``delta`` keyword.
    VB loss requires additional coefficients; use :func:`vb_losses` directly.
    """
    if isinstance(loss_type, str):
        loss_type = LossFunction(loss_type.upper())

    _map: dict[LossFunction, callable] = {
        LossFunction.MSE: mse_loss,
        LossFunction.MAE: mae_loss,
        LossFunction.HUBER: huber_loss,
        LossFunction.LOG_COSH: log_cosh_loss,
    }
    fn = _map.get(loss_type)
    if fn is None:
        raise ValueError(f"No simple loss factory for {loss_type}; use vb_losses() for VB loss.")
    return fn


# --------------------------------------------------------------------------- #
# 2.1 cont - Variational Bound (VB) Loss
# --------------------------------------------------------------------------- #


def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """KL divergence between two diagonal Gaussians."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """Fast approximation of the standard normal CDF."""
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: Tensor,
    means: Tensor,
    log_scales: Tensor,
) -> Tensor:
    """Log-likelihood of a Gaussian discretized to uint8 range [-1, 1]."""
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


def _extract_into_tensor(tensor: Tensor, timesteps: Tensor, broadcast_shape: tuple) -> Tensor:
    """Index ``tensor`` by ``timesteps`` and broadcast to ``broadcast_shape``."""
    res = tensor[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res


def _predict_x0_from_eps(
    sqrt_recip_alphas_cumprod: Tensor,
    sqrt_recipm1_alphas_cumprod: Tensor,
    x_t: Tensor,
    t: Tensor,
    eps: Tensor,
) -> Tensor:
    return (
        _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )


def _q_posterior_mean_variance(
    posterior_mean_coef1: Tensor,
    posterior_mean_coef2: Tensor,
    posterior_log_variance_clipped: Tensor,
    x_0: Tensor,
    x_t: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute q(x_{t-1} | x_t, x_0) mean and log-variance."""
    mean = (
        _extract_into_tensor(posterior_mean_coef1, t, x_t.shape) * x_0
        + _extract_into_tensor(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    log_var = _extract_into_tensor(posterior_log_variance_clipped, t, x_t.shape)
    return mean, log_var


def vb_losses(
    coefficients: object,
    x_0: Tensor,
    x_t: Tensor,
    t: Tensor,
    predicted_eps: Tensor,
    predicted_var_values: Tensor,
) -> Tensor:
    """Variational bound loss for learned-variance diffusion models.

    ``coefficients`` must expose: betas, posterior_log_variance_clipped,
    posterior_mean_coef1, posterior_mean_coef2, sqrt_recip_alphas_cumprod,
    sqrt_recipm1_alphas_cumprod  (e.g. a ``DiffusionSchedule``).
    """
    frozen_eps = predicted_eps.detach()

    # True posterior
    pred_x0 = _predict_x0_from_eps(
        coefficients.sqrt_recip_alphas_cumprod,
        coefficients.sqrt_recipm1_alphas_cumprod,
        x_t, t, frozen_eps,
    )
    true_mean, true_log_var = _q_posterior_mean_variance(
        coefficients.posterior_mean_coef1,
        coefficients.posterior_mean_coef2,
        coefficients.posterior_log_variance_clipped,
        pred_x0, x_t, t,
    )

    # Learned variance interpolation
    min_log = _extract_into_tensor(coefficients.posterior_log_variance_clipped, t, x_t.shape)
    max_log = _extract_into_tensor(torch.log(coefficients.betas), t, x_t.shape)
    frac = (predicted_var_values + 1) / 2
    pred_log_var = frac * max_log + (1 - frac) * min_log

    pred_mean, _ = _q_posterior_mean_variance(
        coefficients.posterior_mean_coef1,
        coefficients.posterior_mean_coef2,
        coefficients.posterior_log_variance_clipped,
        pred_x0, x_t, t,
    )

    kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var) / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x=x_0, means=pred_mean, log_scales=0.5 * pred_log_var,
    ) / np.log(2.0)

    t_broad = t.clone()
    while t_broad.dim() < decoder_nll.dim():
        t_broad = t_broad.unsqueeze(-1)
    return torch.where(t_broad == 0, decoder_nll, kl)


# --------------------------------------------------------------------------- #
# 2.2 - Loss Weighting
# --------------------------------------------------------------------------- #


def _compute_snr(
    timesteps: Tensor,
    coefficients: object | None = None,
    alphas_cumprod_fn: callable | None = None,
) -> Tensor:
    """Signal-to-noise ratio from coefficients or callable."""
    if coefficients is not None:
        all_snr = (coefficients.sqrt_alphas_cumprod / coefficients.sqrt_one_minus_alphas_cumprod) ** 2
        return all_snr[timesteps]
    if alphas_cumprod_fn is not None:
        ac = alphas_cumprod_fn(timesteps, 1)
        return ac / (1.0 - ac)
    raise ValueError("Need coefficients or alphas_cumprod_fn to compute SNR")


def min_snr_gamma_weight(
    timesteps: Tensor,
    gamma: float,
    device: torch.device,
    v_prediction: bool = False,
    coefficients: object | None = None,
    alphas_cumprod_fn: callable | None = None,
) -> Tensor:
    """MIN_SNR_GAMMA loss weight."""
    snr = _compute_snr(timesteps, coefficients, alphas_cumprod_fn)
    min_snr = torch.minimum(snr, torch.full_like(snr, gamma))
    denom = snr + (1.0 if v_prediction else 0.0)
    return (min_snr / denom).to(device)


def debiased_estimation_weight(
    timesteps: Tensor,
    device: torch.device,
    v_prediction: bool = False,
    coefficients: object | None = None,
    alphas_cumprod_fn: callable | None = None,
) -> Tensor:
    """Debiased estimation loss weight (Kohya variant)."""
    snr = _compute_snr(timesteps, coefficients, alphas_cumprod_fn)
    weight = snr.clone()
    torch.clip(weight, max=1.0e3, out=weight)
    if v_prediction:
        weight += 1.0
    torch.rsqrt(weight, out=weight)
    return weight.to(device)


def p2_loss_weight(
    timesteps: Tensor,
    gamma: float,
    device: torch.device,
    v_prediction: bool = False,
    coefficients: object | None = None,
    alphas_cumprod_fn: callable | None = None,
) -> Tensor:
    """Perception prioritized (P2) loss weight."""
    snr = _compute_snr(timesteps, coefficients, alphas_cumprod_fn)
    if v_prediction:
        snr = snr + 1.0
    return ((1.0 + snr) ** -gamma).to(device)


def sigma_loss_weight(
    timesteps: Tensor,
    sigmas: Tensor,
    device: torch.device,
) -> Tensor:
    """Sigma-based loss weight for flow matching models."""
    return sigmas[timesteps].to(device=device)


def create_loss_weight(
    weight_type: str,
    timesteps: Tensor,
    device: torch.device,
    *,
    gamma: float = 5.0,
    v_prediction: bool = False,
    coefficients: object | None = None,
    alphas_cumprod_fn: callable | None = None,
    sigmas: Tensor | None = None,
) -> Tensor:
    """Factory: produce per-sample loss weights from a LossWeight enum value.

    ``weight_type`` should match a ``LossWeight`` enum string:
    CONSTANT, MIN_SNR_GAMMA, P2, DEBIASED_ESTIMATION, SIGMA.
    """
    wt = weight_type.upper() if isinstance(weight_type, str) else str(weight_type).upper()

    if wt == "CONSTANT":
        return torch.ones(timesteps.shape[0], device=device)
    elif wt == "MIN_SNR_GAMMA":
        return min_snr_gamma_weight(
            timesteps, gamma, device, v_prediction, coefficients, alphas_cumprod_fn,
        )
    elif wt == "DEBIASED_ESTIMATION":
        return debiased_estimation_weight(
            timesteps, device, v_prediction, coefficients, alphas_cumprod_fn,
        )
    elif wt == "P2":
        return p2_loss_weight(
            timesteps, gamma, device, v_prediction, coefficients, alphas_cumprod_fn,
        )
    elif wt == "SIGMA":
        if sigmas is None:
            raise ValueError("sigmas required for SIGMA weight type")
        return sigma_loss_weight(timesteps, sigmas, device)
    else:
        raise ValueError(f"Unknown loss weight type: {weight_type}")


# --------------------------------------------------------------------------- #
# 2.3 - Loss Scaling
# --------------------------------------------------------------------------- #


def get_loss_scale(
    scaler_type: str,
    batch_size: int,
    accumulation_steps: int,
    world_size: int = 1,
) -> float:
    """Compute a scalar loss multiplier.

    ``scaler_type`` should be a ``LossScaler`` enum string from core.enums.
    Kept as a standalone function for convenience.
    """
    st = scaler_type.upper() if isinstance(scaler_type, str) else str(scaler_type).upper()
    _map = {
        "NONE": 1,
        "BATCH": batch_size,
        "GLOBAL_BATCH": batch_size * world_size,
        "GRADIENT_ACCUMULATION": accumulation_steps,
        "BOTH": accumulation_steps * batch_size,
        "GLOBAL_BOTH": accumulation_steps * batch_size * world_size,
    }
    if st not in _map:
        raise ValueError(f"Unknown loss scaler type: {scaler_type}")
    return float(_map[st])


# --------------------------------------------------------------------------- #
# 2.4 - Masked Loss
# --------------------------------------------------------------------------- #


def masked_loss(
    losses: Tensor,
    mask: Tensor,
    unmasked_weight: float = 0.0,
    normalize_masked_area_loss: bool = True,
) -> Tensor:
    """Apply spatial mask to per-element losses.

    The mask should be broadcastable to ``losses``. Unmasked (zero-mask)
    regions receive ``unmasked_weight`` contribution.
    """
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses = losses * clamped_mask
    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=list(range(1, clamped_mask.ndim)), keepdim=True)
    return losses


def masked_loss_with_prior(
    losses: Tensor,
    prior_losses: Tensor | None,
    mask: Tensor,
    unmasked_weight: float = 0.0,
    normalize_masked_area_loss: bool = True,
    prior_preservation_weight: float = 0.0,
) -> Tensor:
    """Apply spatial mask with optional prior preservation on unmasked regions."""
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses = losses * clamped_mask
    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=list(range(1, clamped_mask.ndim)), keepdim=True)
    if prior_preservation_weight == 0 or prior_losses is None:
        return losses
    inv_mask = 1 - clamped_mask
    prior_losses = prior_losses * inv_mask * prior_preservation_weight
    if normalize_masked_area_loss:
        prior_losses = prior_losses / inv_mask.mean(dim=list(range(1, inv_mask.ndim)), keepdim=True)
    return losses + prior_losses


# --------------------------------------------------------------------------- #
# Legacy API (kept for backward compatibility)
# --------------------------------------------------------------------------- #


def flow_matching_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Flow matching loss (MSE)."""
    return F.mse_loss(pred, target, reduction=reduction)


def snr_weighted_loss(pred: Tensor, target: Tensor, snr: Tensor | None = None) -> Tensor:
    """SNR-weighted MSE loss."""
    loss = F.mse_loss(pred, target, reduction="none")
    if snr is None:
        return loss.mean()
    weight = 1.0 / (snr + 1.0)
    while weight.dim() < loss.dim():
        weight = weight.unsqueeze(-1)
    return (loss * weight).mean()


def velocity_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Velocity prediction loss (MSE)."""
    return F.mse_loss(pred, target, reduction=reduction)


__all__ = [
    # 2.1 - Loss functions
    "LossFunction",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "log_cosh_loss",
    "create_loss_function",
    # VB loss
    "normal_kl",
    "approx_standard_normal_cdf",
    "discretized_gaussian_log_likelihood",
    "vb_losses",
    # 2.2 - Loss weighting
    "min_snr_gamma_weight",
    "debiased_estimation_weight",
    "p2_loss_weight",
    "sigma_loss_weight",
    "create_loss_weight",
    # 2.3 - Loss scaling
    "get_loss_scale",
    # 2.4 - Masked loss
    "masked_loss",
    "masked_loss_with_prior",
    # Legacy
    "flow_matching_loss",
    "snr_weighted_loss",
    "velocity_loss",
]
