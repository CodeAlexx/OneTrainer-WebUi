"""Diffusion schedule coefficients for epsilon-prediction models.

Pre-computes all standard DDPM quantities from a beta schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiffusionSchedule:
    """Pre-computed diffusion schedule tensors derived from betas.

    Used by SD1.5/SDXL epsilon-prediction models and the VB loss.
    All tensors are 1-D with shape ``(num_timesteps,)``.
    """

    num_timesteps: int
    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    log_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas_cumprod: Tensor
    sqrt_recipm1_alphas_cumprod: Tensor
    posterior_variance: Tensor
    posterior_log_variance_clipped: Tensor
    posterior_mean_coef1: Tensor
    posterior_mean_coef2: Tensor

    @staticmethod
    def from_betas(betas: Tensor) -> DiffusionSchedule:
        """Create schedule from a 1-D beta tensor (e.g. from a noise scheduler)."""
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=alphas_cumprod.dtype, device=betas.device), alphas_cumprod[:-1])
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]).clamp(min=1e-20)
        )

        return DiffusionSchedule(
            num_timesteps=betas.shape[0],
            betas=betas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1 - alphas_cumprod),
            log_one_minus_alphas_cumprod=torch.log(1 - alphas_cumprod),
            sqrt_recip_alphas_cumprod=torch.rsqrt(alphas_cumprod),
            sqrt_recipm1_alphas_cumprod=torch.sqrt(1 / alphas_cumprod - 1),
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=(betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
            posterior_mean_coef2=((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

    @staticmethod
    def from_linear_schedule(
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        device: torch.device | None = None,
    ) -> DiffusionSchedule:
        """Create from a standard linear (scaled_linear) beta schedule."""
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, device=device) ** 2
        return DiffusionSchedule.from_betas(betas)

    def to(self, device: torch.device) -> DiffusionSchedule:
        """Move all tensors to a device (returns a new instance)."""
        return DiffusionSchedule(
            num_timesteps=self.num_timesteps,
            betas=self.betas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            alphas_cumprod_prev=self.alphas_cumprod_prev.to(device),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device),
            log_one_minus_alphas_cumprod=self.log_one_minus_alphas_cumprod.to(device),
            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod.to(device),
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod.to(device),
            posterior_variance=self.posterior_variance.to(device),
            posterior_log_variance_clipped=self.posterior_log_variance_clipped.to(device),
            posterior_mean_coef1=self.posterior_mean_coef1.to(device),
            posterior_mean_coef2=self.posterior_mean_coef2.to(device),
        )


__all__ = [
    "DiffusionSchedule",
]
