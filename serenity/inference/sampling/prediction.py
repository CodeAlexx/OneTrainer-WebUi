"""Prediction types for diffusion model denoising — eps, v, flow, flux, and EDM."""

from __future__ import annotations

import logging
import math
from enum import Enum

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "PredictionType",
    "Prediction",
    "EpsPrediction",
    "VPrediction",
    "FlowPrediction",
    "FluxPrediction",
    "EDMPrediction",
    "ContinuousEDMPrediction",
    "ContinuousVPrediction",
    "DiscreteFlowPrediction",
    "get_prediction",
]


class PredictionType(str, Enum):
    """Supported model prediction types."""

    EPS = "eps"
    V_PREDICTION = "v_prediction"
    FLOW = "flow"
    FLOW_FLUX = "flow_flux"
    EDM = "edm"
    CONTINUOUS_EDM = "continuous_edm"
    CONTINUOUS_V = "continuous_v"
    DISCRETE_FLOW = "discrete_flow"


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class Prediction:
    """Base prediction type — defines the interface for noise/denoised conversion."""

    def __init__(self, sigma_data: float = 1.0) -> None:
        self.sigma_data = sigma_data

    @staticmethod
    def _broadcast_sigma(sigma: Tensor, target: Tensor) -> Tensor:
        """Reshape sigma for broadcasting: (B,) -> (B, 1, 1, ...)."""
        return sigma.view(sigma.shape[:1] + (1,) * (target.ndim - 1))

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        """Scale noisy input for the model (c_in)."""
        sigma = self._broadcast_sigma(sigma, noise)
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor
    ) -> Tensor:
        """Convert raw model output to denoised prediction (c_out)."""
        raise NotImplementedError

    def noise_scaling(
        self, sigma: Tensor, noise: Tensor, latent: Tensor, max_denoise: bool = False,
    ) -> Tensor:
        """Add noise to a latent at the given sigma level."""
        sigma = self._broadcast_sigma(sigma, noise)
        return latent + noise * sigma

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Reverse noise_scaling to recover original noise."""
        raise NotImplementedError

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        """Convert continuous sigma to the discrete timestep the model expects."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Epsilon prediction (SD 1.5, SDXL)
# --------------------------------------------------------------------------- #


class EpsPrediction(Prediction):
    """Epsilon (noise) prediction — standard for SD 1.5 and SDXL."""

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor
    ) -> Tensor:
        sigma = self._broadcast_sigma(sigma, model_output)
        return model_input - model_output * sigma

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Recover noise from scaled = latent + noise * sigma."""
        sigma = self._broadcast_sigma(sigma, scaled)
        return scaled / sigma

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        """Log-space lookup — requires a registered sigma schedule.

        For standalone use, returns sigma directly. Real integrations
        override with a proper log-sigma table lookup.
        """
        return sigma


# --------------------------------------------------------------------------- #
# V-prediction (SD 2.x)
# --------------------------------------------------------------------------- #


class VPrediction(Prediction):
    """V-prediction — used by Stable Diffusion 2.x models."""

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor
    ) -> Tensor:
        sigma = self._broadcast_sigma(sigma, model_output)
        sd2 = self.sigma_data**2
        return (
            model_input * sd2 / (sigma**2 + sd2)
            - model_output * sigma * self.sigma_data / (sigma**2 + sd2) ** 0.5
        )

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Recover noise from scaled = latent + noise * sigma (v-prediction)."""
        sigma = self._broadcast_sigma(sigma, scaled)
        return (scaled - latent) / sigma

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma


# --------------------------------------------------------------------------- #
# Flow prediction (Wan, Qwen, Lumina, ZImage, etc.)
# --------------------------------------------------------------------------- #


class FlowPrediction(Prediction):
    """Discrete flow matching — used by Wan, Qwen, Lumina, ZImage, and similar models."""

    def __init__(self, sigma_data: float = 1.0, shift: float = 1.0, multiplier: float = 1000.0) -> None:
        super().__init__(sigma_data=sigma_data)
        self.shift = shift
        self.multiplier = multiplier

    def _time_snr_shift(self, t: Tensor) -> Tensor:
        """Apply SNR-based time shift: alpha*t / (1 + (alpha-1)*t)."""
        if self.shift == 1.0:
            return t
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        """Flow models use identity input scaling (no c_in division)."""
        return noise

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor
    ) -> Tensor:
        sigma = self._broadcast_sigma(sigma, model_output)
        return model_input - model_output * sigma

    def noise_scaling(
        self, sigma: Tensor, noise: Tensor, latent: Tensor, max_denoise: bool = False,
    ) -> Tensor:
        sigma = self._broadcast_sigma(sigma, noise)
        if max_denoise:
            return noise
        return sigma * noise + (1.0 - sigma) * latent

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Recover noise from scaled = sigma * noise + (1 - sigma) * latent."""
        sigma = self._broadcast_sigma(sigma, scaled)
        return (scaled - (1.0 - sigma) * latent) / sigma.clamp(min=1e-8)

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma * self.multiplier

    def apply_wan_shift(
        self, sigmas: Tensor, high_shift: float = 17.0, low_shift: float = 1.0,
    ) -> Tensor:
        """Apply Wan-style dynamic shift — high shift for noisy, low shift for clean."""
        shifted = torch.where(
            sigmas > 0.5,
            high_shift * sigmas / (1.0 + (high_shift - 1.0) * sigmas),
            low_shift * sigmas / (1.0 + (low_shift - 1.0) * sigmas),
        )
        return shifted


# --------------------------------------------------------------------------- #
# Flux prediction (Flux 1/2 with mu-based exponential sigma shifting)
# --------------------------------------------------------------------------- #


def _calculate_flux_mu(
    seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute mu for Flux exponential sigma shifting based on sequence length.

    Linearly interpolates shift between base_shift and max_shift based on
    how seq_len relates to base_seq_len and max_seq_len.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * seq_len + b
    return float(mu)


def _flux_time_shift_exponential(mu: float, sigma: float, t: Tensor) -> Tensor:
    """Exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)**sigma)."""
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0) ** sigma)


class FluxPrediction(FlowPrediction):
    """Flux-specific flow matching with mu-based exponential sigma shifting."""

    def __init__(
        self,
        sigma_data: float = 1.0,
        mu: float | None = None,
        seq_len: int = 4096,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> None:
        super().__init__(sigma_data=sigma_data, shift=1.0, multiplier=1.0)
        if mu is not None:
            self.mu = mu
        else:
            self.mu = _calculate_flux_mu(
                seq_len=seq_len,
                base_seq_len=base_seq_len,
                max_seq_len=max_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            )

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        """Flux uses identity input scaling."""
        return noise

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        """Flux treats sigma as the timestep directly."""
        return sigma

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Recover noise from scaled = sigma * noise + (1 - sigma) * latent."""
        sigma = self._broadcast_sigma(sigma, scaled)
        return (scaled - (1.0 - sigma) * latent) / sigma.clamp(min=1e-8)

    def apply_sigma_shift(self, sigmas: Tensor) -> Tensor:
        """Apply exponential time shift to a sigma schedule."""
        return _flux_time_shift_exponential(self.mu, 1.0, sigmas)


# --------------------------------------------------------------------------- #
# EDM prediction (Karras et al. 2022)
# --------------------------------------------------------------------------- #


class EDMPrediction(Prediction):
    """EDM (Elucidating Diffusion Models) prediction — Karras et al. 2022."""

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor
    ) -> Tensor:
        sigma = self._broadcast_sigma(sigma, model_output)
        sd2 = self.sigma_data**2
        return (
            model_input * sd2 / (sigma**2 + sd2)
            + model_output * sigma * self.sigma_data / (sigma**2 + sd2) ** 0.5
        )

    def inverse_noise_scaling(
        self, sigma: Tensor, scaled: Tensor, latent: Tensor,
    ) -> Tensor:
        """Recover noise from EDM noise_scaling: scaled = latent + noise * sigma."""
        sigma = self._broadcast_sigma(sigma, scaled)
        return scaled / sigma.clamp(min=1e-8)

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()


# --------------------------------------------------------------------------- #
# Continuous EDM prediction
# --------------------------------------------------------------------------- #


class ContinuousEDMPrediction(EDMPrediction):
    """EDM prediction with raw sigma as timestep (no log transform)."""

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma


# --------------------------------------------------------------------------- #
# Continuous V prediction
# --------------------------------------------------------------------------- #


class ContinuousVPrediction(VPrediction):
    """V-prediction with raw sigma as timestep."""

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma


# --------------------------------------------------------------------------- #
# Discrete flow prediction
# --------------------------------------------------------------------------- #


class DiscreteFlowPrediction(FlowPrediction):
    """Flow prediction with discretized timesteps."""

    def __init__(
        self,
        sigma_data: float = 1.0,
        shift: float = 1.0,
        multiplier: float = 1000.0,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__(sigma_data, shift, multiplier)
        self.num_timesteps = num_timesteps

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return (sigma * self.num_timesteps).round().clamp(0, self.num_timesteps)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_prediction(prediction_type: PredictionType | str, **kwargs) -> Prediction:
    """Create a Prediction instance for the given type."""
    prediction_type = PredictionType(prediction_type)

    factories: dict[PredictionType, type[Prediction]] = {
        PredictionType.EPS: EpsPrediction,
        PredictionType.V_PREDICTION: VPrediction,
        PredictionType.FLOW: FlowPrediction,
        PredictionType.FLOW_FLUX: FluxPrediction,
        PredictionType.EDM: EDMPrediction,
        PredictionType.CONTINUOUS_EDM: ContinuousEDMPrediction,
        PredictionType.CONTINUOUS_V: ContinuousVPrediction,
        PredictionType.DISCRETE_FLOW: DiscreteFlowPrediction,
    }

    cls = factories[prediction_type]
    return cls(**kwargs)
