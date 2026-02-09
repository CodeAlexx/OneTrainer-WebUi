"""Abstract model adapter interface for the inference engine."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "BaseModelAdapter",
    "ModelAdapter",
    "place_model",
    "log_state_dict_info",
]

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract interface for architecture-specific model adapters.

    Each diffusion model family (SD 1.5, SDXL, Flux, etc.) provides a
    concrete adapter that knows how to construct the model, prepare
    conditioning, and expose architecture-specific defaults.
    """

    @property
    @abstractmethod
    def architecture(self) -> ModelArchitecture:
        """The model architecture this adapter handles."""

    @abstractmethod
    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate the model and load *state_dict* into it."""

    @abstractmethod
    def get_text_encoder_types(self) -> list[str]:
        """Return text encoder identifiers, e.g. ``["clip_l"]``."""

    @abstractmethod
    def get_prediction_type(self) -> str:
        """Return the noise-prediction type (``"eps"``, ``"v"``, ``"flow"``)."""

    @abstractmethod
    def get_vae_scaling_factor(self) -> float:
        """Return the VAE latent scaling factor for this architecture."""

    @abstractmethod
    def get_default_resolution(self) -> tuple[int, int]:
        """Return the default ``(height, width)`` for generation."""

    @abstractmethod
    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare model-specific conditioning from text-encoder outputs."""


def place_model(
    model: nn.Module,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    ops_context: object | None = None,
) -> nn.Module:
    """Place model on device/dtype, respecting offload context.

    When ``ops_context`` is provided (offloading active), only dtype is set
    and the ModelManager handles device placement later.
    """
    if ops_context is not None:
        model = model.to(dtype=dtype)
    else:
        model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def log_state_dict_info(
    missing: list[str],
    unexpected: list[str],
    model_name: str,
) -> None:
    """Log missing/unexpected keys from state_dict loading."""
    if missing:
        logger.warning("%s: %d missing keys", model_name, len(missing))
    if unexpected:
        logger.debug("%s: %d unexpected keys", model_name, len(unexpected))


class BaseModelAdapter(ModelAdapter):
    """Concrete base with sensible defaults for common architectures.

    Subclasses only need to override the methods that differ from the
    SD 1.5 defaults.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.SD15

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        raise NotImplementedError(
            "BaseModelAdapter.create_model must be overridden for each architecture."
        )

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_l"]

    def get_prediction_type(self) -> str:
        return "eps"

    def get_vae_scaling_factor(self) -> float:
        return 0.18215

    def get_default_resolution(self) -> tuple[int, int]:
        return (512, 512)

    def get_prediction_kwargs(self) -> dict[str, object]:
        """Return extra kwargs for the denoising prediction call."""
        return {}

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        return text_outputs
