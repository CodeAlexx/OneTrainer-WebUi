"""Wan 2.2 model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "WanAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class WanAdapter(BaseModelAdapter):
    """Model adapter for Wan 2.2 (T2V and I2V variants).

    Wan is a video/image generation model using a custom DiT architecture
    with a T5-XXL text encoder and discrete flow-matching prediction.
    """

    def __init__(self, variant: str = "t2v") -> None:
        if variant not in ("t2v", "i2v"):
            raise ValueError(f"WanAdapter variant must be 't2v' or 'i2v', got {variant!r}")
        self.variant = variant

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.WAN

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate a Wan 2.2 transformer and load weights.

        Requires ``diffusers`` to be installed.
        """
        try:
            from diffusers.models import Transformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "WanAdapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info(
            "Creating Wan 2.2 (%s) transformer on %s (%s)",
            self.variant,
            device,
            dtype,
        )
        # Wan uses a custom 3D DiT; diffusers may provide WanTransformer3DModel
        # or a compatible class.  Fall back to a generic Transformer2DModel
        # for the initial adapter skeleton.
        try:
            from diffusers.models import WanTransformer3DModel  # type: ignore[import-untyped]

            model = WanTransformer3DModel()
        except (ImportError, AttributeError):
            raise NotImplementedError(
                "WanAdapter.create_model requires a diffusers version with "
                "WanTransformer3DModel support."
            )

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["t5_xxl"]

    def get_prediction_type(self) -> str:
        return "flow"

    def get_vae_scaling_factor(self) -> float:
        return 0.18215

    def get_default_resolution(self) -> tuple[int, int]:
        if self.variant == "i2v":
            return (480, 832)
        # T2V default — video frames at 480x832
        return (480, 832)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare Wan conditioning from T5-XXL text encoder outputs.

        Wan uses only T5 hidden states as cross-attention context.
        The I2V variant additionally expects ``image_latents`` passed
        via *kwargs*.
        """
        t5_out = text_outputs.get("cond")
        if t5_out is None:
            t5_out = text_outputs.get("encoder_hidden_states")
        if t5_out is None:
            return text_outputs

        result: dict[str, torch.Tensor] = {"encoder_hidden_states": t5_out}

        # I2V variant: inject image latents when provided
        if self.variant == "i2v":
            image_latents = kwargs.get("image_latents")
            if image_latents is not None and isinstance(image_latents, torch.Tensor):
                result["image_latents"] = image_latents

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.WAN: WanAdapter}
