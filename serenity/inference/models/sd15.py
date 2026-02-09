"""SD 1.5 model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter, log_state_dict_info, place_model
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "SD15Adapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SD 1.5 UNet configuration
# ---------------------------------------------------------------------------

_SD15_UNET_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "cross_attention_dim": 768,
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "attention_head_dim": 8,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SD15Adapter(BaseModelAdapter):
    """Model adapter for Stable Diffusion 1.5."""

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
        """Instantiate a SD 1.5 UNet and load weights.

        Requires ``diffusers`` to be installed.
        """
        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_ldm_unet_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        try:
            from diffusers.models import UNet2DConditionModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "SD15Adapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info("Creating SD 1.5 UNet on %s (%s)", device, dtype)

        # Extract UNet keys and convert from LDM to diffusers format
        unet_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        unet_sd = convert_ldm_unet_to_diffusers(unet_sd)

        model = UNet2DConditionModel(**_SD15_UNET_CONFIG)
        missing, unexpected, mismatched = safe_load_state_dict(model, unet_sd)
        log_state_dict_info(missing, unexpected, "SD1.5 UNet")

        model = place_model(model, device, dtype, ops_context=kwargs.get("ops_context"))
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_l"]

    def get_prediction_type(self) -> str:
        return "eps"

    def get_vae_scaling_factor(self) -> float:
        return 0.18215

    def get_default_resolution(self) -> tuple[int, int]:
        return (512, 512)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare SD 1.5 conditioning.

        SD 1.5 uses a single CLIP-L text encoder.  The conditioning is
        simply the hidden-state sequence passed as cross-attention context.
        """
        cond = text_outputs.get("cond")
        if cond is None:
            return text_outputs

        return {"encoder_hidden_states": cond}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.SD15: SD15Adapter}
