"""Chroma model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter, log_state_dict_info, place_model
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "ChromaAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chroma transformer configuration
# ---------------------------------------------------------------------------

# Chroma is a Flux variant with distilled guidance — it replaces the
# guidance_embeds mechanism with a distilled_guidance_layer, eliminating the
# need for classifier-free guidance and negative prompts.

_CHROMA_CONFIG = {
    "in_channels": 64,
    "num_layers": 19,  # double_blocks (same as Flux)
    "num_single_layers": 38,  # single_blocks (same as Flux)
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": False,
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ChromaAdapter(BaseModelAdapter):
    """Model adapter for Chroma — a Flux variant with distilled guidance.

    Chroma uses T5-XXL only (no CLIP) and does not require negative prompts
    or classifier-free guidance.  The distilled guidance layer handles what
    CFG would normally do.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.CHROMA

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> nn.Module:
        """Instantiate a Chroma transformer and load weights.

        Requires ``diffusers`` to be installed.  Uses the Flux transformer
        architecture with modified configuration for distilled guidance.
        """
        try:
            from diffusers.models import FluxTransformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "ChromaAdapter.create_model requires the 'diffusers' package."
            ) from exc

        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_flux_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        logger.info("Creating Chroma transformer on %s (%s)", device, dtype)

        model_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        model_sd = convert_flux_to_diffusers(model_sd)

        model = FluxTransformer2DModel(**_CHROMA_CONFIG)
        missing, unexpected, mismatched = safe_load_state_dict(model, model_sd)
        log_state_dict_info(missing, unexpected, "Chroma")

        model = place_model(model, device, dtype, ops_context=kwargs.get("ops_context"))
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["t5_xxl"]

    def get_prediction_type(self) -> str:
        return "flow_flux"

    def get_vae_scaling_factor(self) -> float:
        return 0.3611

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def get_prediction_kwargs(self) -> dict[str, Any]:
        """Return kwargs for constructing a FluxPrediction instance.

        Chroma uses a fixed mu of 1.0 (same as Flux Schnell).
        """
        return {"mu": 1.0}

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Prepare Chroma conditioning from text-encoder outputs.

        Chroma uses T5 hidden states as the sole conditioning signal.
        No CLIP pooled output or negative prompts are needed — the distilled
        guidance layer handles guidance internally.
        """
        # Lazy import to avoid circular dependency at module load time
        from serenity.inference.models.flux import compute_img_ids

        t5_hidden = text_outputs.get("t5_xxl", text_outputs.get("encoder_hidden_states"))

        result: dict[str, torch.Tensor] = {}

        if t5_hidden is not None:
            result["encoder_hidden_states"] = t5_hidden
            # Text IDs — zeros like Flux
            txt_ids = torch.zeros(
                t5_hidden.shape[-2],
                3,
                device=t5_hidden.device,
                dtype=t5_hidden.dtype,
            )
            result["txt_ids"] = txt_ids

        # Image IDs computed from latent dimensions when available
        height = kwargs.get("height")
        width = kwargs.get("width")
        if height is not None and width is not None:
            result["img_ids"] = compute_img_ids(
                int(height),
                int(width),
                device=t5_hidden.device if t5_hidden is not None else "cpu",
                dtype=t5_hidden.dtype if t5_hidden is not None else torch.float32,
            )

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.CHROMA: ChromaAdapter}
