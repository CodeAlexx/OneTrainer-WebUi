"""SD 3.5 (MMDiT) model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "SD3Adapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SD3 transformer configuration
# ---------------------------------------------------------------------------

_SD3_MEDIUM_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
}

_SD3_LARGE_CONFIG = {
    **_SD3_MEDIUM_CONFIG,
    "num_layers": 38,
    "num_attention_heads": 38,
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SD3Adapter(BaseModelAdapter):
    """Model adapter for Stable Diffusion 3 / 3.5 (MMDiT joint transformer)."""

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.SD3

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate an SD3 MMDiT transformer and load weights.

        Requires ``diffusers`` to be installed.  The number of layers is
        auto-detected from the state dict (24 = medium, 38 = large).
        """
        try:
            from diffusers.models import SD3Transformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "SD3Adapter.create_model requires the 'diffusers' package."
            ) from exc

        # Auto-detect depth from state_dict keys
        depth = 0
        while f"transformer_blocks.{depth}.attn.to_q.weight" in state_dict or \
              f"joint_blocks.{depth}.x_block.attn.qkv.weight" in state_dict:
            depth += 1

        if depth == 0:
            depth = int(kwargs.get("num_layers", 24))  # type: ignore[arg-type]

        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_sd3_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        config = _SD3_LARGE_CONFIG if depth >= 38 else _SD3_MEDIUM_CONFIG
        config = {**config, "num_layers": depth}

        logger.info(
            "Creating SD3 MMDiT (%d layers) on %s (%s)", depth, device, dtype,
        )

        model_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        model_sd = convert_sd3_to_diffusers(model_sd)

        model = SD3Transformer2DModel(**config)
        missing, unexpected, mismatched = safe_load_state_dict(model, model_sd)
        if missing:
            logger.warning("SD3: %d missing keys", len(missing))
        if unexpected:
            logger.debug("SD3: %d unexpected keys", len(unexpected))

        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_l", "clip_g", "t5_xxl"]

    def get_prediction_type(self) -> str:
        return "flow"

    def get_vae_scaling_factor(self) -> float:
        return 1.5305

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare SD3 conditioning.

        SD3 needs:
        - ``encoder_hidden_states``: concatenation of CLIP-L, CLIP-G, and T5
          hidden-state sequences along the sequence dimension.
        - ``pooled_projections``: concatenation of CLIP-L and CLIP-G pooled
          outputs along the feature dimension.
        """
        result: dict[str, torch.Tensor] = {}

        # Build concatenated hidden states from all text encoders
        parts: list[torch.Tensor] = []
        for key in ("cond_l", "cond_g", "cond_t5"):
            val = text_outputs.get(key)
            if val is not None:
                parts.append(val)

        if parts:
            result["encoder_hidden_states"] = torch.cat(parts, dim=1)
        elif "cond" in text_outputs:
            result["encoder_hidden_states"] = text_outputs["cond"]

        # Pooled projections from CLIP-L and CLIP-G
        pooled_parts: list[torch.Tensor] = []
        for key in ("pooled_l", "pooled_g"):
            val = text_outputs.get(key)
            if val is not None:
                pooled_parts.append(val)

        if pooled_parts:
            result["pooled_projections"] = torch.cat(pooled_parts, dim=-1)
        elif "pooled" in text_outputs:
            result["pooled_projections"] = text_outputs["pooled"]

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.SD3: SD3Adapter}
