"""SDXL and SDXL Refiner model adapters for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "SDXLAdapter",
    "SDXLRefinerAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDXL UNet configuration
# ---------------------------------------------------------------------------

_SDXL_UNET_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "cross_attention_dim": 2048,
    "block_out_channels": (320, 640, 1280),
    "layers_per_block": 2,
    "transformer_layers_per_block": [1, 2, 10],
    "attention_head_dim": [5, 10, 20],
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "addition_embed_type_num_heads": 64,
    "projection_class_embeddings_input_dim": 2816,
    "down_block_types": (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ),
}

_SDXL_REFINER_UNET_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "cross_attention_dim": 1280,
    "block_out_channels": (384, 768, 1536, 1536),
    "layers_per_block": 2,
    "transformer_layers_per_block": [1, 4, 4, 4],
    "attention_head_dim": [5, 10, 20, 20],
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "addition_embed_type_num_heads": 64,
    "projection_class_embeddings_input_dim": 2560,
    "down_block_types": (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_add_time_ids(
    original_size: tuple[int, int] = (1024, 1024),
    crop_coords: tuple[int, int] = (0, 0),
    target_size: tuple[int, int] = (1024, 1024),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the ``add_time_ids`` tensor for SDXL conditioning.

    Layout: ``[original_h, original_w, crop_top, crop_left, target_h, target_w]``
    """
    return torch.tensor(
        [
            original_size[0],
            original_size[1],
            crop_coords[0],
            crop_coords[1],
            target_size[0],
            target_size[1],
        ],
        dtype=dtype,
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# SDXL Adapter
# ---------------------------------------------------------------------------


class SDXLAdapter(BaseModelAdapter):
    """Model adapter for Stable Diffusion XL (base)."""

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.SDXL

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate an SDXL UNet and load weights.

        Requires ``diffusers`` to be installed.
        """
        try:
            from diffusers.models import UNet2DConditionModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "SDXLAdapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info("Creating SDXL UNet on %s (%s)", device, dtype)
        model = UNet2DConditionModel(**_SDXL_UNET_CONFIG)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_l", "clip_g"]

    def get_prediction_type(self) -> str:
        return "eps"

    def get_vae_scaling_factor(self) -> float:
        return 0.13025

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare SDXL conditioning.

        SDXL requires:
        - ``encoder_hidden_states``: concatenation of CLIP-L and CLIP-G
          hidden states along the feature dimension.
        - ``added_cond_kwargs.text_embeds``: CLIP-G pooled output.
        - ``added_cond_kwargs.time_ids``: original/crop/target size encoding.
        """
        result: dict[str, torch.Tensor] = {}

        # Concatenate CLIP-L and CLIP-G hidden states
        cond_l = text_outputs.get("cond_l")
        cond_g = text_outputs.get("cond_g")

        if cond_l is not None and cond_g is not None:
            # Pad to same sequence length if needed
            max_len = max(cond_l.shape[1], cond_g.shape[1])
            if cond_l.shape[1] < max_len:
                pad = cond_l.new_zeros(cond_l.shape[0], max_len - cond_l.shape[1], cond_l.shape[2])
                cond_l = torch.cat([cond_l, pad], dim=1)
            if cond_g.shape[1] < max_len:
                pad = cond_g.new_zeros(cond_g.shape[0], max_len - cond_g.shape[1], cond_g.shape[2])
                cond_g = torch.cat([cond_g, pad], dim=1)
            result["encoder_hidden_states"] = torch.cat([cond_l, cond_g], dim=2)
        elif "cond" in text_outputs:
            result["encoder_hidden_states"] = text_outputs["cond"]

        # Pooled output from CLIP-G
        pooled = text_outputs.get("pooled")
        if pooled is not None:
            result["pooled"] = pooled

        # Time IDs for size conditioning
        original_size = kwargs.get("original_size", (1024, 1024))
        crop_coords = kwargs.get("crop_coords", (0, 0))
        target_size = kwargs.get("target_size", (1024, 1024))
        _dtype = pooled.dtype if pooled is not None else torch.float32

        result["add_time_ids"] = _make_add_time_ids(
            original_size=original_size,  # type: ignore[arg-type]
            crop_coords=crop_coords,  # type: ignore[arg-type]
            target_size=target_size,  # type: ignore[arg-type]
            dtype=_dtype,
        )

        return result


# ---------------------------------------------------------------------------
# SDXL Refiner Adapter
# ---------------------------------------------------------------------------


class SDXLRefinerAdapter(BaseModelAdapter):
    """Model adapter for Stable Diffusion XL Refiner.

    The refiner uses only CLIP-G (no CLIP-L) and has a different UNet
    configuration with ``model_channels=384`` and ``context_dim=1280``.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.SDXL_REFINER

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate an SDXL Refiner UNet and load weights."""
        try:
            from diffusers.models import UNet2DConditionModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "SDXLRefinerAdapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info("Creating SDXL Refiner UNet on %s (%s)", device, dtype)
        model = UNet2DConditionModel(**_SDXL_REFINER_UNET_CONFIG)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_g"]

    def get_prediction_type(self) -> str:
        return "eps"

    def get_vae_scaling_factor(self) -> float:
        return 0.13025

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare SDXL Refiner conditioning.

        The refiner uses CLIP-G only.  Conditioning includes the CLIP-G
        hidden states, pooled output, and aesthetic-score time IDs.
        """
        result: dict[str, torch.Tensor] = {}

        cond_g = text_outputs.get("cond_g")
        if cond_g is not None:
            result["encoder_hidden_states"] = cond_g
        elif "cond" in text_outputs:
            result["encoder_hidden_states"] = text_outputs["cond"]

        pooled = text_outputs.get("pooled")
        if pooled is not None:
            result["pooled"] = pooled

        # Refiner time IDs: [original_h, original_w, crop_top, crop_left, aesthetic_score]
        original_size = kwargs.get("original_size", (1024, 1024))
        crop_coords = kwargs.get("crop_coords", (0, 0))
        aesthetic_score = kwargs.get("aesthetic_score", 6.0)
        _dtype = pooled.dtype if pooled is not None else torch.float32

        result["add_time_ids"] = torch.tensor(
            [
                original_size[0],  # type: ignore[index]
                original_size[1],  # type: ignore[index]
                crop_coords[0],  # type: ignore[index]
                crop_coords[1],  # type: ignore[index]
                aesthetic_score,  # type: ignore[arg-type]
            ],
            dtype=_dtype,
        ).unsqueeze(0)

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {
    ModelArchitecture.SDXL: SDXLAdapter,
    ModelArchitecture.SDXL_REFINER: SDXLRefinerAdapter,
}
