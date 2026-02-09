"""Flux Dev/Schnell model adapters for the Serenity inference engine."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from serenity.inference.models.base import BaseModelAdapter, log_state_dict_info, place_model
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "FluxAdapter",
    "FluxKlein4BAdapter",
    "FluxKlein9BAdapter",
    "FluxSchnellAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flux transformer configuration
# ---------------------------------------------------------------------------

_FLUX_CONFIG = {
    "in_channels": 64,
    "num_layers": 19,  # double_blocks
    "num_single_layers": 38,  # single_blocks
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}

_FLUX_SCHNELL_CONFIG = {
    **_FLUX_CONFIG,
    "guidance_embeds": False,
}

_FLUX_KLEIN_4B_CONFIG = {
    "in_channels": 64,
    "num_layers": 12,
    "num_single_layers": 24,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}

# ---------------------------------------------------------------------------
# Positional encoding helpers
# ---------------------------------------------------------------------------


def compute_img_ids(
    height: int,
    width: int,
    patch_size: int = 2,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create positional ID tensor for image patches.

    Returns a tensor of shape ``(h_patches * w_patches, 3)`` where each row
    contains ``(batch_idx, y, x)`` coordinates for a single patch.
    """
    h_patches = height // patch_size
    w_patches = width // patch_size

    img_ids = torch.zeros(h_patches, w_patches, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h_patches, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device, dtype=dtype)[None, :]

    return img_ids.reshape(-1, 3)


def _compute_txt_ids(
    seq_len: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create positional ID tensor for text tokens.

    Returns a zero tensor of shape ``(seq_len, 3)`` — Flux uses zero IDs
    for text positions.
    """
    return torch.zeros(seq_len, 3, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Flux Dev adapter
# ---------------------------------------------------------------------------


class FluxAdapter(BaseModelAdapter):
    """Model adapter for Flux Dev and its variants.

    Flux is a DiT (Diffusion Transformer) with double-stream architecture that
    uses both CLIP-L and T5-XXL text encoders with flow matching prediction
    and mu-based sigma shifting.

    Args:
        variant: Either ``"dev"`` or ``"schnell"``.
    """

    def __init__(self, variant: str = "dev") -> None:
        self._variant = variant

    @property
    def architecture(self) -> ModelArchitecture:
        if self._variant == "schnell":
            return ModelArchitecture.FLUX_SCHNELL
        return ModelArchitecture.FLUX_DEV

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> nn.Module:
        """Instantiate a Flux transformer and load weights.

        Requires ``diffusers`` to be installed.
        """
        try:
            from diffusers.models import FluxTransformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "FluxAdapter.create_model requires the 'diffusers' package."
            ) from exc

        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_flux_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        config = _FLUX_SCHNELL_CONFIG if self._variant == "schnell" else _FLUX_CONFIG
        logger.info(
            "Creating Flux %s transformer on %s (%s)",
            self._variant,
            device,
            dtype,
        )

        model_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        model_sd = convert_flux_to_diffusers(model_sd)

        model = FluxTransformer2DModel(**config)
        missing, unexpected, mismatched = safe_load_state_dict(model, model_sd)
        log_state_dict_info(missing, unexpected, f"Flux {self._variant}")

        ops_context = kwargs.get("ops_context")
        model = place_model(model, device, dtype, ops_context=ops_context)
        if ops_context is not None:
            logger.info("Flux %s: offload active, keeping model on CPU for managed placement", self._variant)
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["clip_l", "t5_xxl"]

    def get_prediction_type(self) -> str:
        return "flow_flux"

    def get_vae_scaling_factor(self) -> float:
        return 0.3611

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def get_prediction_kwargs(self) -> dict[str, Any]:
        """Return kwargs for constructing a FluxPrediction instance.

        Flux dev uses resolution-dependent mu computed from sequence length.
        Flux schnell uses a fixed mu of 1.0.
        """
        if self._variant == "schnell":
            return {"mu": 1.0}
        return {
            "seq_len": 4096,
            "base_seq_len": 256,
            "max_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

    @property
    def supports_kontext(self) -> bool:
        """Whether this adapter supports Kontext image conditioning."""
        return True

    @property
    def distilled_cfg_scale(self) -> float:
        """CFG scale modifier for distilled models. 0.0 = disabled."""
        return 0.0

    def prepare_kontext_conditioning(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
    ) -> Tensor:
        """Prepare Kontext-style conditioning by concatenating image and text embeddings.

        Args:
            image_embeds: Image encoder output (B, N_img, D).
            text_embeds: Text encoder output (B, N_txt, D).

        Returns:
            Combined conditioning (B, N_img + N_txt, D).
        """
        # Ensure matching dimensions
        if image_embeds.shape[-1] != text_embeds.shape[-1]:
            # Project image embeds to match text dimension
            logger.warning(
                "Kontext dim mismatch: image=%d, text=%d — truncating/padding",
                image_embeds.shape[-1],
                text_embeds.shape[-1],
            )
            d = text_embeds.shape[-1]
            if image_embeds.shape[-1] > d:
                image_embeds = image_embeds[..., :d]
            else:
                pad = torch.zeros(
                    *image_embeds.shape[:-1],
                    d - image_embeds.shape[-1],
                    device=image_embeds.device,
                    dtype=image_embeds.dtype,
                )
                image_embeds = torch.cat([image_embeds, pad], dim=-1)

        return torch.cat([image_embeds, text_embeds], dim=1)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Prepare Flux conditioning from text-encoder outputs.

        Flux requires T5 hidden states as the main conditioning, CLIP-L pooled
        output as vector conditioning, and positional ID tensors for both image
        patches and text tokens.
        """
        t5_hidden = text_outputs.get("t5_xxl", text_outputs.get("encoder_hidden_states"))
        clip_pooled = text_outputs.get("clip_l_pooled", text_outputs.get("pooled_projections"))

        result: dict[str, torch.Tensor] = {}

        if t5_hidden is not None:
            result["encoder_hidden_states"] = t5_hidden
            txt_ids = _compute_txt_ids(
                t5_hidden.shape[-2],
                device=t5_hidden.device,
                dtype=t5_hidden.dtype,
            )
            result["txt_ids"] = txt_ids

        if clip_pooled is not None:
            result["pooled_projections"] = clip_pooled

        # Image IDs are typically computed at sampling time from the actual
        # latent dimensions.  If height/width are provided, we compute them.
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
# Flux Schnell convenience subclass
# ---------------------------------------------------------------------------


class FluxSchnellAdapter(FluxAdapter):
    """Model adapter for Flux Schnell — fewer inference steps, no CFG needed."""

    def __init__(self) -> None:
        super().__init__(variant="schnell")


# ---------------------------------------------------------------------------
# Flux 2 Klein adapters
# ---------------------------------------------------------------------------


class FluxKlein4BAdapter(FluxAdapter):
    """Model adapter for Flux 2 Klein 4B — compact Flux 2 variant.

    Klein 4B uses 12 double blocks and 24 single blocks, roughly half the
    parameters of the full Flux architecture.
    """

    def __init__(self) -> None:
        super().__init__(variant="klein_4b")

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.FLUX_2_KLEIN_4B

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> nn.Module:
        """Instantiate a Flux 2 Klein 4B transformer and load weights."""
        try:
            from diffusers.models import FluxTransformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "FluxKlein4BAdapter.create_model requires the 'diffusers' package."
            ) from exc

        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_flux_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        logger.info(
            "Creating Flux 2 Klein 4B transformer on %s (%s)",
            device,
            dtype,
        )

        model_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        model_sd = convert_flux_to_diffusers(model_sd)

        model = FluxTransformer2DModel(**_FLUX_KLEIN_4B_CONFIG)
        missing, unexpected, mismatched = safe_load_state_dict(model, model_sd)
        log_state_dict_info(missing, unexpected, "Flux Klein 4B")

        ops_context = kwargs.get("ops_context")
        model = place_model(model, device, dtype, ops_context=ops_context)
        if ops_context is not None:
            logger.info("Flux Klein 4B: offload active, keeping model on CPU for managed placement")
        return model


class FluxKlein9BAdapter(FluxAdapter):
    """Model adapter for Flux 2 Klein 9B — same depth as full Flux but Flux 2 arch.

    Klein 9B uses 19 double blocks and 38 single blocks, same as the standard
    Flux architecture but with Flux 2 double-stream modulation.
    """

    def __init__(self) -> None:
        super().__init__(variant="klein_9b")

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.FLUX_2_KLEIN_9B

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> nn.Module:
        """Instantiate a Flux 2 Klein 9B transformer and load weights."""
        try:
            from diffusers.models import FluxTransformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "FluxKlein9BAdapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info(
            "Creating Flux 2 Klein 9B transformer on %s (%s)",
            device,
            dtype,
        )
        from serenity.inference.models.convert import (
            _UNET_PREFIXES,
            convert_flux_to_diffusers,
            extract_submodel,
            safe_load_state_dict,
        )

        model_sd = extract_submodel(state_dict, _UNET_PREFIXES)
        model_sd = convert_flux_to_diffusers(model_sd)

        # Klein 9B uses the same config as standard Flux (19/38 blocks)
        model = FluxTransformer2DModel(**_FLUX_CONFIG)
        missing, unexpected, mismatched = safe_load_state_dict(model, model_sd)
        log_state_dict_info(missing, unexpected, "Flux Klein 9B")

        ops_context = kwargs.get("ops_context")
        model = place_model(model, device, dtype, ops_context=ops_context)
        if ops_context is not None:
            logger.info("Flux Klein 9B: offload active, keeping model on CPU for managed placement")
        return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {
    ModelArchitecture.FLUX_DEV: FluxAdapter,
    ModelArchitecture.FLUX_SCHNELL: FluxSchnellAdapter,
    ModelArchitecture.FLUX_2_KLEIN_4B: FluxKlein4BAdapter,
    ModelArchitecture.FLUX_2_KLEIN_9B: FluxKlein9BAdapter,
}
