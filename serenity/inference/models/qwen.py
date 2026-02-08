"""Qwen Image model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "QwenAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QwenAdapter(BaseModelAdapter):
    """Model adapter for Qwen Image.

    Qwen Image is a DiT-based image generation model using a Qwen 2.5
    text encoder with discrete flow-matching prediction.  It supports
    optional vision-language conditioning through reference images.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.QWEN

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate a Qwen Image transformer and load weights.

        Requires ``diffusers`` to be installed.
        """
        try:
            from diffusers.models import Transformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "QwenAdapter.create_model requires the 'diffusers' package."
            ) from exc

        logger.info("Creating Qwen Image transformer on %s (%s)", device, dtype)
        try:
            model = Transformer2DModel()
        except Exception:
            raise NotImplementedError(
                "QwenAdapter.create_model requires a compatible diffusers "
                "Transformer2DModel."
            )

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

    def get_text_encoder_types(self) -> list[str]:
        return ["qwen"]

    def get_prediction_type(self) -> str:
        return "flow"

    def get_vae_scaling_factor(self) -> float:
        return 0.18215

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare Qwen Image conditioning from Qwen text encoder outputs.

        Qwen uses its own text hidden-state format as cross-attention
        context.  Reference image latents can be injected via *kwargs*
        for vision-language conditioning.
        """
        qwen_out = text_outputs.get("cond")
        if qwen_out is None:
            qwen_out = text_outputs.get("encoder_hidden_states")
        if qwen_out is None:
            return text_outputs

        result: dict[str, torch.Tensor] = {"encoder_hidden_states": qwen_out}

        # Optional reference image latents for VL conditioning
        ref_latents = kwargs.get("ref_latents")
        if ref_latents is not None and isinstance(ref_latents, torch.Tensor):
            result["ref_latents"] = ref_latents

        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.QWEN: QwenAdapter}
