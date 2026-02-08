"""Qwen Image model adapter for the Serenity inference engine.

Status: NOT YET IMPLEMENTED. Detection works, but model loading requires
architecture-specific Transformer2DModel configuration that has not been
implemented yet. Contributions welcome.
"""

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

    .. warning::
        This adapter is **not yet functional**. The ``create_model`` method
        will raise ``NotImplementedError``.  Detection of Qwen checkpoints
        still works via :mod:`serenity.inference.models.detection`.
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
        """Not yet implemented.

        Qwen requires architecture-specific Transformer config (block count,
        hidden dim, head count) inferred from the state dict.  A generic
        ``Transformer2DModel()`` with no config args cannot load real
        checkpoints.
        """
        raise NotImplementedError(
            "Qwen adapter is not yet implemented — Transformer2DModel "
            "requires architecture-specific config (block count, hidden dim, "
            "head count) inferred from the checkpoint. "
            "Contributions welcome!"
        )

    def get_text_encoder_types(self) -> list[str]:
        return ["qwen"]

    def get_prediction_type(self) -> str:
        return "flow"

    def get_vae_scaling_factor(self) -> float:
        return 0.3611

    def get_default_resolution(self) -> tuple[int, int]:
        return (1024, 1024)

    def prepare_conditioning(
        self,
        text_outputs: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Prepare Qwen Image conditioning from Qwen text encoder outputs."""
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
# Registry — empty: Qwen is not yet loadable.
# ---------------------------------------------------------------------------

ADAPTERS: dict = {}
