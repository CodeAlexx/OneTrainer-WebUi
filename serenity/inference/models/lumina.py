"""Lumina 2 model adapter for the Serenity inference engine.

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
    "LuminaAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LuminaAdapter(BaseModelAdapter):
    """Model adapter for Lumina 2.

    Lumina is a DiT-based image generation model using a Gemma text encoder
    with discrete flow-matching prediction and cap_embedder conditioning.

    .. warning::
        This adapter is **not yet functional**. The ``create_model`` method
        will raise ``NotImplementedError``.  Detection of Lumina checkpoints
        still works via :mod:`serenity.inference.models.detection`.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.LUMINA

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Not yet implemented.

        Lumina requires architecture-specific Transformer config (layer count,
        hidden dim, head count) inferred from the state dict.  A generic
        ``Transformer2DModel()`` with no config args cannot load real
        checkpoints.
        """
        raise NotImplementedError(
            "Lumina adapter is not yet implemented — Transformer2DModel "
            "requires architecture-specific config (layer count, hidden dim, "
            "head count) inferred from the checkpoint. "
            "Contributions welcome!"
        )

    def get_text_encoder_types(self) -> list[str]:
        return ["gemma"]

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
        """Prepare Lumina 2 conditioning from Gemma text encoder outputs."""
        text_out = text_outputs.get("cond")
        if text_out is None:
            text_out = text_outputs.get("encoder_hidden_states")
        if text_out is None:
            return text_outputs

        return {"encoder_hidden_states": text_out}


# ---------------------------------------------------------------------------
# Registry — empty: Lumina is not yet loadable.
# ---------------------------------------------------------------------------

ADAPTERS: dict = {}
