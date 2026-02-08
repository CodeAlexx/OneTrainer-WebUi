"""Lumina 2 model adapter for the Serenity inference engine."""

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
        """Instantiate a Lumina 2 transformer and load weights.

        Requires ``diffusers`` to be installed.  Infers layer count and
        hidden dim from the state dict for debugging.
        """
        try:
            from diffusers.models import Transformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "LuminaAdapter.create_model requires the 'diffusers' package."
            ) from exc

        # Infer config from state dict key patterns
        num_layers = 0
        hidden_dim: int | None = None
        for key in state_dict:
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_layers = max(num_layers, int(parts[1]) + 1)
            # Infer hidden dim from cap_embedder output weight shape
            if key == "cap_embedder.1.weight" and hidden_dim is None:
                hidden_dim = state_dict[key].shape[0]

        self._inferred_config = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        }
        logger.info(
            "Creating Lumina 2 transformer on %s (%s) — inferred %d layers, "
            "hidden_dim=%s",
            device,
            dtype,
            num_layers,
            hidden_dim,
        )

        # Lumina uses a custom transformer architecture.  A future diffusers
        # release may ship a dedicated class; for now we use the generic model.
        try:
            model = Transformer2DModel()
        except Exception:
            raise NotImplementedError(
                "LuminaAdapter.create_model requires a compatible diffusers "
                "Transformer2DModel."
            )

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=torch.device(device), dtype=dtype)
        model.eval()
        return model

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
        """Prepare Lumina 2 conditioning from Gemma text encoder outputs.

        Lumina uses Gemma hidden states as cross-attention context and
        optionally applies a cap_embedder for additional conditioning.
        """
        text_out = text_outputs.get("cond")
        if text_out is None:
            text_out = text_outputs.get("encoder_hidden_states")
        if text_out is None:
            return text_outputs

        return {"encoder_hidden_states": text_out}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {ModelArchitecture.LUMINA: LuminaAdapter}
