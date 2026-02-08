"""Z-Image model adapter for the Serenity inference engine."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture

__all__ = [
    "ZImageAdapter",
    "ADAPTERS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ZImageAdapter(BaseModelAdapter):
    """Model adapter for Z-Image.

    Z-Image is a DiT-based image generation model structurally similar to
    Lumina but differentiated by its Qwen3 text encoder (presented as
    ``gemma`` in the Serenity text-encoder taxonomy) and distinct weight
    layout (``cap_pad_token`` key, higher layer count / hidden dim).
    Uses discrete flow-matching prediction.
    """

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.ZIMAGE

    def create_model(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate a Z-Image transformer and load weights.

        Requires ``diffusers`` to be installed.  Infers layer count from
        the state dict for debugging.
        """
        try:
            from diffusers.models import Transformer2DModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotImplementedError(
                "ZImageAdapter.create_model requires the 'diffusers' package."
            ) from exc

        # Infer config from state dict key patterns
        num_layers = 0
        hidden_dim: int | None = None
        for key in state_dict:
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_layers = max(num_layers, int(parts[1]) + 1)
            if key == "cap_embedder.1.weight" and hidden_dim is None:
                hidden_dim = state_dict[key].shape[0]

        self._inferred_config = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        }
        logger.info(
            "Creating Z-Image transformer on %s (%s) — inferred %d layers, "
            "hidden_dim=%s",
            device,
            dtype,
            num_layers,
            hidden_dim,
        )

        try:
            model = Transformer2DModel()
        except Exception:
            raise NotImplementedError(
                "ZImageAdapter.create_model requires a compatible diffusers "
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
        """Prepare Z-Image conditioning from text encoder outputs.

        Z-Image uses text hidden states as cross-attention context,
        similar to Lumina but with a distinct cap_embedder path.
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

ADAPTERS = {ModelArchitecture.ZIMAGE: ZImageAdapter}
