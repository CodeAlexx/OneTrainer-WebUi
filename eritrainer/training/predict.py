"""Prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Flux2KleinPredictor:
    """Predictor for Flux2 Klein (no guidance embeddings)."""

    def __call__(
        self,
        transformer,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=True,
        )
        return output.sample


def get_predictor(model_type: str):
    if "klein" in model_type.lower():
        return Flux2KleinPredictor()
    raise ValueError(f"No predictor for model type: {model_type}")
