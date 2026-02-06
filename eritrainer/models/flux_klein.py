"""Flux Schnell adapter."""

from __future__ import annotations

from eritrainer.core.interfaces import ModelType
from eritrainer.models.flux1 import Flux1Model


class FluxKleinModel(Flux1Model):
    """Flux Schnell uses the Flux 1 architecture with a distinct model id."""

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.FLUX_SCHNELL)
