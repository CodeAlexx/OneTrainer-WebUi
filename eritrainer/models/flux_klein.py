"""Flux Klein (Flux 1 variant) wrapper."""

from __future__ import annotations

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


class FluxKleinModel(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.FLUX_SCHNELL)
