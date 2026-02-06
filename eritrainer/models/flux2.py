"""Flux 2 model wrapper (minimal)."""

from __future__ import annotations

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


class Flux2Model(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.FLUX_2_DEV)
