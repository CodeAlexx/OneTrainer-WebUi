"""Stable Diffusion 1.5 wrapper."""

from __future__ import annotations

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


class SD15Model(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SD15)
