"""Stable Diffusion 3 / 3.5 model wrappers."""

from __future__ import annotations

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


class SD3Model(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SD3)


class SD35Model(BaseModelImpl):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SD35)
