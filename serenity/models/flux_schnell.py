"""Flux Schnell model adapter.

Flux Schnell is a Flux 1 variant and is separate from FLUX.2 Klein models.
"""

from __future__ import annotations

from serenity.core.interfaces import ModelType
from serenity.models.flux1 import Flux1Model


class FluxSchnellModel(Flux1Model):
    """Flux Schnell uses the Flux 1 architecture with a distinct model id."""

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.FLUX_SCHNELL)
