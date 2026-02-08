"""Model type detection and loading for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.models.base import BaseModelAdapter, ModelAdapter
from serenity.inference.models.detection import (
    ModelArchitecture,
    ModelConfig,
    detect_from_file,
    detect_model_type,
)
from serenity.inference.models.loader import load_model, load_state_dict, load_vae

__all__ = [
    "BaseModelAdapter",
    "ModelAdapter",
    "ModelArchitecture",
    "ModelConfig",
    "detect_from_file",
    "detect_model_type",
    "load_model",
    "load_state_dict",
    "load_vae",
]
