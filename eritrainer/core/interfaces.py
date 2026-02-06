"""Core interfaces and enums shared across EriTrainer layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Protocol, runtime_checkable


class ModelType(str, Enum):
    """Canonical model type identifiers."""

    # Flux family
    FLUX_DEV = "flux_dev"
    FLUX_SCHNELL = "flux_schnell"
    FLUX_2_DEV = "flux_2_dev"
    FLUX_2_KLEIN = "flux_2_klein"
    FLUX_2_KLEIN_4B = "flux_2_klein_4b"
    FLUX_2_KLEIN_9B = "flux_2_klein_9b"
    FLUX_2_KLEIN_4B_BASE = "flux_2_klein_4b_base"
    FLUX_2_KLEIN_9B_BASE = "flux_2_klein_9b_base"

    # Stable Diffusion
    SD15 = "sd15"
    SDXL = "sdxl"
    SDXL_10_BASE = "sdxl_10_base"
    SD3 = "sd3"
    SD35 = "sd35"

    # Z-Image / video
    ZIMAGE = "zimage"
    Z_IMAGE = "z_image"  # alias for compatibility
    LTX2 = "ltx2"

    # Qwen
    QWEN = "qwen"
    QWEN_IMAGE_EDIT = "qwen_image_edit"


@runtime_checkable
class BaseModel(Protocol):
    """Minimal interface all EriTrainer models should implement."""

    model_type: ModelType

    def to(self, device):  # pragma: no cover - simple protocol
        ...

    def train(self):  # pragma: no cover - simple protocol
        ...

    def eval(self):  # pragma: no cover - simple protocol
        ...

    def parameters(self) -> Iterable:  # pragma: no cover - simple protocol
        ...


@dataclass
class ModelInfo:
    """Lightweight metadata container for model instances."""

    model_type: ModelType
    name: str
    variant: Optional[str] = None
