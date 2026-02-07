"""Core interfaces and enums shared across EriTrainer layers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ModelType(str, Enum):
    """Canonical model type identifiers."""

    # Flux family
    FLUX_DEV = "flux_dev"
    FLUX_FILL_DEV = "flux_fill_dev"
    FLUX_SCHNELL = "flux_schnell"
    FLUX_2 = "flux_2"
    FLUX_2_DEV = "flux_2_dev"
    FLUX_2_KLEIN = "flux_2_klein"
    FLUX_2_KLEIN_4B = "flux_2_klein_4b"
    FLUX_2_KLEIN_9B = "flux_2_klein_9b"
    FLUX_2_KLEIN_4B_BASE = "flux_2_klein_4b_base"
    FLUX_2_KLEIN_9B_BASE = "flux_2_klein_9b_base"

    # Stable Diffusion
    SD15 = "sd15"
    SD15_INPAINTING = "sd15_inpainting"
    SD20 = "sd20"
    SD20_BASE = "sd20_base"
    SD20_INPAINTING = "sd20_inpainting"
    SD20_DEPTH = "sd20_depth"
    SD21 = "sd21"
    SD21_BASE = "sd21_base"
    SDXL = "sdxl"
    SDXL_10_BASE = "sdxl_10_base"
    SDXL_INPAINTING = "sdxl_inpainting"
    SD3 = "sd3"
    SD35 = "sd35"

    # Wuerstchen family
    WUERSTCHEN_2 = "wuerstchen_2"
    STABLE_CASCADE_1 = "stable_cascade_1"

    # Other diffusion families
    PIXART_ALPHA = "pixart_alpha"
    PIXART_SIGMA = "pixart_sigma"
    SANA = "sana"
    HUNYUAN_VIDEO = "hunyuan_video"
    HI_DREAM_FULL = "hi_dream_full"
    CHROMA_1 = "chroma_1"

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
    variant: str | None = None
