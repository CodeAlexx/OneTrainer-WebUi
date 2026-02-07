"""Core interfaces and enums shared across Serenity layers."""

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

    # ------------------------------------------------------------------
    # Helper methods (matching OneTrainer's ModelType helpers)
    # ------------------------------------------------------------------

    def is_stable_diffusion(self) -> bool:
        """True for SD 1.x and SD 2.x variants."""
        return self in (
            ModelType.SD15, ModelType.SD15_INPAINTING,
            ModelType.SD20, ModelType.SD20_BASE,
            ModelType.SD20_INPAINTING, ModelType.SD20_DEPTH,
            ModelType.SD21, ModelType.SD21_BASE,
        )

    def is_sd(self) -> bool:
        """Alias for ``is_stable_diffusion``."""
        return self.is_stable_diffusion()

    def is_sd_v1(self) -> bool:
        """True for SD 1.5 variants."""
        return self in (ModelType.SD15, ModelType.SD15_INPAINTING)

    def is_sd_v2(self) -> bool:
        """True for SD 2.x variants."""
        return self in (
            ModelType.SD20, ModelType.SD20_BASE,
            ModelType.SD20_INPAINTING, ModelType.SD20_DEPTH,
            ModelType.SD21, ModelType.SD21_BASE,
        )

    def is_sdxl(self) -> bool:
        """True for SDXL variants."""
        return self in (
            ModelType.SDXL, ModelType.SDXL_10_BASE, ModelType.SDXL_INPAINTING,
        )

    def is_sd3(self) -> bool:
        """True for SD3 / SD3.5."""
        return self in (ModelType.SD3, ModelType.SD35)

    def is_sd35(self) -> bool:
        """True for SD 3.5 specifically."""
        return self == ModelType.SD35

    def is_flux(self) -> bool:
        """True for any Flux family model."""
        return self in (
            ModelType.FLUX_DEV, ModelType.FLUX_FILL_DEV,
            ModelType.FLUX_SCHNELL,
            ModelType.FLUX_2, ModelType.FLUX_2_DEV,
            ModelType.FLUX_2_KLEIN, ModelType.FLUX_2_KLEIN_4B,
            ModelType.FLUX_2_KLEIN_9B,
            ModelType.FLUX_2_KLEIN_4B_BASE, ModelType.FLUX_2_KLEIN_9B_BASE,
        )

    def is_flux_1(self) -> bool:
        """True for Flux 1.x models."""
        return self in (
            ModelType.FLUX_DEV, ModelType.FLUX_FILL_DEV, ModelType.FLUX_SCHNELL,
        )

    def is_flux_2(self) -> bool:
        """True for Flux 2.x models."""
        return self in (
            ModelType.FLUX_2, ModelType.FLUX_2_DEV,
            ModelType.FLUX_2_KLEIN, ModelType.FLUX_2_KLEIN_4B,
            ModelType.FLUX_2_KLEIN_9B,
            ModelType.FLUX_2_KLEIN_4B_BASE, ModelType.FLUX_2_KLEIN_9B_BASE,
        )

    def is_flux_2_klein(self) -> bool:
        """True for Flux 2 Klein variants."""
        return self in (
            ModelType.FLUX_2_KLEIN,
            ModelType.FLUX_2_KLEIN_4B, ModelType.FLUX_2_KLEIN_9B,
            ModelType.FLUX_2_KLEIN_4B_BASE, ModelType.FLUX_2_KLEIN_9B_BASE,
        )

    def is_pixart(self) -> bool:
        """True for PixArt models."""
        return self in (ModelType.PIXART_ALPHA, ModelType.PIXART_SIGMA)

    def is_wuerstchen(self) -> bool:
        """True for Wuerstchen / Stable Cascade."""
        return self in (ModelType.WUERSTCHEN_2, ModelType.STABLE_CASCADE_1)

    def is_chroma(self) -> bool:
        """True for Chroma models."""
        return self == ModelType.CHROMA_1

    def is_qwen(self) -> bool:
        """True for Qwen models."""
        return self in (ModelType.QWEN, ModelType.QWEN_IMAGE_EDIT)

    def is_sana(self) -> bool:
        """True for Sana models."""
        return self == ModelType.SANA

    def is_hunyuan_video(self) -> bool:
        """True for Hunyuan Video models."""
        return self == ModelType.HUNYUAN_VIDEO

    def is_hi_dream(self) -> bool:
        """True for HiDream models."""
        return self == ModelType.HI_DREAM_FULL

    def is_zimage(self) -> bool:
        """True for Z-Image models."""
        return self in (ModelType.ZIMAGE, ModelType.Z_IMAGE)

    def is_ltx(self) -> bool:
        """True for LTX video models."""
        return self == ModelType.LTX2

    def is_video(self) -> bool:
        """True for video generation models."""
        return self in (ModelType.HUNYUAN_VIDEO, ModelType.LTX2)

    def is_inpainting(self) -> bool:
        """True for inpainting model variants."""
        return self in (
            ModelType.SD15_INPAINTING,
            ModelType.SD20_INPAINTING,
            ModelType.SDXL_INPAINTING,
            ModelType.FLUX_FILL_DEV,
        )

    def has_mask_input(self) -> bool:
        """True if the model natively accepts mask inputs."""
        return self.is_inpainting()

    def has_conditioning_image_input(self) -> bool:
        """True if the model accepts a conditioning image for inpainting."""
        return self.is_inpainting()

    def has_depth_input(self) -> bool:
        """True if the model accepts depth map inputs."""
        return self == ModelType.SD20_DEPTH

    def has_text_encoder_2(self) -> bool:
        """True if the model uses a second text encoder."""
        return (
            self.is_sdxl()
            or self.is_sd3()
            or self.is_flux_1()
            or self.is_hunyuan_video()
            or self.is_hi_dream()
        )

    def is_flow_matching(self) -> bool:
        """True for flow-matching based models (vs. DDPM/noise-prediction)."""
        return (
            self.is_sd3()
            or self.is_flux()
            or self.is_chroma()
            or self.is_qwen()
            or self.is_sana()
            or self.is_hunyuan_video()
            or self.is_hi_dream()
        )


@runtime_checkable
class BaseModel(Protocol):
    """Minimal interface all Serenity models should implement."""

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
