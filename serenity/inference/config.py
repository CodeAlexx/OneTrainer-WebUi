"""Inference engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["InferenceConfig", "VRAMMode", "QuantizationMode", "AttentionBackend"]


class VRAMMode(str, Enum):
    """VRAM management strategy."""
    AUTO = "auto"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    NO_VRAM = "no_vram"


class QuantizationMode(str, Enum):
    """Model quantization mode."""
    NONE = "none"
    INT8 = "int8"
    FP8 = "fp8"
    BNB_NF4 = "bnb_nf4"
    BNB_FP4 = "bnb_fp4"
    GGUF = "gguf"


class AttentionBackend(str, Enum):
    """Attention computation backend."""
    AUTO = "auto"
    SAGE = "sage"
    FLASH = "flash"
    XFORMERS = "xformers"
    SDP = "sdp"
    EINSUM = "einsum"


@dataclass
class InferenceConfig:
    """Configuration for the inference engine."""

    # Model
    model_path: str = ""
    model_dtype: str = "float16"
    vae_path: str = ""

    # VRAM management
    vram_mode: VRAMMode = VRAMMode.AUTO
    offload_streams: int = 2
    pin_memory: bool = True

    # Attention
    attention_backend: AttentionBackend = AttentionBackend.AUTO

    # Quantization
    quantization: QuantizationMode = QuantizationMode.NONE

    # LoRA
    lora_paths: list[str] = field(default_factory=list)
    lora_weights: list[float] = field(default_factory=list)

    # Sampling defaults
    sampler: str = "euler"
    scheduler: str = "normal"
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512

    # Quality features
    rescale_cfg: float = 0.0
    mahiro: bool = False

    # VAE
    vae_tiling: bool = False
    vae_tile_size: int = 512

    # Cache
    cache_text_encodings: bool = True
    max_cache_memory_mb: int = 1024
