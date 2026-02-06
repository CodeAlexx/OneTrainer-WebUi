"""Configuration objects and helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from eritrainer.core.interfaces import ModelType


class TrainingMethod(str, Enum):
    LORA = "lora"
    FINE_TUNE = "fine_tune"
    EMBEDDING = "embedding"


def _coerce_model_type(value: ModelType | str) -> ModelType:
    if isinstance(value, ModelType):
        return value

    normalized = str(value).lower()
    # Allow common aliases
    alias_map = {
        "z_image": ModelType.ZIMAGE,
        "zimage": ModelType.ZIMAGE,
        "z-image": ModelType.ZIMAGE,
        "sd_15": ModelType.SD15,
        "sd15_inpaint": ModelType.SD15_INPAINTING,
        "sd_15_inpainting": ModelType.SD15_INPAINTING,
        "sd15_inpainting": ModelType.SD15_INPAINTING,
        "sd_20": ModelType.SD20,
        "sd2": ModelType.SD20,
        "sd2_0": ModelType.SD20,
        "sd_20_base": ModelType.SD20_BASE,
        "sd20_base": ModelType.SD20_BASE,
        "sd_20_inpainting": ModelType.SD20_INPAINTING,
        "sd20_inpainting": ModelType.SD20_INPAINTING,
        "sd_20_depth": ModelType.SD20_DEPTH,
        "sd20_depth": ModelType.SD20_DEPTH,
        "sd_21": ModelType.SD21,
        "sd21": ModelType.SD21,
        "sd_21_base": ModelType.SD21_BASE,
        "sd21_base": ModelType.SD21_BASE,
        "sdxl_base": ModelType.SDXL_10_BASE,
        "sdxl_10_base_inpainting": ModelType.SDXL_INPAINTING,
        "sdxl_inpainting": ModelType.SDXL_INPAINTING,
        "sdxl_inpaint": ModelType.SDXL_INPAINTING,
        "sd3": ModelType.SD3,
        "sd_3": ModelType.SD3,
        "sd35": ModelType.SD35,
        "sd_35": ModelType.SD35,
        "sd3.5": ModelType.SD35,
        "stable_diffusion_3": ModelType.SD3,
        "stable_diffusion_35": ModelType.SD35,
        "stable_diffusion_3.5": ModelType.SD35,
        "flux2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
        "flux2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
        "flux_2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
        "flux_2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
        "flux_fill": ModelType.FLUX_FILL_DEV,
        "flux_fill_dev": ModelType.FLUX_FILL_DEV,
        "flux_fill_dev_1": ModelType.FLUX_FILL_DEV,
        "flux_2": ModelType.FLUX_2,
        "flux2": ModelType.FLUX_2,
        "wuerstchen": ModelType.WUERSTCHEN_2,
        "wuerstchen_2": ModelType.WUERSTCHEN_2,
        "stable_cascade": ModelType.STABLE_CASCADE_1,
        "stable_cascade_1": ModelType.STABLE_CASCADE_1,
        "pixart": ModelType.PIXART_ALPHA,
        "pixart_alpha": ModelType.PIXART_ALPHA,
        "pixart_sigma": ModelType.PIXART_SIGMA,
        "sana": ModelType.SANA,
        "hunyuan_video": ModelType.HUNYUAN_VIDEO,
        "hidream": ModelType.HI_DREAM_FULL,
        "hi_dream_full": ModelType.HI_DREAM_FULL,
        "chroma": ModelType.CHROMA_1,
        "chroma_1": ModelType.CHROMA_1,
        "ltx2": ModelType.LTX2,
        "ltx": ModelType.LTX2,
        "ltx_video": ModelType.LTX2,
        "ltxvideo": ModelType.LTX2,
    }
    if normalized in alias_map:
        return alias_map[normalized]

    try:
        return ModelType(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown model type: {value}") from exc


def _coerce_training_method(value: TrainingMethod | str) -> TrainingMethod:
    if isinstance(value, TrainingMethod):
        return value

    normalized = str(value).lower()
    try:
        return TrainingMethod(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown training method: {value}") from exc


@dataclass
class TrainConfig:
    """Training configuration used by the pipeline and trainer."""

    model_type: ModelType | str
    training_method: TrainingMethod | str
    transformer_path: str
    output_dir: str
    concepts: list[Any]

    # Common defaults
    learning_rate: float = 1e-4
    train_device: str = "cuda"
    temp_device: str = "cpu"
    seed: int = 42

    # Memory & offload
    layer_offload_fraction: float = 0.0
    gradient_checkpointing: str = "off"
    enable_activation_offloading: bool = False
    enable_async_offloading: bool = False

    # Other placeholders
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.model_type = _coerce_model_type(self.model_type)
        self.training_method = _coerce_training_method(self.training_method)
        self.output_dir = str(self.output_dir)


TrainerConfig = TrainConfig


@dataclass
class NoiseConfig:
    """Noise configuration options for SDXL-style offset noise."""

    offset_noise_weight: float = 0.0
    generalized_offset_noise: bool = False


def load_config(path: str | Path) -> TrainConfig:
    """Load a config file (JSON/YAML) into a TrainConfig."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data: dict[str, Any]
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text())
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML required to load YAML config") from exc
        data = yaml.safe_load(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return TrainConfig(**data)


__all__ = [
    "TrainingMethod",
    "TrainConfig",
    "TrainerConfig",
    "ModelType",
    "NoiseConfig",
    "load_config",
]
