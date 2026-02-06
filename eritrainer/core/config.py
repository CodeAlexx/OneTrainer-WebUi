"""Configuration objects and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

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
        "sd_15": ModelType.SD15,
        "sdxl_base": ModelType.SDXL_10_BASE,
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
    concepts: List[Any]

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

    data: Dict[str, Any]
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
