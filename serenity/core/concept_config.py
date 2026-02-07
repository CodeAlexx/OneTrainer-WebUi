"""Concept configuration for training datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptImageConfig:
    """Image augmentation settings for a concept."""

    # Crop jitter
    enable_crop_jitter: bool = True

    # Flip
    enable_random_flip: bool = False
    enable_fixed_flip: bool = False

    # Rotation
    enable_random_rotate: bool = False
    enable_fixed_rotate: bool = False
    random_rotate_max_angle: float = 0.0

    # Brightness
    enable_random_brightness: bool = False
    enable_fixed_brightness: bool = False
    random_brightness_max_strength: float = 0.0

    # Contrast
    enable_random_contrast: bool = False
    enable_fixed_contrast: bool = False
    random_contrast_max_strength: float = 0.0

    # Saturation
    enable_random_saturation: bool = False
    enable_fixed_saturation: bool = False
    random_saturation_max_strength: float = 0.0

    # Hue
    enable_random_hue: bool = False
    enable_fixed_hue: bool = False
    random_hue_max_strength: float = 0.0

    # Resolution override
    enable_resolution_override: bool = False
    resolution_override: str = "512"

    # Mask augmentations
    enable_random_circular_mask_shrink: bool = False
    enable_random_mask_rotate_crop: bool = False


@dataclass
class ConceptTextConfig:
    """Text/caption processing settings for a concept."""

    prompt_source: str = "sample"
    prompt_path: str = ""
    enable_tag_shuffling: bool = False
    tag_delimiter: str = ","
    keep_tags_count: int = 1

    # Tag dropout
    tag_dropout_enable: bool = False
    tag_dropout_mode: str = "FULL"
    tag_dropout_probability: float = 0.0
    tag_dropout_special_tags_mode: str = "NONE"
    tag_dropout_special_tags: str = ""
    tag_dropout_special_tags_regex: bool = False

    # Capitalization randomization
    caps_randomize_enable: bool = False
    caps_randomize_probability: float = 0.0
    caps_randomize_mode: str = "capslock, title, first, random"
    caps_randomize_lowercase: bool = False


@dataclass
class ConceptConfig:
    """Configuration for a single training concept (dataset subset)."""

    name: str = ""
    path: str = ""
    seed: int = field(default_factory=lambda: random.randint(-(1 << 30), 1 << 30))
    enabled: bool = True
    include_subdirectories: bool = False
    image_variations: int = 1
    text_variations: int = 1
    balancing: float = 1.0
    loss_weight: float = 1.0

    image: ConceptImageConfig = field(default_factory=ConceptImageConfig)
    text: ConceptTextConfig = field(default_factory=ConceptTextConfig)

    def __post_init__(self) -> None:
        if isinstance(self.image, dict):
            self.image = ConceptImageConfig(**self.image)
        if isinstance(self.text, dict):
            self.text = ConceptTextConfig(**self.text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptConfig:
        """Construct a ConceptConfig from a dict, handling nested sub-configs."""
        image_data = data.pop("image", {})
        text_data = data.pop("text", {})
        return cls(
            image=ConceptImageConfig(**image_data) if isinstance(image_data, dict) else image_data,
            text=ConceptTextConfig(**text_data) if isinstance(text_data, dict) else text_data,
            **data,
        )


__all__ = [
    "ConceptImageConfig",
    "ConceptTextConfig",
    "ConceptConfig",
]
