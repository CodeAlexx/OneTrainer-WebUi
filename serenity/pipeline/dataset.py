"""Training dataset with image loading, caption handling, and augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from serenity.core.concept_config import ConceptConfig
from serenity.pipeline.concepts import (
    ImageCaptionPair,
    discover_images,
    load_caption_for_image,
    resolve_prompt,
)


class TrainSample:
    """A single training sample: image path + resolved caption + metadata."""

    __slots__ = ("image_path", "caption", "concept_index", "resolution_override")

    def __init__(
        self,
        image_path: Path,
        caption: str,
        concept_index: int = 0,
        resolution_override: str | None = None,
    ) -> None:
        self.image_path = image_path
        self.caption = caption
        self.concept_index = concept_index
        self.resolution_override = resolution_override


def _default_image_transform(image_path: Path, resolution: int) -> torch.Tensor:
    """Load and transform an image to a tensor.

    Returns a [C, H, W] float32 tensor normalized to [-1, 1].
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert("RGB")
    img = TF.resize(img, resolution, antialias=True)
    img = TF.center_crop(img, (resolution, resolution))
    tensor = TF.to_tensor(img)  # [0, 1]
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    return tensor


def _apply_augmentations(
    tensor: torch.Tensor,
    concept: ConceptConfig,
) -> torch.Tensor:
    """Apply random augmentations based on concept image config."""
    img_cfg = concept.image

    # Random horizontal flip
    if img_cfg.enable_random_flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[-1])
    elif img_cfg.enable_fixed_flip:
        tensor = torch.flip(tensor, dims=[-1])

    return tensor


class SerenityDataset(Dataset):
    """Training dataset that loads images with captions from concept configs.

    Each __getitem__ returns a dict with:
    - pixel_values: [C, H, W] float tensor normalized to [-1, 1]
    - caption: str caption text
    - concept_index: int index of the source concept
    - image_path: str path to the source image
    """

    def __init__(
        self,
        concepts: list[ConceptConfig],
        resolution: int = 512,
        image_transform: Callable[[Path, int], torch.Tensor] | None = None,
        enable_augmentations: bool = True,
    ) -> None:
        self.concepts = concepts
        self.resolution = resolution
        self.image_transform = image_transform or _default_image_transform
        self.enable_augmentations = enable_augmentations
        self._samples: list[TrainSample] = []
        self._build_sample_list()

    def _build_sample_list(self) -> None:
        """Scan all concepts and build the flat sample list with repeats."""
        self._samples.clear()
        for concept_idx, concept in enumerate(self.concepts):
            if not concept.enabled:
                continue

            concept_path = Path(concept.path)
            if not concept_path.exists():
                continue

            images = discover_images(
                concept_path,
                include_subdirectories=concept.include_subdirectories,
            )

            resolution_override = None
            if concept.image.enable_resolution_override:
                resolution_override = concept.image.resolution_override

            for img_path in images:
                caption = resolve_prompt(
                    img_path,
                    concept.text,
                    concept_name=concept.name,
                )
                # Apply repeats (balancing)
                repeat_count = max(1, int(concept.balancing))
                for _ in range(repeat_count):
                    self._samples.append(TrainSample(
                        image_path=img_path,
                        caption=caption,
                        concept_index=concept_idx,
                        resolution_override=resolution_override,
                    ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]

        # Determine resolution
        if sample.resolution_override:
            try:
                res = int(sample.resolution_override.split(",")[0].strip())
            except (ValueError, IndexError):
                res = self.resolution
        else:
            res = self.resolution

        # Load and transform image
        pixel_values = self.image_transform(sample.image_path, res)

        # Apply augmentations
        if self.enable_augmentations and sample.concept_index < len(self.concepts):
            concept = self.concepts[sample.concept_index]
            pixel_values = _apply_augmentations(pixel_values, concept)

        return {
            "pixel_values": pixel_values,
            "caption": sample.caption,
            "concept_index": sample.concept_index,
            "image_path": str(sample.image_path),
        }


# Backward-compat alias
EriDataset = SerenityDataset


def collate_train_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for SerenityDataset batches.

    Stacks pixel_values into a batch tensor, keeps captions as a list.
    """
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "concept_indices": torch.tensor([b["concept_index"] for b in batch]),
        "image_paths": [b["image_path"] for b in batch],
    }


__all__ = [
    "TrainSample",
    "SerenityDataset",
    "EriDataset",
    "collate_train_samples",
]
