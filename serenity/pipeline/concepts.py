"""Concept scanning and caption utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from serenity.pipeline.concept import Concept
from serenity.core.concept_config import ConceptConfig, ConceptTextConfig


# Supported image extensions
IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
})

# Caption file extension
CAPTION_EXTENSION = ".txt"


class ConceptScanner:
    """Scan directories for training concepts and their images."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def scan(self) -> list[Concept]:
        """Discover concepts from subdirectories under root."""
        if not self.root.exists():
            return []
        concepts: list[Concept] = []
        for child in self.root.iterdir():
            if child.is_dir():
                caption_file = child / "captions.txt"
                concepts.append(Concept(name=child.name, path=child, caption_file=caption_file))
        return concepts


class ImageCaptionPair:
    """An image file paired with its caption text."""

    __slots__ = ("image_path", "caption")

    def __init__(self, image_path: Path, caption: str = "") -> None:
        self.image_path = image_path
        self.caption = caption

    def __repr__(self) -> str:
        return f"ImageCaptionPair({self.image_path.name!r}, caption_len={len(self.caption)})"


class CaptionLoader:
    """Load captions from files, with support for per-image and directory-level captions."""

    def __init__(self, caption_file: Path) -> None:
        self.caption_file = Path(caption_file)

    def load(self) -> list[str]:
        """Load lines from a directory-level caption file."""
        if not self.caption_file.exists():
            return []
        return [line.strip() for line in self.caption_file.read_text().splitlines() if line.strip()]


def load_caption_for_image(image_path: Path) -> str:
    """Load the caption text for a single image.

    Looks for a .txt file with the same stem next to the image.
    Returns empty string if no caption file exists.
    """
    caption_path = image_path.with_suffix(CAPTION_EXTENSION)
    if caption_path.exists():
        return caption_path.read_text().strip()
    return ""


def discover_images(
    directory: Path,
    include_subdirectories: bool = False,
) -> list[Path]:
    """Find all image files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if include_subdirectories:
        results = []
        for ext in IMAGE_EXTENSIONS:
            results.extend(directory.rglob(f"*{ext}"))
        return sorted(results)
    else:
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )


def load_image_caption_pairs(
    directory: Path,
    include_subdirectories: bool = False,
    default_caption: str = "",
) -> list[ImageCaptionPair]:
    """Load all images from a directory, each paired with its caption.

    Caption priority:
    1. Per-image .txt file (same name as image, .txt extension)
    2. default_caption parameter (e.g., from concept config)
    """
    images = discover_images(directory, include_subdirectories)
    pairs = []
    for img_path in images:
        caption = load_caption_for_image(img_path)
        if not caption:
            caption = default_caption
        pairs.append(ImageCaptionPair(image_path=img_path, caption=caption))
    return pairs


def resolve_prompt(
    image_path: Path,
    text_config: ConceptTextConfig,
    concept_name: str = "",
) -> str:
    """Resolve the prompt/caption for an image based on text config settings.

    prompt_source modes:
    - "sample": Use the per-image .txt file, fallback to concept name
    - "concept": Use the concept name as the caption
    - "directory": Use the directory-level captions.txt
    - "file": Use the text_config.prompt_path file contents
    """
    source = text_config.prompt_source.lower().strip()

    if source == "concept":
        return concept_name

    if source == "file" and text_config.prompt_path:
        prompt_file = Path(text_config.prompt_path)
        if prompt_file.exists():
            return prompt_file.read_text().strip()
        return concept_name

    if source == "directory":
        caption_file = image_path.parent / "captions.txt"
        if caption_file.exists():
            lines = [l.strip() for l in caption_file.read_text().splitlines() if l.strip()]
            if lines:
                return lines[0]
        return concept_name

    # Default: "sample" - per-image caption file
    caption = load_caption_for_image(image_path)
    if caption:
        return _apply_tag_processing(caption, text_config)
    return concept_name


def _apply_tag_processing(caption: str, text_config: ConceptTextConfig) -> str:
    """Apply tag shuffling and dropout to a caption."""
    if not caption:
        return caption

    delimiter = text_config.tag_delimiter
    tags = [t.strip() for t in caption.split(delimiter) if t.strip()]

    if not tags:
        return caption

    # Tag dropout
    if text_config.tag_dropout_enable and text_config.tag_dropout_probability > 0:
        keep_count = text_config.keep_tags_count
        protected = tags[:keep_count]
        droppable = tags[keep_count:]

        if text_config.tag_dropout_mode.upper() == "FULL":
            droppable = [
                t for t in droppable
                if random.random() > text_config.tag_dropout_probability
            ]

        tags = protected + droppable

    # Tag shuffling
    if text_config.enable_tag_shuffling:
        keep_count = text_config.keep_tags_count
        protected = tags[:keep_count]
        shuffleable = tags[keep_count:]
        random.shuffle(shuffleable)
        tags = protected + shuffleable

    return (delimiter + " ").join(tags)


__all__ = [
    "ConceptScanner",
    "CaptionLoader",
    "ImageCaptionPair",
    "IMAGE_EXTENSIONS",
    "load_caption_for_image",
    "discover_images",
    "load_image_caption_pairs",
    "resolve_prompt",
]
