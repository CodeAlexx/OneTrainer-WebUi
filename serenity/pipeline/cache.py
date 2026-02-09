"""Latent and text embedding cache for training acceleration."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """Compute a fast hash of a file's path + mtime + size for cache invalidation."""
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    """Metadata for a single cached tensor."""

    source_path: Path
    cache_path: Path
    source_hash: str


@dataclass
class CacheManager:
    """Disk-based cache for latent tensors and text embeddings.

    Encodes images/prompts once, saves as .pt files, and loads during training
    instead of re-encoding.
    """

    root: Path
    latent_subdir: str = "latents"
    text_subdir: str = "text_embeds"
    _entries: dict[str, CacheEntry] = field(default_factory=dict, repr=False)

    def ensure(self) -> None:
        """Create cache directories."""
        self.latent_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latent_dir(self) -> Path:
        return self.root / self.latent_subdir

    @property
    def text_dir(self) -> Path:
        return self.root / self.text_subdir

    # ------------------------------------------------------------------
    # Latent caching
    # ------------------------------------------------------------------

    def latent_cache_path(self, image_path: Path) -> Path:
        """Return the cache file path for a given image's latent."""
        name = f"{image_path.stem}_{_file_hash(image_path)}.pt"
        return self.latent_dir / name

    def has_latent(self, image_path: Path) -> bool:
        """Check if a valid cached latent exists for an image."""
        cache_path = self.latent_cache_path(image_path)
        return cache_path.exists()

    def save_latent(self, image_path: Path, latent: torch.Tensor) -> Path:
        """Save a latent tensor to the cache."""
        self.ensure()
        cache_path = self.latent_cache_path(image_path)
        torch.save(latent, cache_path)
        self._entries[str(image_path)] = CacheEntry(
            source_path=image_path,
            cache_path=cache_path,
            source_hash=_file_hash(image_path),
        )
        return cache_path

    def load_latent(self, image_path: Path) -> torch.Tensor | None:
        """Load a cached latent tensor, or return None if not cached."""
        cache_path = self.latent_cache_path(image_path)
        if not cache_path.exists():
            return None
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError):
            logger.warning("Failed to load cached latent: %s", cache_path)
            return None

    def encode_and_cache_latent(
        self,
        image_path: Path,
        encode_fn: Callable[[Path], torch.Tensor],
        force: bool = False,
    ) -> torch.Tensor:
        """Encode an image to latent and cache the result.

        Args:
            image_path: Path to the source image.
            encode_fn: Function that takes image_path and returns a latent tensor.
            force: If True, re-encode even if cached.

        Returns:
            The latent tensor (from cache or freshly encoded).
        """
        if not force:
            cached = self.load_latent(image_path)
            if cached is not None:
                return cached

        latent = encode_fn(image_path)
        self.save_latent(image_path, latent)
        return latent

    # ------------------------------------------------------------------
    # Text embedding caching
    # ------------------------------------------------------------------

    def text_cache_path(self, caption: str) -> Path:
        """Return the cache file path for a caption's text embedding."""
        h = hashlib.sha256(caption.encode()).hexdigest()[:16]
        return self.text_dir / f"text_{h}.pt"

    def has_text_embedding(self, caption: str) -> bool:
        """Check if a cached text embedding exists for a caption."""
        return self.text_cache_path(caption).exists()

    def save_text_embedding(self, caption: str, embedding: torch.Tensor) -> Path:
        """Save a text embedding to the cache."""
        self.ensure()
        cache_path = self.text_cache_path(caption)
        torch.save(embedding, cache_path)
        return cache_path

    def load_text_embedding(self, caption: str) -> torch.Tensor | None:
        """Load a cached text embedding, or return None if not cached."""
        cache_path = self.text_cache_path(caption)
        if not cache_path.exists():
            return None
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError):
            logger.warning("Failed to load cached text embedding: %s", cache_path)
            return None

    def encode_and_cache_text(
        self,
        caption: str,
        encode_fn: Callable[[str], torch.Tensor],
        force: bool = False,
    ) -> torch.Tensor:
        """Encode a caption to text embedding and cache the result."""
        if not force:
            cached = self.load_text_embedding(caption)
            if cached is not None:
                return cached

        embedding = encode_fn(caption)
        self.save_text_embedding(caption, embedding)
        return embedding

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all cached files."""
        for d in (self.latent_dir, self.text_dir):
            if d.exists():
                for f in d.iterdir():
                    if f.suffix == ".pt":
                        f.unlink()
        self._entries.clear()

    def invalidate(self, image_path: Path) -> bool:
        """Remove cached latent for a specific image if the source has changed."""
        key = str(image_path)
        entry = self._entries.get(key)
        if entry is None:
            return False

        try:
            current_hash = _file_hash(image_path)
        except FileNotFoundError:
            # Source file deleted, remove cache
            if entry.cache_path.exists():
                entry.cache_path.unlink()
            del self._entries[key]
            return True

        if current_hash != entry.source_hash:
            if entry.cache_path.exists():
                entry.cache_path.unlink()
            del self._entries[key]
            return True

        return False

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        latent_count = sum(1 for _ in self.latent_dir.glob("*.pt")) if self.latent_dir.exists() else 0
        text_count = sum(1 for _ in self.text_dir.glob("*.pt")) if self.text_dir.exists() else 0
        return {
            "latent_cached": latent_count,
            "text_cached": text_count,
            "tracked_entries": len(self._entries),
        }


__all__ = [
    "CacheManager",
    "CacheEntry",
]
