"""Model directory scanner for discovering available models on disk."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["ModelInfo", "ModelScanner"]

# File extensions that correspond to model weight files.
_MODEL_EXTENSIONS: frozenset[str] = frozenset({
    ".safetensors",
    ".ckpt",
    ".pt",
    ".bin",
    ".gguf",
})

# Extensions to explicitly skip (documentation, metadata, images, databases).
_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".json", ".txt", ".md", ".jpg", ".jpeg", ".png",
    ".ldb", ".yaml", ".yml", ".csv", ".log", ".html", ".css", ".js",
})

# Maps subdirectory names (lowercased) to a canonical category label.
_CATEGORY_MAP: dict[str, str] = {
    "stable-diffusion": "checkpoint",
    "diffusion_models": "diffusion_model",
    "lora": "lora",
    "vae": "vae",
    "embeddings": "embedding",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
}


@dataclass(slots=True)
class ModelInfo:
    """Metadata for a single discovered model file."""

    name: str
    path: str
    size_mb: float
    format: str
    category: str
    modified: float


@dataclass
class ModelScanner:
    """Scans one or more root directories for model weight files.

    Results are cached after the first scan.  Call ``refresh()`` to
    force a re-scan.
    """

    root_dirs: list[str]
    _cache: dict[str, list[ModelInfo]] = field(default_factory=dict, repr=False, init=False)
    _cache_time: float = field(default=0.0, repr=False, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_all(self) -> dict[str, list[ModelInfo]]:
        """Scan every recognised subdirectory and return models grouped by category."""
        if self._cache:
            return self._cache

        result: dict[str, list[ModelInfo]] = {}
        for root in self.root_dirs:
            root_path = Path(root)
            if not root_path.is_dir():
                logger.warning("Model root directory does not exist: %s", root)
                continue

            for entry in sorted(root_path.iterdir()):
                if not entry.is_dir():
                    continue
                category = _CATEGORY_MAP.get(entry.name.lower(), entry.name.lower())
                models = self._scan_directory(entry, category)
                if models:
                    existing = result.get(category, [])
                    existing.extend(models)
                    result[category] = existing

        # Sort each category by modification time (newest first).
        for category in result:
            result[category].sort(key=lambda m: m.modified, reverse=True)

        self._cache = result
        self._cache_time = time.monotonic()
        logger.info(
            "Scanned %d model(s) across %d categor(ies)",
            sum(len(v) for v in result.values()),
            len(result),
        )
        return result

    def scan_checkpoints(self) -> list[ModelInfo]:
        """Return checkpoint and diffusion_model entries."""
        all_models = self.scan_all()
        out: list[ModelInfo] = []
        out.extend(all_models.get("checkpoint", []))
        out.extend(all_models.get("diffusion_model", []))
        return out

    def scan_loras(self) -> list[ModelInfo]:
        """Return LoRA entries."""
        return self.scan_all().get("lora", [])

    def scan_vaes(self) -> list[ModelInfo]:
        """Return VAE entries."""
        return self.scan_all().get("vae", [])

    def scan_embeddings(self) -> list[ModelInfo]:
        """Return embedding / textual-inversion entries."""
        return self.scan_all().get("embedding", [])

    def refresh(self) -> dict[str, list[ModelInfo]]:
        """Clear cache and perform a fresh scan."""
        self._cache.clear()
        self._cache_time = 0.0
        return self.scan_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_directory(directory: Path, category: str) -> list[ModelInfo]:
        """Recursively scan *directory* for model files."""
        models: list[ModelInfo] = []
        try:
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    fpath = Path(root) / fname
                    ext = fpath.suffix.lower()
                    if ext not in _MODEL_EXTENSIONS:
                        continue

                    try:
                        stat = fpath.stat()
                    except OSError:
                        continue

                    models.append(ModelInfo(
                        name=fpath.stem,
                        path=str(fpath),
                        size_mb=round(stat.st_size / (1024 * 1024), 2),
                        format=ext.lstrip("."),
                        category=category,
                        modified=stat.st_mtime,
                    ))
        except OSError as exc:
            logger.warning("Error scanning %s: %s", directory, exc)

        return models
