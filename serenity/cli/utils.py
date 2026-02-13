"""Shared CLI utilities for native training paths.

Type coercion, config helpers, image/caption loading, dataset scanning.
Extracted from native_flux2.py and native_diffusion.py to eliminate duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COND_LABEL_SUFFIXES = ("-condlabel", "_condlabel")

# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_dtype(value: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Convert various formats to torch.dtype."""
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default

    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert various types to boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", "null"}:
        return False
    return default


def coerce_int_or_none(value: Any) -> int | None:
    """Safely convert value to int or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def optional_int(value: Any) -> int | None:
    """Alias for coerce_int_or_none."""
    return coerce_int_or_none(value)


def optional_float(value: Any) -> float | None:
    """Safely convert value to float or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def normalize_model_type(value: Any) -> str:
    """Normalize model type strings to lowercase with underscores."""
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")


def first_config_value(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Cascading config lookup across primary/secondary dicts."""
    for key in keys:
        if key in primary and primary.get(key) is not None:
            return primary.get(key)
        if key in secondary and secondary.get(key) is not None:
            return secondary.get(key)
    return default


def resolve_hf_local_path(model_path: str) -> str:
    """Resolve model paths locally or from HuggingFace cache."""
    expanded = Path(model_path).expanduser()
    if expanded.exists():
        return str(expanded)

    if "/" not in model_path:
        raise FileNotFoundError(
            f"Model path not found locally: {model_path}. "
            "Expected a local path or a cached HF repo id."
        )

    def _ensure_complete_zimage_snapshot(snapshot_path: Path, revision: str | None = None) -> None:
        # Some local Z-Image caches were created without metadata files required
        # by diffusers pipeline loading (model_index/scheduler config).
        if model_path.strip().lower() != "tongyi-mai/z-image":
            return

        required_files = (
            "model_index.json",
            "scheduler/scheduler_config.json",
        )
        missing = [rel for rel in required_files if not (snapshot_path / rel).exists()]
        if not missing:
            return

        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            return

        rev = revision or snapshot_path.name
        for rel in missing:
            try:
                hf_hub_download(
                    repo_id=model_path,
                    filename=rel,
                    revision=rev,
                    local_files_only=False,
                )
            except Exception:
                # Non-fatal: caller can still use component fallback if metadata
                # fetch is unavailable (offline/no network/etc.).
                continue

    org, name = model_path.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{org}--{name}"
    if not repo_dir.exists():
        raise FileNotFoundError(f"HF cache not found for {model_path}: {repo_dir}")

    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            _ensure_complete_zimage_snapshot(snapshot, revision=revision)
            return str(snapshot)

    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
    if snapshots:
        _ensure_complete_zimage_snapshot(snapshots[-1], revision=snapshots[-1].name)
        return str(snapshots[-1])

    raise FileNotFoundError(f"No HF snapshots found in cache for {model_path}")


# ---------------------------------------------------------------------------
# Image / caption loading
# ---------------------------------------------------------------------------


def is_condlabel_image(path: Path) -> bool:
    """Check if image filename has condlabel suffix (reference image marker)."""
    stem = path.stem.lower()
    return any(stem.endswith(suffix) for suffix in COND_LABEL_SUFFIXES)


def strip_condlabel_suffix(stem: str) -> str:
    """Remove condlabel suffix from filename stem."""
    lowered = stem.lower()
    for suffix in COND_LABEL_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_image_tensor(
    image_path: Path, resolution: int, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Load image, resize, normalize to [-1, 1], return as tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=dtype)


def load_caption(image_path: Path, caption_ext: str) -> str:
    """Load caption from sidecar file or generate from filename."""
    caption_path = image_path.with_suffix(caption_ext)
    if caption_path.exists():
        text = caption_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return image_path.stem.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------


def collect_concept_dirs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    """Extract concept directories and caption extensions from config."""
    data_block = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    concepts = data_block.get("concepts") or config.get("concepts") or []

    out: list[tuple[Path, str]] = []
    for concept in concepts:
        if isinstance(concept, str):
            out.append((Path(concept).expanduser(), ".txt"))
            continue

        if not isinstance(concept, dict):
            continue

        concept_path = concept.get("path")
        if not concept_path:
            continue

        caption_ext = str(concept.get("caption_file_ext") or ".txt")
        if not caption_ext.startswith("."):
            caption_ext = f".{caption_ext}"

        out.append((Path(concept_path).expanduser(), caption_ext))

    return out


def build_training_pairs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    """Build list of (image_path, caption) from concept directories."""
    pairs: list[tuple[Path, str]] = []
    for concept_dir, caption_ext in collect_concept_dirs(config):
        if not concept_dir.exists():
            continue
        for image_path in sorted(concept_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if is_condlabel_image(image_path):
                continue
            caption = load_caption(image_path, caption_ext)
            if caption:
                pairs.append((image_path, caption))
    return pairs


def resolve_reference_image_path(image_path: Path) -> Path | None:
    """Find matching reference image for edit training."""
    base_stem = strip_condlabel_suffix(image_path.stem)
    parent = image_path.parent
    for suffix in COND_LABEL_SUFFIXES:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            candidate = parent / f"{base_stem}{suffix}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def build_edit_training_pairs(config: dict[str, Any]) -> list[tuple[Path, Path, str]]:
    """Build list of (target_path, reference_path, caption) for edit training."""
    pairs: list[tuple[Path, Path, str]] = []
    for concept_dir, caption_ext in collect_concept_dirs(config):
        if not concept_dir.exists():
            continue
        for image_path in sorted(concept_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if is_condlabel_image(image_path):
                continue

            reference_path = resolve_reference_image_path(image_path)
            if reference_path is None:
                continue

            caption = load_caption(image_path, caption_ext)
            if caption:
                pairs.append((image_path, reference_path, caption))
    return pairs


__all__ = [
    "IMAGE_EXTENSIONS",
    "COND_LABEL_SUFFIXES",
    "coerce_dtype",
    "as_bool",
    "coerce_int_or_none",
    "optional_int",
    "optional_float",
    "normalize_model_type",
    "first_config_value",
    "resolve_hf_local_path",
    "is_condlabel_image",
    "strip_condlabel_suffix",
    "load_image_tensor",
    "load_caption",
    "collect_concept_dirs",
    "build_training_pairs",
    "resolve_reference_image_path",
    "build_edit_training_pairs",
]
