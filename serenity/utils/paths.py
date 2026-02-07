"""Path utilities for file discovery, safe naming, and atomic writes.

Provides helpers matching OneTrainer's ``path_util`` module, plus
additional utilities for safe filename generation and atomic file
operations.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif",
})

SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".webm", ".mkv", ".flv", ".avi", ".mov", ".wmv", ".mp4", ".mpeg", ".m4v",
})

SUPPORTED_CAPTION_EXTENSIONS = frozenset({
    ".txt", ".caption",
})


def supported_image_extensions() -> frozenset[str]:
    """Return the set of supported image file extensions (lowercase, with dot)."""
    return SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> frozenset[str]:
    """Return the set of supported video file extensions (lowercase, with dot)."""
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_image(path: str | Path) -> bool:
    """Check if a path has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_video(path: str | Path) -> bool:
    """Check if a path has a supported video extension."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Safe filename
# ---------------------------------------------------------------------------

_LEGAL_CHARS = frozenset(" ._-#")


def safe_filename(
    text: str,
    allow_spaces: bool = True,
    max_length: int | None = 32,
) -> str:
    """Sanitize text into a safe filename.

    Removes characters that are unsafe for file systems, optionally
    replacing spaces and truncating to ``max_length``.
    """
    if not allow_spaces:
        text = text.replace(" ", "_")

    text = "".join(c for c in text if c.isalnum() or c in _LEGAL_CHARS).strip()

    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    return text.strip()


# ---------------------------------------------------------------------------
# Directory utilities
# ---------------------------------------------------------------------------

def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.  Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonical_join(base_path: str, *paths: str) -> str:
    """Join paths and normalize separators to forward slashes."""
    joined = os.path.join(base_path, *paths)
    return joined.replace("\\", "/")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_files_by_extension(
    directory: str | Path,
    extensions: frozenset[str] | set[str],
    include_subdirectories: bool = False,
    exclude_postfix: list[str] | None = None,
) -> list[Path]:
    """Find files matching any of the given extensions.

    Args:
        directory: Root directory to search.
        extensions: Set of extensions to match (lowercase, with dot).
        include_subdirectories: If True, recurse into subdirectories.
        exclude_postfix: Exclude files whose stems end with these postfixes
            (e.g., ["-masklabel", "-condlabel"]).

    Returns:
        Sorted list of matching file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    exclude_postfix = exclude_postfix or []

    results: list[Path] = []
    if include_subdirectories:
        for ext in extensions:
            for p in directory.rglob(f"*{ext}"):
                if any(p.stem.endswith(pf) for pf in exclude_postfix):
                    continue
                results.append(p)
    else:
        for p in directory.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in extensions:
                continue
            if any(p.stem.endswith(pf) for pf in exclude_postfix):
                continue
            results.append(p)

    return sorted(results)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

@contextmanager
def atomic_write(path: str | Path, mode: str = "w", **kwargs):
    """Context manager for atomic file writes.

    Writes to a temporary file in the same directory, then atomically
    renames to the target path on success.  On failure, the temporary
    file is cleaned up and the original is preserved.

    Example::

        with atomic_write("config.json") as f:
            json.dump(data, f, indent=2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f".{path.name}."
    )
    try:
        with os.fdopen(fd, mode, **kwargs) as f:
            yield f
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_json_atomic(path: str | Path, obj: Any, indent: int = 4) -> None:
    """Write a JSON-serializable object to a file atomically."""
    with atomic_write(path) as f:
        json.dump(obj, f, indent=indent)


__all__ = [
    "SUPPORTED_CAPTION_EXTENSIONS",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "atomic_write",
    "canonical_join",
    "ensure_directory",
    "find_files_by_extension",
    "is_supported_image",
    "is_supported_video",
    "safe_filename",
    "supported_image_extensions",
    "supported_video_extensions",
    "write_json_atomic",
]
