"""LRU stage-output cache for the inference pipeline."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import torch

__all__ = [
    "CacheEntry",
    "StageCache",
    "compute_cache_key",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Single entry in the stage cache."""

    value: torch.Tensor | dict
    size_bytes: int
    last_accessed: float
    stage: str
    key: str


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


def _tensor_fingerprint(t: torch.Tensor) -> str:
    """Create a deterministic fingerprint of a tensor.

    Uses shape, dtype, and the first few elements so that two tensors
    with different content produce different hashes.
    """
    parts: list[str] = [
        str(tuple(t.shape)),
        str(t.dtype),
    ]
    flat = t.detach().flatten()
    n = min(8, flat.numel())
    if n > 0:
        parts.append(str(flat[:n].tolist()))
    return "|".join(parts)


def compute_cache_key(**kwargs: object) -> str:
    """Hash arbitrary keyword arguments into a deterministic cache key.

    Supports :class:`torch.Tensor`, strings, numbers, and ``Path``-like
    objects.  Arguments are sorted by name so that call order does not
    matter.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    for name in sorted(kwargs):
        val = kwargs[name]
        h.update(name.encode())
        if isinstance(val, torch.Tensor):
            h.update(_tensor_fingerprint(val).encode())
        else:
            h.update(str(val).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage cache
# ---------------------------------------------------------------------------


def _estimate_bytes(value: torch.Tensor | dict) -> int:
    """Estimate memory footprint of *value* in bytes."""
    if isinstance(value, torch.Tensor):
        return value.nelement() * value.element_size()
    total = 0
    for v in value.values():
        if isinstance(v, torch.Tensor):
            total += v.nelement() * v.element_size()
        else:
            # Rough estimate for non-tensor values
            total += len(str(v))
    return max(total, 1)


class StageCache:
    """LRU cache for inference-pipeline stage outputs.

    Stores tensors or dicts keyed by ``(stage, key)`` pairs and evicts
    least-recently-used entries when the memory budget is exceeded.

    Args:
        max_memory_bytes: Maximum total memory in bytes (default 1 GiB).
    """

    def __init__(self, max_memory_bytes: int = 1024 * 1024 * 1024) -> None:
        self._max_bytes = max_memory_bytes
        self._cache: dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, stage: str, key: str) -> torch.Tensor | dict | None:
        """Retrieve a cached value, or ``None`` on miss."""
        full_key = self._make_key(stage, key)
        entry = self._cache.get(full_key)
        if entry is None:
            self._misses += 1
            return None
        entry.last_accessed = time.monotonic()
        self._hits += 1
        return entry.value

    def put(self, stage: str, key: str, value: torch.Tensor | dict) -> None:
        """Store *value* in the cache, evicting LRU entries if needed."""
        full_key = self._make_key(stage, key)
        size = _estimate_bytes(value)

        # If updating an existing entry, remove its old size first
        old = self._cache.get(full_key)
        if old is not None:
            del self._cache[full_key]

        # Evict until there is room
        self._evict_lru(size)

        self._cache[full_key] = CacheEntry(
            value=value,
            size_bytes=size,
            last_accessed=time.monotonic(),
            stage=stage,
            key=key,
        )

    def invalidate(self, stage: str | None = None) -> None:
        """Clear entries for *stage*, or all entries if ``None``."""
        if stage is None:
            self._cache.clear()
            return
        to_remove = [k for k, e in self._cache.items() if e.stage == stage]
        for k in to_remove:
            del self._cache[k]

    @property
    def memory_used(self) -> int:
        """Total bytes currently stored in the cache."""
        return sum(e.size_bytes for e in self._cache.values())

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit / miss statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._cache),
            "memory_bytes": self.memory_used,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(stage: str, key: str) -> str:
        return f"{stage}::{key}"

    def _evict_lru(self, bytes_needed: int) -> None:
        """Remove least-recently-used entries until *bytes_needed* fit."""
        while self.memory_used + bytes_needed > self._max_bytes and self._cache:
            lru_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
            logger.debug("Evicting cache entry %s (%d bytes)", lru_key, self._cache[lru_key].size_bytes)
            del self._cache[lru_key]
