"""Tests for the inference stage cache."""

from __future__ import annotations

import time

import pytest
import torch

from serenity.inference.cache.store import CacheEntry, StageCache, compute_cache_key


# ---------------------------------------------------------------------------
# compute_cache_key tests
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    """Tests for deterministic cache key generation."""

    def test_same_inputs_same_key(self) -> None:
        k1 = compute_cache_key(prompt="a cat", steps=20)
        k2 = compute_cache_key(prompt="a cat", steps=20)
        assert k1 == k2

    def test_different_inputs_different_key(self) -> None:
        k1 = compute_cache_key(prompt="a cat", steps=20)
        k2 = compute_cache_key(prompt="a dog", steps=20)
        assert k1 != k2

    def test_order_independent(self) -> None:
        k1 = compute_cache_key(a="x", b="y")
        k2 = compute_cache_key(b="y", a="x")
        assert k1 == k2

    def test_handles_tensor(self) -> None:
        t = torch.randn(2, 3)
        k1 = compute_cache_key(tensor=t)
        k2 = compute_cache_key(tensor=t)
        assert k1 == k2

    def test_different_tensors_different_key(self) -> None:
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(2, 3)
        k1 = compute_cache_key(tensor=t1)
        k2 = compute_cache_key(tensor=t2)
        assert k1 != k2

    def test_handles_numbers(self) -> None:
        k1 = compute_cache_key(val=42)
        k2 = compute_cache_key(val=42)
        assert k1 == k2

    def test_handles_strings(self) -> None:
        k1 = compute_cache_key(path="/model.safetensors")
        k2 = compute_cache_key(path="/model.safetensors")
        assert k1 == k2

    def test_key_is_hex_string(self) -> None:
        k = compute_cache_key(foo="bar")
        assert isinstance(k, str)
        assert len(k) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# CacheEntry tests
# ---------------------------------------------------------------------------


class TestCacheEntry:
    """Tests for the CacheEntry dataclass."""

    def test_creation(self) -> None:
        entry = CacheEntry(
            value=torch.zeros(4),
            size_bytes=16,
            last_accessed=time.monotonic(),
            stage="text",
            key="abc",
        )
        assert entry.stage == "text"
        assert entry.key == "abc"
        assert entry.size_bytes == 16


# ---------------------------------------------------------------------------
# StageCache tests
# ---------------------------------------------------------------------------


class TestStageCache:
    """Tests for the LRU stage cache."""

    def test_put_and_get(self) -> None:
        cache = StageCache()
        t = torch.randn(4, 4)
        cache.put("text_encoding", "k1", t)
        result = cache.get("text_encoding", "k1")
        assert result is not None
        assert torch.equal(result, t)

    def test_get_miss_returns_none(self) -> None:
        cache = StageCache()
        assert cache.get("text_encoding", "missing") is None

    def test_get_wrong_stage_returns_none(self) -> None:
        cache = StageCache()
        cache.put("text_encoding", "k1", torch.zeros(2))
        assert cache.get("other_stage", "k1") is None

    def test_invalidate_stage(self) -> None:
        cache = StageCache()
        cache.put("text", "k1", torch.zeros(2))
        cache.put("text", "k2", torch.zeros(2))
        cache.put("unet", "k3", torch.zeros(2))
        cache.invalidate("text")
        assert cache.get("text", "k1") is None
        assert cache.get("text", "k2") is None
        # Other stage untouched
        assert cache.get("unet", "k3") is not None

    def test_invalidate_all(self) -> None:
        cache = StageCache()
        cache.put("text", "k1", torch.zeros(2))
        cache.put("unet", "k2", torch.zeros(2))
        cache.invalidate()
        assert cache.get("text", "k1") is None
        assert cache.get("unet", "k2") is None

    def test_lru_eviction(self) -> None:
        """Oldest entry should be evicted when cache is full."""
        # Allow only 96 bytes (3 * 32 byte entries fit, 4th triggers eviction)
        cache = StageCache(max_memory_bytes=96)

        # Each float32 tensor of 8 elements = 32 bytes
        cache.put("s", "old", torch.zeros(8, dtype=torch.float32))
        time.sleep(0.01)
        cache.put("s", "new", torch.zeros(8, dtype=torch.float32))
        time.sleep(0.01)
        cache.put("s", "newer", torch.zeros(8, dtype=torch.float32))
        time.sleep(0.01)

        # Inserting a 4th should evict the oldest
        cache.put("s", "newest", torch.zeros(8, dtype=torch.float32))

        # "old" should have been evicted
        assert cache.get("s", "old") is None
        # "newest" should still be there
        assert cache.get("s", "newest") is not None

    def test_stats_tracking(self) -> None:
        cache = StageCache()
        cache.put("s", "k1", torch.zeros(4))

        # Hit
        cache.get("s", "k1")
        # Miss
        cache.get("s", "missing")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_memory_tracking(self) -> None:
        cache = StageCache()
        t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
        cache.put("s", "k1", t)
        assert cache.memory_used == 400

    def test_put_dict_value(self) -> None:
        cache = StageCache()
        d = {"emb": torch.randn(4, 8), "mask": torch.ones(4)}
        cache.put("text", "k1", d)
        result = cache.get("text", "k1")
        assert result is not None
        assert isinstance(result, dict)
        assert torch.equal(result["emb"], d["emb"])

    def test_update_existing_entry(self) -> None:
        cache = StageCache()
        cache.put("s", "k1", torch.zeros(4))
        cache.put("s", "k1", torch.ones(4))
        result = cache.get("s", "k1")
        assert result is not None
        assert torch.equal(result, torch.ones(4))
