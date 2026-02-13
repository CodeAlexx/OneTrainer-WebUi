"""Phase 1 acceptance tests for PinnedPool.

All tests use small pool sizes (total_mb=64, slab_mb=16 -> 4 slabs) for speed.
Tests run without a real GPU — pin_memory falls back to regular memory when
CUDA is unavailable.
"""
from __future__ import annotations

import threading
import time
from unittest import mock

import pytest

from serenity.stagehand.errors import StagehandOOMError
from serenity.stagehand.pool import PinnedPool, PinnedSlab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_MB = 64
SLAB_MB = 16
NUM_SLABS = TOTAL_MB // SLAB_MB  # 4


def _make_pool(**overrides) -> PinnedPool:
    defaults = {"total_mb": TOTAL_MB, "slab_mb": SLAB_MB}
    defaults.update(overrides)
    return _make_pool_raw(**defaults)


def _make_pool_raw(**kwargs) -> PinnedPool:
    return PinnedPool(**kwargs)


# ---------------------------------------------------------------------------
# 1. Basic creation and slab count
# ---------------------------------------------------------------------------

class TestPoolCreation:
    def test_creates_correct_number_of_slabs(self) -> None:
        pool = _make_pool()
        s = pool.stats()
        assert s["total"] == NUM_SLABS
        assert s["free"] == NUM_SLABS
        assert s["in_use"] == 0

    def test_slab_bytes_matches_config(self) -> None:
        pool = _make_pool()
        assert pool.slab_bytes == SLAB_MB * 1024 * 1024

    def test_rejects_indivisible_sizes(self) -> None:
        with pytest.raises(ValueError, match="evenly divisible"):
            PinnedPool(total_mb=100, slab_mb=30)

    def test_rejects_zero_or_negative(self) -> None:
        with pytest.raises(ValueError):
            PinnedPool(total_mb=0, slab_mb=16)
        with pytest.raises(ValueError):
            PinnedPool(total_mb=64, slab_mb=0)


# ---------------------------------------------------------------------------
# 2. Acquire all slabs — free-list must be empty
# ---------------------------------------------------------------------------

class TestAcquireAll:
    def test_acquire_all_slabs(self) -> None:
        pool = _make_pool()
        slabs: list[PinnedSlab] = []
        for _ in range(NUM_SLABS):
            slab = pool.acquire(1)  # 1 byte — fits in a single slab
            assert isinstance(slab, PinnedSlab)
            slabs.append(slab)

        s = pool.stats()
        assert s["free"] == 0
        assert s["in_use"] == NUM_SLABS

        # Release them all.
        for slab in slabs:
            pool.release(slab)
        s = pool.stats()
        assert s["free"] == NUM_SLABS
        assert s["in_use"] == 0


# ---------------------------------------------------------------------------
# 3 & 4. Blocking acquire + release wakes blocked thread
# ---------------------------------------------------------------------------

class TestBlockingAcquire:
    def test_acquire_blocks_and_release_wakes(self) -> None:
        pool = _make_pool()

        # Exhaust the pool.
        slabs = [pool.acquire(1) for _ in range(NUM_SLABS)]
        assert pool.stats()["free"] == 0

        # Another thread tries to acquire — should block.
        result: list[PinnedSlab | None] = [None]
        acquired_event = threading.Event()
        started_event = threading.Event()

        def _blocked_acquire() -> None:
            started_event.set()
            result[0] = pool.acquire(1)
            acquired_event.set()

        t = threading.Thread(target=_blocked_acquire, daemon=True)
        t.start()

        # Wait until the thread has started (it should then block inside acquire).
        started_event.wait(timeout=1.0)
        time.sleep(0.05)  # Give the thread time to actually block.

        # It must NOT have acquired yet.
        assert not acquired_event.is_set(), "Thread acquired a slab when pool was empty"

        # Release one slab — this should wake the blocked thread.
        pool.release(slabs.pop())
        acquired_event.wait(timeout=1.0)
        assert acquired_event.is_set(), "Blocked thread did not wake after release"
        assert isinstance(result[0], PinnedSlab)

        t.join(timeout=1.0)

        # Clean up remaining.
        pool.release(result[0])
        for s in slabs:
            pool.release(s)


# ---------------------------------------------------------------------------
# 5. Stats correctness after full cycle
# ---------------------------------------------------------------------------

class TestStatsAfterCycle:
    def test_stats_after_full_acquire_release_cycle(self) -> None:
        pool = _make_pool()
        slabs = [pool.acquire(1) for _ in range(NUM_SLABS)]
        for s in slabs:
            pool.release(s)

        s = pool.stats()
        assert s["total"] == NUM_SLABS
        assert s["free"] == NUM_SLABS
        assert s["in_use"] == 0
        assert s["peak_in_use"] == NUM_SLABS


# ---------------------------------------------------------------------------
# 6. 1000 acquire/release cycles — no crashes, consistent stats
# ---------------------------------------------------------------------------

class TestStressCycles:
    def test_thousand_cycles(self) -> None:
        pool = _make_pool()
        for _ in range(1000):
            slab = pool.acquire(1)
            pool.release(slab)

        s = pool.stats()
        assert s["total"] == NUM_SLABS
        assert s["free"] == NUM_SLABS
        assert s["in_use"] == 0
        assert s["peak_in_use"] == 1  # Only ever held 1 at a time.
        assert s["wait_count"] == 0   # Free-list was never empty.


# ---------------------------------------------------------------------------
# 7. Stats report correct counts throughout lifecycle
# ---------------------------------------------------------------------------

class TestStatsProgressive:
    def test_stats_track_progressive_acquire(self) -> None:
        pool = _make_pool()
        slabs: list[PinnedSlab] = []
        for i in range(NUM_SLABS):
            slabs.append(pool.acquire(1))
            s = pool.stats()
            assert s["in_use"] == i + 1
            assert s["free"] == NUM_SLABS - (i + 1)
            assert s["peak_in_use"] == i + 1

        # Release in reverse.
        for i, slab in enumerate(reversed(slabs)):
            pool.release(slab)
            s = pool.stats()
            assert s["in_use"] == NUM_SLABS - (i + 1)
            assert s["free"] == i + 1
            assert s["peak_in_use"] == NUM_SLABS


# ---------------------------------------------------------------------------
# 8. Acquiring more than pool capacity raises StagehandOOMError
# ---------------------------------------------------------------------------

class TestOOMError:
    def test_oversized_request_raises_immediately(self) -> None:
        pool = _make_pool()
        # Request more bytes than total pool capacity.
        too_large = (TOTAL_MB + 1) * 1024 * 1024
        with pytest.raises(StagehandOOMError):
            pool.acquire(too_large)

    def test_exactly_pool_capacity_succeeds(self) -> None:
        pool = _make_pool()
        result = pool.acquire(TOTAL_MB * 1024 * 1024)
        assert isinstance(result, list)
        assert len(result) == NUM_SLABS
        pool.release(result)


# ---------------------------------------------------------------------------
# 9. Thread-safety: concurrent acquire/release from 4 threads
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_acquire_release_no_corruption(self) -> None:
        pool = _make_pool()
        errors: list[Exception] = []
        iterations_per_thread = 250
        barrier = threading.Barrier(4)

        def _worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(iterations_per_thread):
                    slab = pool.acquire(1)
                    # Simulate a tiny amount of work.
                    time.sleep(0)
                    pool.release(slab)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Thread errors: {errors}"

        s = pool.stats()
        assert s["in_use"] == 0
        assert s["free"] == NUM_SLABS
        # Peak could be up to NUM_SLABS if all 4 threads held slabs concurrently.
        assert 1 <= s["peak_in_use"] <= NUM_SLABS


# ---------------------------------------------------------------------------
# 10. Release with wrong pool_id raises AssertionError
# ---------------------------------------------------------------------------

class TestWrongPoolId:
    def test_release_wrong_pool_raises(self) -> None:
        pool = _make_pool()
        slab = pool.acquire(1)

        # Tamper with the pool_id.
        fake_slab = PinnedSlab(
            slab_id=slab.slab_id,
            buffer=slab.buffer,
            size_bytes=slab.size_bytes,
            pool_id=slab.pool_id + 1,  # wrong!
        )
        with pytest.raises(AssertionError):
            pool.release(fake_slab)

        # Clean up the real slab.
        pool.release(slab)


# ---------------------------------------------------------------------------
# 11. Multi-slab acquire for large requests
# ---------------------------------------------------------------------------

class TestMultiSlabAcquire:
    def test_acquire_two_slabs(self) -> None:
        pool = _make_pool()
        # Request slightly more than 1 slab.
        two_slab_bytes = SLAB_MB * 1024 * 1024 + 1
        result = pool.acquire(two_slab_bytes)
        assert isinstance(result, list)
        assert len(result) == 2

        s = pool.stats()
        assert s["in_use"] == 2
        assert s["free"] == NUM_SLABS - 2

        pool.release(result)
        s = pool.stats()
        assert s["in_use"] == 0
        assert s["free"] == NUM_SLABS


# ---------------------------------------------------------------------------
# 12. Wait stats are tracked
# ---------------------------------------------------------------------------

class TestWaitStats:
    def test_wait_count_incremented_on_block(self) -> None:
        pool = _make_pool()
        # Exhaust pool.
        slabs = [pool.acquire(1) for _ in range(NUM_SLABS)]

        released = threading.Event()
        acquired = threading.Event()

        def _delayed_release() -> None:
            time.sleep(0.05)
            pool.release(slabs[0])
            released.set()

        t_release = threading.Thread(target=_delayed_release, daemon=True)
        t_release.start()

        # This will block briefly until the release thread frees a slab.
        slab = pool.acquire(1)
        acquired.set()
        t_release.join(timeout=2.0)

        s = pool.stats()
        assert s["wait_count"] >= 1
        assert s["total_wait_ms"] > 0

        # Clean up.
        pool.release(slab)
        for s_obj in slabs[1:]:
            pool.release(s_obj)


# ---------------------------------------------------------------------------
# 13. acquire() must NEVER call torch.empty() — critical spec requirement
# ---------------------------------------------------------------------------

class TestAcquireNoAllocation:
    def test_acquire_never_calls_torch_empty(self) -> None:
        """Spec 2.1.1: acquire() must never call torch.empty() or any allocator."""
        pool = _make_pool()
        import torch

        original_empty = torch.empty
        calls_during_acquire: list[tuple] = []

        def _tracking_empty(*args, **kwargs):
            calls_during_acquire.append((args, kwargs))
            return original_empty(*args, **kwargs)

        # After init, patch torch.empty to track calls.
        with mock.patch("torch.empty", side_effect=_tracking_empty):
            # Acquire and release several times — no torch.empty should be called.
            for _ in range(10):
                slab = pool.acquire(1)
                pool.release(slab)

        assert len(calls_during_acquire) == 0, (
            f"torch.empty was called {len(calls_during_acquire)} times during acquire/release"
        )

    def test_acquire_multi_slab_no_allocation(self) -> None:
        """Multi-slab acquire must also avoid torch.empty()."""
        pool = _make_pool()
        import torch

        original_empty = torch.empty
        calls: list[tuple] = []

        def _tracking_empty(*args, **kwargs):
            calls.append((args, kwargs))
            return original_empty(*args, **kwargs)

        with mock.patch("torch.empty", side_effect=_tracking_empty):
            # Acquire 2 slabs at once.
            two_slab_bytes = SLAB_MB * 1024 * 1024 + 1
            result = pool.acquire(two_slab_bytes)
            pool.release(result)

        assert len(calls) == 0, (
            f"torch.empty was called {len(calls)} times during multi-slab acquire"
        )
