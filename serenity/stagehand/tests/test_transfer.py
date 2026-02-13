"""Phase 3 acceptance tests — AsyncTransferEngine."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest
import torch

from serenity.stagehand.transfer import AsyncTransferEngine, TransferHandle


# ── mock PinnedSlab / PinnedPool ─────────────────────────────────────────


@dataclass
class MockPinnedSlab:
    """Lightweight stand-in for PinnedSlab (CPU testing)."""

    slab_id: int = 0
    buffer: torch.Tensor | None = None
    size_bytes: int = 0
    pool_id: int = 0


class MockPinnedPool:
    """Minimal mock that satisfies the pool interface used by the engine."""

    def __init__(self, slab_bytes: int = 1024 * 1024) -> None:
        self._slab_bytes = slab_bytes

    def acquire(self, size_bytes: int) -> MockPinnedSlab:
        buf = torch.empty(size_bytes, dtype=torch.uint8)
        return MockPinnedSlab(buffer=buf, size_bytes=size_bytes)

    def release(self, slab: object) -> None:
        pass

    @property
    def slab_bytes(self) -> int:
        return self._slab_bytes


# ── helpers ──────────────────────────────────────────────────────────────


def _make_engine(max_inflight: int = 2) -> AsyncTransferEngine:
    pool = MockPinnedPool()
    return AsyncTransferEngine(pool=pool, max_inflight=max_inflight)


def _make_slab(numel: int, dtype: torch.dtype = torch.float32) -> MockPinnedSlab:
    """Create a mock slab with data matching a tensor of *numel* elements."""
    size_bytes = numel * dtype.itemsize
    pin = torch.cuda.is_available()
    buf = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=pin)
    return MockPinnedSlab(slab_id=0, buffer=buf, size_bytes=size_bytes, pool_id=0)


def _device() -> torch.device:
    """Return cuda:0 if available, else cpu."""
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _make_pair(
    numel: int = 256, dtype: torch.dtype = torch.float32
) -> tuple[MockPinnedSlab, torch.Tensor]:
    """Return (slab, gpu_tensor) pair with matching sizes."""
    slab = _make_slab(numel, dtype)
    gpu = torch.zeros(numel, dtype=dtype, device=_device())
    return slab, gpu


# ── tests ────────────────────────────────────────────────────────────────


class TestTransferHandleBasics:
    def test_handle_fields(self) -> None:
        h = TransferHandle(
            handle_id=0,
            block_id="blk.0",
            direction="h2d",
            event=None,
            submitted_at=time.monotonic(),
            size_bytes=1024,
        )
        assert h.handle_id == 0
        assert h.block_id == "blk.0"
        assert h.direction == "h2d"
        assert h.event is None
        assert h.size_bytes == 1024
        assert h.completed is False


class TestBasicTransfers:
    """Engine should work with or without CUDA."""

    def test_h2d_roundtrip(self) -> None:
        engine = _make_engine()
        slab, gpu = _make_pair(64)
        # Write recognizable data into slab.
        slab.buffer[:64 * 4].view(torch.float32).fill_(42.0)

        handle = engine.submit_h2d("blk.0", slab, gpu)
        engine.wait(handle)
        assert handle.completed is True
        assert engine.poll(handle) is True
        assert gpu[0].item() == 42.0

    def test_d2h_roundtrip(self) -> None:
        engine = _make_engine()
        numel = 64
        gpu = torch.full((numel,), 7.0, dtype=torch.float32, device=_device())
        slab = _make_slab(numel, torch.float32)

        handle = engine.submit_d2h("blk.1", gpu, slab)
        engine.wait(handle)
        assert handle.completed is True
        result = slab.buffer[: numel * 4].view(torch.float32)
        assert result[0].item() == 7.0

    def test_poll_and_wait(self) -> None:
        engine = _make_engine()
        slab, gpu = _make_pair(32)
        handle = engine.submit_h2d("blk.0", slab, gpu)
        engine.wait(handle)
        assert engine.poll(handle) is True
        assert handle.completed is True


class TestBackpressure:
    """Backpressure: engine with max_inflight=2 blocks on the 3rd submit."""

    def test_inflight_limit(self) -> None:
        engine = _make_engine(max_inflight=2)

        handles: list[TransferHandle] = []
        for i in range(2):
            slab, gpu = _make_pair(64)
            h = engine.submit_h2d(f"blk.{i}", slab, gpu)
            handles.append(h)

        # With CUDA the first two finish quickly; with CPU they complete immediately.
        # Either way, the third submit should succeed (backpressure reaps completed handles).
        slab3, gpu3 = _make_pair(64)
        h3 = engine.submit_h2d("blk.2", slab3, gpu3)
        engine.wait(h3)
        assert h3.completed is True

    def test_backpressure_with_delayed_completion(self) -> None:
        """Simulate backpressure by keeping handles in inflight manually."""
        engine = _make_engine(max_inflight=2)

        # Directly inject two non-completed handles into inflight.
        fake_h1 = TransferHandle(
            handle_id=100, block_id="fake.0", direction="h2d",
            event=None, submitted_at=time.monotonic(), size_bytes=0,
            completed=False,
        )
        fake_h2 = TransferHandle(
            handle_id=101, block_id="fake.1", direction="h2d",
            event=None, submitted_at=time.monotonic(), size_bytes=0,
            completed=False,
        )
        with engine._lock:
            engine._inflight.append(fake_h1)
            engine._inflight.append(fake_h2)

        # Third submit should block. Use a thread with a timeout.
        submitted = threading.Event()

        def submit_third() -> None:
            slab, gpu = _make_pair(32)
            engine.submit_h2d("blk.2", slab, gpu)
            submitted.set()

        t = threading.Thread(target=submit_third, daemon=True)
        t.start()

        # Give it a moment — should NOT have submitted yet.
        time.sleep(0.05)
        assert not submitted.is_set(), "Third submit should be blocked"

        # Now mark one fake handle as completed so reap frees a slot.
        fake_h1.completed = True

        # Wait for the thread to unblock.
        t.join(timeout=2.0)
        assert submitted.is_set(), "Third submit should have completed after slot freed"


class TestDrain:
    def test_drain_empties_inflight(self) -> None:
        engine = _make_engine(max_inflight=4)
        for i in range(4):
            slab, gpu = _make_pair(64)
            engine.submit_h2d(f"blk.{i}", slab, gpu)

        engine.drain()
        assert engine.inflight_count() == 0

    def test_drain_empty(self) -> None:
        engine = _make_engine()
        engine.drain()  # no-op, should not raise
        assert engine.inflight_count() == 0


class TestStressTransfer:
    def test_100_cycles_no_errors(self) -> None:
        """Run 100 H2D + D2H cycles — no exceptions, no leaks."""
        engine = _make_engine(max_inflight=2)

        for i in range(100):
            numel = 128
            slab_h2d, gpu = _make_pair(numel)
            # Fill slab with recognizable data.
            slab_h2d.buffer[: numel * 4].view(torch.float32).fill_(float(i))

            h = engine.submit_h2d(f"blk.{i}", slab_h2d, gpu)
            engine.wait(h)
            assert gpu[0].item() == float(i)

            # D2H: read back from GPU to a fresh slab.
            slab_d2h = _make_slab(numel, torch.float32)
            h2 = engine.submit_d2h(f"blk.{i}", gpu, slab_d2h)
            engine.wait(h2)
            result = slab_d2h.buffer[: numel * 4].view(torch.float32)
            assert result[0].item() == float(i)

        engine.drain()
        assert engine.inflight_count() == 0

    def test_100_h2d_cycles(self) -> None:
        """100 rapid-fire H2D submits with drain."""
        engine = _make_engine(max_inflight=4)
        for i in range(100):
            slab, gpu = _make_pair(64)
            engine.submit_h2d(f"blk.{i}", slab, gpu)

        engine.drain()
        assert engine.inflight_count() == 0


class TestEngineConsistency:
    """Verify engine behaves consistently regardless of CUDA availability."""

    def test_h2d_d2h_data_integrity(self) -> None:
        engine = _make_engine()
        assert engine._has_cuda == torch.cuda.is_available()

        numel = 256
        slab, gpu = _make_pair(numel)
        slab.buffer[: numel * 4].view(torch.float32).fill_(3.14)

        h = engine.submit_h2d("test.rt", slab, gpu)
        engine.wait(h)
        assert engine.poll(h) is True
        assert abs(gpu[0].item() - 3.14) < 1e-5

    def test_inflight_count_after_reap(self) -> None:
        engine = _make_engine(max_inflight=2)
        slab, gpu = _make_pair(32)
        h = engine.submit_h2d("x", slab, gpu)
        engine.wait(h)
        engine._reap_completed()
        assert engine.inflight_count() == 0
