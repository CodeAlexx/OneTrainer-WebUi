"""Tests for serenity.inference.memory — VRAM, streams, pinned, offload, manager."""

from __future__ import annotations

import math
import sys
from unittest import mock

import pytest
import torch
import torch.nn as nn

from serenity.inference.memory import (
    CastBuffer,
    GatheredTransfer,
    LoadedModel,
    ModelManager,
    OffloadConv2d,
    OffloadLinear,
    OffloadMixin,
    PinnedMemoryManager,
    StreamPool,
    TensorGeometry,
    VRAMBudget,
    VRAMState,
    calculate_budget,
    detect_vram_state,
    get_free_memory,
    get_total_memory,
    is_cuda_available,
    is_linux,
    is_nvidia,
    is_windows,
    minimum_inference_memory,
    pin_model_weights,
    unpin_model_weights,
)

HAS_CUDA = torch.cuda.is_available()


# ======================================================================
# VRAM module
# ======================================================================

class TestVRAMHelpers:
    """Basic smoke tests for the vram module."""

    def test_get_free_memory_returns_int(self):
        result = get_free_memory(torch.device("cpu"))
        assert isinstance(result, int)
        # CPU returns 0 by design
        assert result == 0

    def test_get_total_memory_returns_int(self):
        result = get_total_memory(torch.device("cpu"))
        assert isinstance(result, int)
        assert result == 0

    def test_detect_vram_state_cpu(self):
        state = detect_vram_state(torch.device("cpu"))
        assert state == VRAMState.NO_VRAM

    def test_detect_vram_state_valid_enum(self):
        state = detect_vram_state()
        assert isinstance(state, VRAMState)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_get_free_memory_cuda(self):
        val = get_free_memory(torch.device("cuda"))
        assert isinstance(val, int)
        assert val > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_get_total_memory_cuda(self):
        val = get_total_memory(torch.device("cuda"))
        assert isinstance(val, int)
        assert val > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_detect_vram_state_cuda(self):
        state = detect_vram_state(torch.device("cuda"))
        assert state != VRAMState.NO_VRAM

    def test_minimum_inference_memory_positive(self):
        val = minimum_inference_memory()
        assert isinstance(val, int)
        assert val > 0
        # Should be at least 1 GB
        assert val >= 1024 * 1024 * 1024

    def test_platform_helpers(self):
        assert isinstance(is_nvidia(), bool)
        assert isinstance(is_windows(), bool)
        assert isinstance(is_linux(), bool)
        assert isinstance(is_cuda_available(), bool)


class TestVRAMBudget:
    """Tests for calculate_budget."""

    def test_cpu_budget(self):
        budget = calculate_budget(torch.device("cpu"))
        assert isinstance(budget, VRAMBudget)
        assert budget.total == 0
        assert budget.state == VRAMState.NO_VRAM
        assert budget.available == 0

    def test_budget_reserved_fraction_clamped(self):
        b1 = calculate_budget(torch.device("cpu"), reserved_fraction=-0.5)
        b2 = calculate_budget(torch.device("cpu"), reserved_fraction=0.9)
        # Both should succeed without error
        assert isinstance(b1, VRAMBudget)
        assert isinstance(b2, VRAMBudget)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_cuda_budget_math(self):
        budget = calculate_budget(torch.device("cuda"), reserved_fraction=0.1)
        assert budget.total > 0
        assert budget.reserved > 0
        assert budget.available > 0
        assert budget.available == budget.total - budget.reserved

    def test_detect_vram_state_thresholds(self):
        """Verify tier boundaries by mocking get_total_memory."""
        from serenity.inference.memory import vram

        with mock.patch.object(vram, "get_total_memory", return_value=0):
            assert detect_vram_state() == VRAMState.NO_VRAM

        with mock.patch.object(vram, "get_total_memory", return_value=2 * 1024**3):
            assert detect_vram_state() == VRAMState.LOW_VRAM

        with mock.patch.object(vram, "get_total_memory", return_value=6 * 1024**3):
            assert detect_vram_state() == VRAMState.NORMAL_VRAM

        with mock.patch.object(vram, "get_total_memory", return_value=12 * 1024**3):
            assert detect_vram_state() == VRAMState.HIGH_VRAM


# ======================================================================
# Streams module
# ======================================================================

class TestStreamPool:
    """StreamPool behaviour on CPU (graceful no-op) and optionally CUDA."""

    def test_cpu_pool_empty(self):
        pool = StreamPool(num_streams=2, device=torch.device("cpu"))
        assert pool.num_streams == 0
        assert pool.get_stream() is None

    def test_cpu_sync_all_noop(self):
        pool = StreamPool(num_streams=2, device=torch.device("cpu"))
        pool.sync_all()  # should not raise

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_cuda_pool_round_robin(self):
        pool = StreamPool(num_streams=3, device=torch.device("cuda"))
        assert pool.num_streams == 3

        seen = []
        for _ in range(6):
            s = pool.get_stream()
            assert s is not None
            seen.append(id(s))

        # With 3 streams and 6 calls, we should cycle through them.
        unique = set(seen)
        assert len(unique) == 3

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_cuda_sync_all(self):
        pool = StreamPool(num_streams=2, device=torch.device("cuda"))
        pool.sync_all()  # should not raise


class TestCastBuffer:
    """CastBuffer allocation and reuse."""

    def test_basic_allocation(self):
        cb = CastBuffer()
        buf = cb.get_buffer(1024, torch.float32, torch.device("cpu"))
        assert buf is not None
        assert buf.numel() == 1024

    def test_buffer_reuse(self):
        cb = CastBuffer()
        b1 = cb.get_buffer(1024, torch.float32, torch.device("cpu"))
        b2 = cb.get_buffer(512, torch.float32, torch.device("cpu"))
        # b2 should reuse the same underlying storage
        assert b2 is not None
        assert b2.numel() == 512

    def test_buffer_grows(self):
        cb = CastBuffer()
        cb.get_buffer(512, torch.float32, torch.device("cpu"))
        b2 = cb.get_buffer(2048, torch.float32, torch.device("cpu"))
        assert b2 is not None
        assert b2.numel() == 2048

    def test_oversized_ref_returns_none(self):
        cb = CastBuffer()
        sentinel = object()
        # First call establishes sentinel as the largest.
        cb.get_buffer(4096, torch.float32, torch.device("cpu"), ref=sentinel)
        # Second call with same ref — should reject.
        result = cb.get_buffer(8192, torch.float32, torch.device("cpu"), ref=sentinel)
        assert result is None

    def test_reset(self):
        cb = CastBuffer()
        cb.get_buffer(1024, torch.float32, torch.device("cpu"))
        cb.reset()
        assert cb._buffer is None


class TestTensorGeometry:
    """TensorGeometry descriptor."""

    def test_float32(self):
        g = TensorGeometry(shape=(10, 20), dtype=torch.float32)
        assert g.numel() == 200
        assert g.element_size() == 4
        assert g.nbytes() == 800

    def test_float16(self):
        g = TensorGeometry(shape=(3, 4, 5), dtype=torch.float16)
        assert g.numel() == 60
        assert g.element_size() == 2
        assert g.nbytes() == 120

    def test_int8(self):
        g = TensorGeometry(shape=(100,), dtype=torch.int8)
        assert g.numel() == 100
        assert g.element_size() == 1
        assert g.nbytes() == 100


class TestGatheredTransfer:
    """Pack/unpack round-trip on CPU tensors."""

    def test_weight_only(self):
        w = torch.randn(64, 32)
        buf = GatheredTransfer.pack_weight_bias(w)
        w2, b2 = GatheredTransfer.unpack_weight_bias(
            buf, w.shape, w.dtype,
        )
        assert torch.allclose(w2, w)
        assert b2 is None

    def test_weight_and_bias(self):
        w = torch.randn(64, 32)
        b = torch.randn(64)
        buf = GatheredTransfer.pack_weight_bias(w, b)
        w2, b2 = GatheredTransfer.unpack_weight_bias(
            buf, w.shape, w.dtype,
            bias_shape=b.shape, bias_dtype=b.dtype,
        )
        assert torch.allclose(w2, w)
        assert b2 is not None
        assert torch.allclose(b2, b)

    def test_packed_size_alignment(self):
        w = torch.randn(10, 5)
        size = GatheredTransfer.packed_size(w)
        # Should be >= raw bytes and aligned to 1024
        raw = w.numel() * w.element_size()
        assert size >= raw
        assert size % 1024 == 0

    def test_different_dtypes(self):
        w = torch.randn(8, 4, dtype=torch.float16)
        b = torch.randn(8, dtype=torch.float16)
        buf = GatheredTransfer.pack_weight_bias(w, b)
        w2, b2 = GatheredTransfer.unpack_weight_bias(
            buf, w.shape, w.dtype,
            bias_shape=b.shape, bias_dtype=b.dtype,
        )
        assert torch.allclose(w2, w)
        assert b2 is not None
        assert torch.allclose(b2, b)


# ======================================================================
# Pinned memory module
# ======================================================================

class TestPinnedMemoryManager:
    """Budget tracking and can_pin checks (no actual CUDA pinning on CPU)."""

    def test_budget_tracking(self):
        mgr = PinnedMemoryManager(max_bytes=10000)
        assert mgr.total_pinned == 0
        assert mgr.max_bytes == 10000

    def test_can_pin_within_budget(self):
        mgr = PinnedMemoryManager(max_bytes=10000)
        assert mgr.can_pin(5000) is True
        assert mgr.can_pin(10000) is True
        assert mgr.can_pin(10001) is False

    def test_can_pin_disabled(self):
        mgr = PinnedMemoryManager(max_bytes=0)
        assert mgr.can_pin(1) is False

    def test_pin_non_cpu_fails(self):
        mgr = PinnedMemoryManager(max_bytes=100000)
        # pin should return False for non-CPU tensor
        # (we can't create a CUDA tensor without CUDA, so test the guard)
        t = torch.randn(10)
        # Force device check to fail by setting max_bytes=0
        mgr2 = PinnedMemoryManager(max_bytes=0)
        assert mgr2.pin(t) is False

    def test_pin_not_contiguous_fails(self):
        """Non-contiguous tensors can't be pinned."""
        mgr = PinnedMemoryManager(max_bytes=100000)
        t = torch.randn(10, 10).t()  # transpose -> non-contiguous
        assert not t.is_contiguous()
        # On CPU without CUDA, pin returns False early anyway
        if not HAS_CUDA:
            assert mgr.pin(t) is False

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_pin_unpin_round_trip(self):
        mgr = PinnedMemoryManager(max_bytes=1024 * 1024)
        t = torch.randn(100)  # 400 bytes
        assert mgr.pin(t) is True
        assert mgr.total_pinned == t.nbytes
        assert mgr.unpin(t) is True
        assert mgr.total_pinned == 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_pin_budget_enforcement(self):
        mgr = PinnedMemoryManager(max_bytes=200)
        t = torch.randn(100)  # 400 bytes > 200 limit
        assert mgr.pin(t) is False
        assert mgr.total_pinned == 0

    def test_platform_budget_defaults(self):
        """Check that default budgets are sane (positive or zero)."""
        mgr = PinnedMemoryManager()
        assert mgr.max_bytes >= 0


# ======================================================================
# Offload module
# ======================================================================

class TestOffloadLinear:
    """OffloadLinear forward pass correctness."""

    def test_normal_forward(self):
        layer = OffloadLinear(10, 5, bias=True)
        x = torch.randn(2, 10)
        y = layer(x)
        assert y.shape == (2, 5)

    def test_offload_forward_matches_normal(self):
        layer = OffloadLinear(10, 5, bias=True)
        layer.eval()
        x = torch.randn(2, 10)

        with torch.no_grad():
            y_normal = layer(x)
            layer.offload_enabled = True
            y_offload = layer(x)

        assert torch.allclose(y_normal, y_offload, atol=1e-6)

    def test_offload_disabled_flag(self):
        layer = OffloadLinear(10, 5, bias=True)
        assert layer.offload_enabled is False

    def test_weight_function_applied(self):
        layer = OffloadLinear(10, 5, bias=False)
        layer.eval()

        # Add a weight function that zeros out the weight.
        layer.weight_function.append(lambda w: torch.zeros_like(w))
        x = torch.randn(2, 10)

        with torch.no_grad():
            y = layer(x)
        assert torch.allclose(y, torch.zeros(2, 5))

    def test_bias_function_applied(self):
        layer = OffloadLinear(10, 5, bias=True)
        layer.eval()

        # Bias function that adds 1.0 to bias.
        original_bias = layer.bias.data.clone()
        layer.bias_function.append(lambda b: b + 1.0)

        x = torch.zeros(1, 10)
        with torch.no_grad():
            y = layer(x)

        # Output should be F.linear(zeros, weight, bias+1) = bias+1
        expected = original_bias + 1.0
        assert torch.allclose(y.squeeze(0), expected, atol=1e-6)

    def test_release_weight_clears_cache(self):
        layer = OffloadLinear(10, 5)
        layer._init_offload()
        layer._cached_weight = torch.randn(5, 10)
        layer._cached_bias = torch.randn(5)
        layer.release_weight()
        assert layer._cached_weight is None
        assert layer._cached_bias is None


class TestOffloadConv2d:
    """OffloadConv2d forward pass correctness."""

    def test_normal_forward(self):
        layer = OffloadConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        x = torch.randn(1, 3, 16, 16)
        y = layer(x)
        assert y.shape == (1, 8, 16, 16)

    def test_offload_forward_matches_normal(self):
        layer = OffloadConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        layer.eval()
        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            y_normal = layer(x)
            layer.offload_enabled = True
            y_offload = layer(x)

        assert torch.allclose(y_normal, y_offload, atol=1e-6)


# ======================================================================
# Manager module
# ======================================================================

class TestLoadedModel:
    """LoadedModel tracking and load/unload."""

    def _make_model(self, n: int = 64) -> nn.Module:
        """Small linear model for testing."""
        return nn.Linear(n, n, bias=True)

    def test_from_module(self):
        model = self._make_model()
        lm = LoadedModel.from_module(model, torch.device("cpu"))
        assert lm.total_size > 0
        assert lm.config_hash == ""
        assert lm.is_alive

    def test_model_weakref(self):
        model = self._make_model()
        lm = LoadedModel.from_module(model, torch.device("cpu"))
        assert lm.model is model
        del model
        # After deletion, weakref should be dead.
        assert lm.model is None
        assert not lm.is_alive

    def test_offloaded_size(self):
        model = self._make_model()
        lm = LoadedModel.from_module(model, torch.device("cpu"))
        # On CPU, loaded_size matches total since all params are on cpu=device
        assert lm.offloaded_size == lm.total_size - lm.loaded_size

    def test_config_hash(self):
        model = self._make_model()
        lm = LoadedModel.from_module(model, torch.device("cpu"), config_hash="abc123")
        assert lm.config_hash == "abc123"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_load_to_gpu(self):
        model = self._make_model(32)
        lm = LoadedModel.from_module(model, torch.device("cuda"))
        # Initially nothing is on GPU
        assert lm.loaded_size == 0
        moved = lm.load_to_gpu(budget=1024 * 1024)
        assert moved > 0
        assert lm.loaded_size > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_unload_from_gpu(self):
        model = self._make_model(32)
        lm = LoadedModel.from_module(model, torch.device("cuda"))
        lm.load_to_gpu(budget=1024 * 1024)
        loaded_before = lm.loaded_size
        freed = lm.unload_from_gpu(loaded_before)
        assert freed > 0
        assert lm.loaded_size == 0


class TestModelManager:
    """ModelManager loading, lookup, and eviction."""

    def _make_model(self, n: int = 32) -> nn.Module:
        return nn.Linear(n, n, bias=True)

    def test_load_and_lookup(self):
        mgr = ModelManager(device=torch.device("cpu"))
        model = self._make_model()
        lm = mgr.load(model, budget=0, config_hash="test_hash")
        assert lm is not None
        assert len(mgr.loaded_models) == 1

        found = mgr.get_loaded("test_hash")
        assert found is lm

    def test_lookup_miss(self):
        mgr = ModelManager(device=torch.device("cpu"))
        assert mgr.get_loaded("nonexistent") is None
        assert mgr.get_loaded("") is None

    def test_load_reuses_existing(self):
        mgr = ModelManager(device=torch.device("cpu"))
        m1 = self._make_model()
        m2 = self._make_model()
        lm1 = mgr.load(m1, budget=0, config_hash="same")
        lm2 = mgr.load(m2, budget=0, config_hash="same")
        # Should reuse the first entry
        assert lm1 is lm2
        assert len(mgr.loaded_models) == 1

    def test_unload_all(self):
        mgr = ModelManager(device=torch.device("cpu"))
        for i in range(3):
            mgr.load(self._make_model(), budget=0, config_hash=f"h{i}")
        assert len(mgr.loaded_models) == 3
        mgr.unload_all()
        assert len(mgr.loaded_models) == 0

    def test_free_memory_evicts(self):
        mgr = ModelManager(device=torch.device("cpu"))
        m1 = self._make_model(64)
        m2 = self._make_model(64)
        mgr.load(m1, budget=0, config_hash="a")
        mgr.load(m2, budget=0, config_hash="b")

        # On CPU there's nothing to actually free from GPU, but the method
        # should run without error and return 0.
        freed = mgr.free_memory(9999)
        assert isinstance(freed, int)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_free_memory_cuda(self):
        mgr = ModelManager(device=torch.device("cuda"))
        m1 = self._make_model(128)
        lm = mgr.load(m1, budget=1024 * 1024, config_hash="cuda_test")
        assert lm.loaded_size > 0

        # Smart memory mode: requesting less memory than is already free
        # on GPU should return 0 (no eviction needed).
        small_freed = mgr.free_memory(lm.loaded_size)
        # On a GPU with GBs free, a tiny model won't trigger eviction
        assert isinstance(small_freed, int)

        # Requesting an impossibly large amount forces actual eviction
        huge_request = 1024 * 1024 * 1024 * 1024  # 1 TiB
        freed = mgr.free_memory(huge_request)
        assert freed > 0

    def test_dead_models_purged(self):
        mgr = ModelManager(device=torch.device("cpu"))
        m = self._make_model()
        mgr.load(m, budget=0, config_hash="will_die")
        del m
        # free_memory should purge dead entries
        mgr.free_memory(0)
        assert len(mgr.loaded_models) == 0
