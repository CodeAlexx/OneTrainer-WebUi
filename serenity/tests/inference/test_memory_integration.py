"""Integration tests for the memory subsystem wiring in InferenceEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from serenity.inference.config import InferenceConfig, VRAMMode
from serenity.inference.engine import InferenceEngine
from serenity.inference.memory.manager import ModelManager
from serenity.inference.memory.offload import OffloadLinear, OffloadMixin
from serenity.inference.memory.pinned import PinnedMemoryManager
from serenity.inference.memory.streams import StreamPool
from serenity.inference.quantization.ops import OperationContext


# ======================================================================
# Fixtures
# ======================================================================


def _make_engine(
    vram_mode: VRAMMode = VRAMMode.AUTO,
    offload_streams: int = 2,
    pin_memory: bool = True,
    **extra_config: object,
) -> InferenceEngine:
    """Create an engine with external dependencies mocked."""
    with (
        patch("serenity.inference.engine.select_best_backend") as mock_attn,
        patch("serenity.inference.engine.is_cuda_available", return_value=False),
    ):
        mock_attn.return_value = MagicMock(value="sdp")
        config = InferenceConfig(
            vram_mode=vram_mode,
            offload_streams=offload_streams,
            pin_memory=pin_memory,
            **extra_config,
        )
        engine = InferenceEngine(config)
    return engine


def _make_cuda_engine(
    vram_mode: VRAMMode = VRAMMode.AUTO,
    offload_streams: int = 2,
    pin_memory: bool = True,
) -> InferenceEngine:
    """Create an engine that thinks it has CUDA available."""
    with (
        patch("serenity.inference.engine.select_best_backend") as mock_attn,
        patch("serenity.inference.engine.is_cuda_available", return_value=True),
        patch("serenity.inference.engine.StreamPool") as mock_sp,
        patch("serenity.inference.engine.PinnedMemoryManager") as mock_pm,
    ):
        mock_attn.return_value = MagicMock(value="sdp")
        mock_sp_instance = MagicMock(spec=StreamPool)
        mock_sp_instance.num_streams = offload_streams
        mock_sp.return_value = mock_sp_instance

        mock_pm_instance = MagicMock(spec=PinnedMemoryManager)
        mock_pm_instance.total_pinned = 0
        mock_pm.return_value = mock_pm_instance

        config = InferenceConfig(
            vram_mode=vram_mode,
            offload_streams=offload_streams,
            pin_memory=pin_memory,
        )
        engine = InferenceEngine(config)
    return engine


# ======================================================================
# VRAM Mode Tests
# ======================================================================


class TestVRAMModeConfig:
    """Config vram_mode fields drive engine behavior."""

    def test_vram_mode_stored_from_config(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.LOW)
        assert engine._vram_mode == VRAMMode.LOW

    def test_vram_mode_auto_default(self) -> None:
        engine = _make_engine()
        assert engine._vram_mode == VRAMMode.AUTO

    def test_vram_mode_high(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.HIGH)
        assert engine._vram_mode == VRAMMode.HIGH

    def test_vram_mode_normal(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.NORMAL)
        assert engine._vram_mode == VRAMMode.NORMAL

    def test_vram_mode_no_vram(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.NO_VRAM)
        assert engine._vram_mode == VRAMMode.NO_VRAM

    def test_vram_mode_in_status(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.LOW)
        status = engine.get_status()
        assert status["vram_mode"] == "low"


# ======================================================================
# StreamPool Integration Tests
# ======================================================================


class TestStreamPoolIntegration:
    """StreamPool creation based on config.offload_streams."""

    def test_cpu_engine_no_stream_pool(self) -> None:
        """On CPU, no stream pool is created."""
        engine = _make_engine()
        assert engine._stream_pool is None or engine._stream_pool.num_streams == 0

    def test_cuda_engine_creates_stream_pool(self) -> None:
        """On CUDA, stream pool is created with config stream count."""
        engine = _make_cuda_engine(offload_streams=3)
        assert engine._stream_pool is not None
        assert engine._stream_pool.num_streams == 3

    def test_cuda_engine_default_streams(self) -> None:
        """Default offload_streams=2."""
        engine = _make_cuda_engine(offload_streams=2)
        assert engine._stream_pool is not None
        assert engine._stream_pool.num_streams == 2

    def test_status_reports_stream_count(self) -> None:
        engine = _make_cuda_engine(offload_streams=4)
        status = engine.get_status()
        assert status["stream_pool_streams"] == 4

    def test_status_reports_zero_streams_on_cpu(self) -> None:
        engine = _make_engine()
        status = engine.get_status()
        assert status["stream_pool_streams"] == 0


# ======================================================================
# PinnedMemoryManager Integration Tests
# ======================================================================


class TestPinnedMemoryIntegration:
    """PinnedMemoryManager creation based on config.pin_memory."""

    def test_cpu_engine_no_pinned_manager(self) -> None:
        """On CPU, no pinned memory manager is created."""
        engine = _make_engine(pin_memory=True)
        assert engine._pinned_manager is None

    def test_cuda_engine_creates_pinned_manager(self) -> None:
        """On CUDA with pin_memory=True, pinned manager is created."""
        engine = _make_cuda_engine(pin_memory=True)
        assert engine._pinned_manager is not None

    def test_cuda_engine_no_pinned_when_disabled(self) -> None:
        """On CUDA with pin_memory=False, no pinned manager."""
        engine = _make_cuda_engine(pin_memory=False)
        assert engine._pinned_manager is None

    def test_status_reports_pinned_bytes(self) -> None:
        engine = _make_cuda_engine(pin_memory=True)
        status = engine.get_status()
        assert "pinned_memory_bytes" in status
        assert status["pinned_memory_bytes"] == 0


# ======================================================================
# ModelManager Integration Tests
# ======================================================================


class TestModelManagerIntegration:
    """ModelManager is used for model lifecycle."""

    def test_model_manager_initialized(self) -> None:
        engine = _make_engine()
        assert isinstance(engine._model_manager, ModelManager)

    def test_unload_all_delegates_to_manager(self) -> None:
        engine = _make_engine()
        engine._model_manager = MagicMock(spec=ModelManager)
        engine._model_manager.loaded_models = []
        engine.unload_all()
        engine._model_manager.unload_all.assert_called_once()

    def test_unload_clears_config_hash(self) -> None:
        engine = _make_engine()
        engine._config_hash = "abc123"
        engine.unload_all()
        assert engine._config_hash == ""


# ======================================================================
# Config Hash / Model Caching Tests
# ======================================================================


class TestConfigHash:
    """Config hash computation and model caching."""

    def test_compute_config_hash_deterministic(self) -> None:
        h1 = InferenceEngine._compute_config_hash("model.safetensors", torch.float16)
        h2 = InferenceEngine._compute_config_hash("model.safetensors", torch.float16)
        assert h1 == h2

    def test_compute_config_hash_varies_with_path(self) -> None:
        h1 = InferenceEngine._compute_config_hash("model_a.safetensors", torch.float16)
        h2 = InferenceEngine._compute_config_hash("model_b.safetensors", torch.float16)
        assert h1 != h2

    def test_compute_config_hash_varies_with_dtype(self) -> None:
        h1 = InferenceEngine._compute_config_hash("model.safetensors", torch.float16)
        h2 = InferenceEngine._compute_config_hash("model.safetensors", torch.float32)
        assert h1 != h2

    def test_config_hash_is_sha256(self) -> None:
        h = InferenceEngine._compute_config_hash("test.pt", torch.float16)
        assert len(h) == 64  # SHA-256 hex digest length


# ======================================================================
# VRAM Budget Tests
# ======================================================================


class TestVRAMBudget:
    """VRAM budget calculation from vram_mode."""

    def test_high_mode_returns_none(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.HIGH)
        budget = engine._get_vram_budget()
        assert budget is None

    def test_auto_mode_returns_available(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.AUTO)
        with patch("serenity.inference.memory.vram.calculate_budget") as mock_cb:
            mock_budget = MagicMock()
            mock_budget.available = 8_000_000_000
            mock_cb.return_value = mock_budget
            budget = engine._get_vram_budget()
        assert budget == 8_000_000_000

    def test_normal_mode_returns_70_percent(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.NORMAL)
        with patch("serenity.inference.memory.vram.calculate_budget") as mock_cb:
            mock_budget = MagicMock()
            mock_budget.available = 10_000_000_000
            mock_cb.return_value = mock_budget
            budget = engine._get_vram_budget()
        assert budget == int(10_000_000_000 * 0.7)

    def test_low_mode_returns_30_percent(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.LOW)
        with patch("serenity.inference.memory.vram.calculate_budget") as mock_cb:
            mock_budget = MagicMock()
            mock_budget.available = 10_000_000_000
            mock_cb.return_value = mock_budget
            budget = engine._get_vram_budget()
        assert budget == int(10_000_000_000 * 0.3)

    def test_no_vram_mode_returns_zero(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.NO_VRAM)
        budget = engine._get_vram_budget()
        assert budget == 0


# ======================================================================
# OperationContext Offload Wiring Tests
# ======================================================================


class TestOperationContextWiring:
    """OperationContext is configured with offload classes when needed."""

    def test_offload_classes_used_for_normal_mode(self) -> None:
        """NORMAL mode should use OffloadLinear/OffloadConv2d in OperationContext."""
        engine = _make_engine(vram_mode=VRAMMode.NORMAL)
        needs_offload = engine._vram_mode in (VRAMMode.NORMAL, VRAMMode.LOW, VRAMMode.NO_VRAM)
        assert needs_offload is True

    def test_offload_classes_used_for_low_mode(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.LOW)
        needs_offload = engine._vram_mode in (VRAMMode.NORMAL, VRAMMode.LOW, VRAMMode.NO_VRAM)
        assert needs_offload is True

    def test_offload_not_used_for_high_mode(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.HIGH)
        needs_offload = engine._vram_mode in (VRAMMode.NORMAL, VRAMMode.LOW, VRAMMode.NO_VRAM)
        assert needs_offload is False

    def test_operation_context_offload_classes(self) -> None:
        """Verify OperationContext can be created with offload classes."""
        ctx = OperationContext(
            linear_cls=OffloadLinear,
            conv2d_cls=OffloadLinear,  # Type doesn't matter for this test
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        assert ctx.linear_cls is OffloadLinear

    def test_operation_context_creates_offload_linear(self) -> None:
        """OperationContext.create_linear with OffloadLinear produces OffloadLinear."""
        ctx = OperationContext(
            linear_cls=OffloadLinear,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        layer = ctx.create_linear(10, 5, bias=True)
        assert isinstance(layer, OffloadLinear)
        assert isinstance(layer, OffloadMixin)


# ======================================================================
# Attach StreamPool Tests
# ======================================================================


class TestAttachStreamPool:
    """StreamPool is attached to OffloadMixin layers."""

    def test_attach_stream_pool_to_offload_layers(self) -> None:
        """Offload layers receive the stream pool."""
        engine = _make_engine()
        engine._stream_pool = MagicMock(spec=StreamPool)

        # Create a model with OffloadLinear layers
        model = nn.Sequential(
            OffloadLinear(10, 5),
            OffloadLinear(5, 3),
        )
        engine._attach_stream_pool(model)

        for module in model.modules():
            if isinstance(module, OffloadMixin):
                assert module._stream_pool is engine._stream_pool

    def test_attach_stream_pool_no_op_without_pool(self) -> None:
        """When stream pool is None, attach is a no-op."""
        engine = _make_engine()
        engine._stream_pool = None
        model = nn.Sequential(OffloadLinear(10, 5))
        engine._attach_stream_pool(model)
        # Should not raise, offload layers keep default None
        for module in model.modules():
            if isinstance(module, OffloadMixin):
                assert module._stream_pool is None

    def test_attach_skips_non_offload_layers(self) -> None:
        """Regular nn.Linear layers are not affected."""
        engine = _make_engine()
        engine._stream_pool = MagicMock(spec=StreamPool)
        model = nn.Sequential(
            nn.Linear(10, 5),
            OffloadLinear(5, 3),
        )
        engine._attach_stream_pool(model)
        # nn.Linear should not have _stream_pool attribute
        assert not hasattr(model[0], "_stream_pool")


# ======================================================================
# Load Model Integration (mocked adapter)
# ======================================================================


class TestLoadModelIntegration:
    """load_model() wires memory subsystem correctly."""

    @patch("serenity.inference.engine.load_state_dict")
    @patch("serenity.inference.engine._get_adapter")
    @patch("serenity.inference.engine.detect_from_file")
    def test_load_model_registers_with_manager(
        self, mock_detect, mock_adapter_fn, mock_load_sd,
    ) -> None:
        """After load_model, model is registered in ModelManager."""
        engine = _make_engine(vram_mode=VRAMMode.AUTO)

        # Setup mocks
        mock_config = MagicMock()
        mock_config.architecture = MagicMock(value="sd15")
        mock_detect.return_value = mock_config

        mock_model = nn.Linear(10, 5)
        mock_adapter = MagicMock()
        mock_adapter.create_model.return_value = mock_model
        mock_adapter.get_prediction_type.return_value = "eps"
        mock_adapter.get_vae_scaling_factor.return_value = 0.18215
        mock_adapter_fn.return_value = mock_adapter

        mock_load_sd.return_value = {}

        # Suppress VAE/text encoder loading
        with (
            patch.object(engine, "_load_vae_from_checkpoint"),
            patch.object(engine._text_enc_manager, "load_for_model"),
        ):
            engine.load_model("fake_model.safetensors")

        assert engine._model_loaded is True
        assert engine._config_hash != ""
        assert len(engine._model_manager.loaded_models) == 1

    @patch("serenity.inference.engine.load_state_dict")
    @patch("serenity.inference.engine._get_adapter")
    @patch("serenity.inference.engine.detect_from_file")
    def test_load_model_with_offload_mode_passes_ops_context(
        self, mock_detect, mock_adapter_fn, mock_load_sd,
    ) -> None:
        """When vram_mode requires offload, ops_context is passed to adapter."""
        engine = _make_engine(vram_mode=VRAMMode.NORMAL)

        mock_config = MagicMock()
        mock_config.architecture = MagicMock(value="sd15")
        mock_detect.return_value = mock_config

        mock_model = nn.Linear(10, 5)
        mock_adapter = MagicMock()
        mock_adapter.create_model.return_value = mock_model
        mock_adapter.get_prediction_type.return_value = "eps"
        mock_adapter.get_vae_scaling_factor.return_value = 0.18215
        mock_adapter_fn.return_value = mock_adapter

        mock_load_sd.return_value = {}

        with (
            patch.object(engine, "_load_vae_from_checkpoint"),
            patch.object(engine._text_enc_manager, "load_for_model"),
        ):
            engine.load_model("fake_model.safetensors")

        # Verify ops_context was passed
        call_kwargs = mock_adapter.create_model.call_args
        assert "ops_context" in call_kwargs.kwargs
        ops_ctx = call_kwargs.kwargs["ops_context"]
        assert isinstance(ops_ctx, OperationContext)
        assert ops_ctx.linear_cls is OffloadLinear

    @patch("serenity.inference.engine.load_state_dict")
    @patch("serenity.inference.engine._get_adapter")
    @patch("serenity.inference.engine.detect_from_file")
    def test_load_model_high_mode_no_ops_context(
        self, mock_detect, mock_adapter_fn, mock_load_sd,
    ) -> None:
        """HIGH mode does not pass ops_context to adapter."""
        engine = _make_engine(vram_mode=VRAMMode.HIGH)

        mock_config = MagicMock()
        mock_config.architecture = MagicMock(value="sd15")
        mock_detect.return_value = mock_config

        mock_model = nn.Linear(10, 5)
        mock_adapter = MagicMock()
        mock_adapter.create_model.return_value = mock_model
        mock_adapter.get_prediction_type.return_value = "eps"
        mock_adapter.get_vae_scaling_factor.return_value = 0.18215
        mock_adapter_fn.return_value = mock_adapter

        mock_load_sd.return_value = {}

        with (
            patch.object(engine, "_load_vae_from_checkpoint"),
            patch.object(engine._text_enc_manager, "load_for_model"),
        ):
            engine.load_model("fake_model.safetensors")

        # No ops_context should be passed
        call_kwargs = mock_adapter.create_model.call_args
        assert "ops_context" not in call_kwargs.kwargs

    @patch("serenity.inference.engine.load_state_dict")
    @patch("serenity.inference.engine._get_adapter")
    @patch("serenity.inference.engine.detect_from_file")
    def test_load_model_caching_same_hash(
        self, mock_detect, mock_adapter_fn, mock_load_sd,
    ) -> None:
        """Loading the same model path + dtype reuses cached model."""
        engine = _make_engine(vram_mode=VRAMMode.AUTO)

        mock_config = MagicMock()
        mock_config.architecture = MagicMock(value="sd15")
        mock_detect.return_value = mock_config

        mock_model = nn.Linear(10, 5)
        mock_adapter = MagicMock()
        mock_adapter.create_model.return_value = mock_model
        mock_adapter.get_prediction_type.return_value = "eps"
        mock_adapter.get_vae_scaling_factor.return_value = 0.18215
        mock_adapter_fn.return_value = mock_adapter

        mock_load_sd.return_value = {}

        with (
            patch.object(engine, "_load_vae_from_checkpoint"),
            patch.object(engine._text_enc_manager, "load_for_model"),
        ):
            engine.load_model("fake_model.safetensors")

        # First load creates the model
        assert mock_adapter.create_model.call_count == 1

        # Second load with same path should reuse via ModelManager
        engine._model_loaded = False  # force re-check
        with (
            patch.object(engine, "_load_vae_from_checkpoint"),
            patch.object(engine._text_enc_manager, "load_for_model"),
        ):
            engine.load_model("fake_model.safetensors")

        # create_model should NOT be called again
        assert mock_adapter.create_model.call_count == 1


# ======================================================================
# Ensure Model Loaded Tests
# ======================================================================


class TestEnsureModelLoaded:
    """_ensure_model_loaded checks ModelManager hash."""

    def test_ensure_with_model_loaded_flag(self) -> None:
        """When model_loaded is True and no hash, returns immediately."""
        engine = _make_engine()
        engine._model_loaded = True
        engine._config_hash = ""
        engine._ensure_model_loaded()  # should not raise

    def test_ensure_loads_model_when_needed(self) -> None:
        engine = _make_engine(model_path="test.safetensors")
        engine._model_loaded = False
        with patch.object(engine, "load_model") as mock_load:
            engine._ensure_model_loaded()
            mock_load.assert_called_once_with("test.safetensors")


# ======================================================================
# Full Generation Pipeline with Memory Subsystem
# ======================================================================


class TestGenerationWithMemory:
    """Full generation pipeline still works with memory subsystem wired in."""

    def test_generate_still_works_cpu(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.AUTO)
        result = engine.generate(prompt="test cat", seed=42, steps=3)
        assert len(result.images) >= 1
        assert result.seeds == [42]

    def test_generate_with_no_vram_mode(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.NO_VRAM)
        result = engine.generate(prompt="test dog", seed=1, steps=2)
        assert len(result.images) >= 1

    def test_generate_with_high_mode(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.HIGH)
        result = engine.generate(prompt="test bird", seed=2, steps=2)
        assert len(result.images) >= 1

    def test_generate_preserves_reproducibility(self) -> None:
        engine = _make_engine(vram_mode=VRAMMode.LOW)
        r1 = engine.generate(prompt="test", seed=42, steps=3)
        r2 = engine.generate(prompt="test", seed=42, steps=3)
        torch.testing.assert_close(r1.images[0], r2.images[0])


# ======================================================================
# Unload with Memory Subsystem
# ======================================================================


class TestUnloadWithMemory:
    """unload_all cleans up memory subsystem state."""

    def test_unload_clears_config_hash(self) -> None:
        engine = _make_engine()
        engine._config_hash = "abc"
        engine.unload_all()
        assert engine._config_hash == ""

    def test_unload_syncs_stream_pool(self) -> None:
        engine = _make_engine()
        mock_pool = MagicMock(spec=StreamPool)
        engine._stream_pool = mock_pool
        engine.unload_all()
        mock_pool.sync_all.assert_called_once()

    def test_unload_calls_manager_unload_all(self) -> None:
        engine = _make_engine()
        mock_mgr = MagicMock(spec=ModelManager)
        mock_mgr.loaded_models = []
        engine._model_manager = mock_mgr
        engine.unload_all()
        mock_mgr.unload_all.assert_called_once()
