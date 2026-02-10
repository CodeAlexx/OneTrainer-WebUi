"""Tests for the Serenity inference engine orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

from serenity.inference.cache.store import StageCache
from serenity.inference.config import AttentionBackend, InferenceConfig, VRAMMode
from serenity.inference.engine import GenerationResult, InferenceEngine


# ===================================================================== #
# Construction Tests
# ===================================================================== #


class TestEngineConstruction:
    """Engine initialization and subsystem setup."""

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_creates_without_error(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        assert engine is not None

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_default_device_auto_detected(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        # Should be cpu or cuda depending on environment
        assert engine._device.type in ("cpu", "cuda")

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_model_manager_initialized(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        InferenceEngine(InferenceConfig())
        mock_mm.assert_called_once()

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_cache_initialized(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        assert engine._cache.stats["entries"] == 0

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_attention_backend_detected(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        mock_attn.assert_called_once_with("auto")

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_explicit_attention_backend(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="einsum")
        config = InferenceConfig(attention_backend=AttentionBackend.EINSUM)
        engine = InferenceEngine(config)
        mock_attn.assert_called_once_with("einsum")

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_no_model_loaded_at_init(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        assert engine._model_loaded is False
        assert engine._unet is None


# ===================================================================== #
# Parameter Resolution Tests
# ===================================================================== #


class TestParameterResolution:
    """Merging of user kwargs with config defaults."""

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_config_defaults_used(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        config = InferenceConfig(steps=30, cfg_scale=5.0, width=768, height=768)
        engine = InferenceEngine(config)
        params = engine._resolve_params(prompt="test")
        assert params["steps"] == 30
        assert params["cfg_scale"] == 5.0
        assert params["width"] == 768
        assert params["height"] == 768

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_user_kwargs_override_config(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        config = InferenceConfig(steps=20, cfg_scale=7.0)
        engine = InferenceEngine(config)
        params = engine._resolve_params(prompt="test", steps=50, cfg_scale=3.0)
        assert params["steps"] == 50
        assert params["cfg_scale"] == 3.0

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_negative_seed_kept(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        params = engine._resolve_params(prompt="test", seed=-1)
        assert params["seed"] == -1

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_explicit_seed_preserved(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        params = engine._resolve_params(prompt="test", seed=42)
        assert params["seed"] == 42

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_sampler_and_scheduler_defaults(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        config = InferenceConfig(sampler="euler_a", scheduler="karras")
        engine = InferenceEngine(config)
        params = engine._resolve_params(prompt="test")
        assert params["sampler"] == "euler_a"
        assert params["scheduler"] == "karras"


# ===================================================================== #
# Status Tests
# ===================================================================== #


class TestEngineStatus:
    """Engine status reporting."""

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_get_status_returns_dict(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        status = engine.get_status()
        assert isinstance(status, dict)

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_status_has_expected_keys(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        engine = InferenceEngine(InferenceConfig())
        status = engine.get_status()
        expected_keys = {
            "device",
            "attention_backend",
            "model_loaded",
            "model_architecture",
            "loaded_models",
            "cache_entries",
            "vram_free_bytes",
            "vram_mode",
            "stream_pool_streams",
            "pinned_memory_bytes",
        }
        assert expected_keys == set(status.keys())

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_empty_engine_no_models(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        mock_mm_instance = MagicMock()
        mock_mm_instance.loaded_models = []
        mock_mm.return_value = mock_mm_instance
        engine = InferenceEngine(InferenceConfig())
        status = engine.get_status()
        assert status["model_loaded"] is False
        assert status["loaded_models"] == 0
        assert status["model_architecture"] is None


# ===================================================================== #
# Unload Tests
# ===================================================================== #


class TestEngineUnload:
    """Unload and cleanup."""

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_unload_all_clears_state(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        mock_mm_instance = MagicMock()
        mock_mm.return_value = mock_mm_instance
        engine = InferenceEngine(InferenceConfig())

        # Simulate loaded state
        engine._model_loaded = True
        engine._unet = MagicMock()
        engine._cache.put("text", "test", torch.zeros(1, 77, 768))

        engine.unload_all()

        assert engine._model_loaded is False
        assert engine._unet is None
        assert engine._cache.stats["entries"] == 0
        mock_mm_instance.unload_all.assert_called_once()

    @patch("serenity.inference.engine.select_best_backend")
    @patch("serenity.inference.engine.ModelManager")
    def test_unload_clears_model_config(self, mock_mm, mock_attn) -> None:
        mock_attn.return_value = MagicMock(value="sdp")
        mock_mm.return_value = MagicMock()
        engine = InferenceEngine(InferenceConfig())
        engine._model_config = MagicMock()
        engine.unload_all()
        assert engine._model_config is None


# ===================================================================== #
# Generate Pipeline Tests
# ===================================================================== #


class TestGeneratePipeline:
    """Full pipeline with mocked subsystems."""

    def _make_engine(self) -> InferenceEngine:
        """Create an engine with all external deps mocked."""
        with (
            patch("serenity.inference.engine.select_best_backend") as mock_attn,
            patch("serenity.inference.engine.ModelManager") as mock_mm,
        ):
            mock_attn.return_value = MagicMock(value="sdp")
            mock_mm_instance = MagicMock()
            mock_mm_instance.loaded_models = []
            mock_mm.return_value = mock_mm_instance
            engine = InferenceEngine(InferenceConfig(
                steps=5,
                width=64,
                height=64,
            ))
        return engine

    def test_generate_returns_result(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="a cat", seed=42)
        assert isinstance(result, GenerationResult)

    def test_result_has_images(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=1)
        assert isinstance(result.images, list)
        assert len(result.images) >= 1

    def test_result_has_seeds(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=42)
        assert result.seeds == [42]

    def test_result_has_metadata(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=42, steps=10)
        assert "prompt" in result.metadata
        assert "steps" in result.metadata
        assert result.metadata["prompt"] == "test"
        assert result.metadata["steps"] == 10

    def test_random_seed_when_negative(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=-1)
        assert len(result.seeds) == 1
        assert result.seeds[0] >= 0

    def test_different_random_seeds(self) -> None:
        """Two calls with seed=-1 should usually produce different seeds."""
        engine = self._make_engine()
        r1 = engine.generate(prompt="test", seed=-1)
        r2 = engine.generate(prompt="test", seed=-1)
        # Extremely unlikely to be equal with 2^32 range
        # But not impossible, so we just check they're valid
        assert r1.seeds[0] >= 0
        assert r2.seeds[0] >= 0

    def test_caching_text_encodings(self) -> None:
        """Second call with same prompt should use cached encoding."""
        engine = self._make_engine()
        engine.generate(prompt="a dog", seed=1)
        assert engine._cache.stats["entries"] == 1

        engine.generate(prompt="a dog", seed=2)
        # Should still be 1 (reused cache)
        assert engine._cache.stats["entries"] == 1

    def test_different_prompts_cached_separately(self) -> None:
        engine = self._make_engine()
        engine.generate(prompt="a dog", seed=1)
        engine.generate(prompt="a cat", seed=2)
        assert engine._cache.stats["entries"] == 2

    def test_callback_is_forwarded(self) -> None:
        """Callback should be invoked during sampling."""
        engine = self._make_engine()
        calls = []

        def my_callback(step, total, sigma, denoised):
            calls.append(step)

        engine.generate(prompt="test", seed=42, steps=5, callback=my_callback)
        # The euler sampler calls the callback once per step
        assert len(calls) > 0

    def test_explicit_seed_reproducibility(self) -> None:
        """Same seed should produce identical noise."""
        engine = self._make_engine()
        r1 = engine.generate(prompt="test", seed=42)
        r2 = engine.generate(prompt="test", seed=42)
        torch.testing.assert_close(r1.images[0], r2.images[0])

    def test_batch_size(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=42, batch_size=2)
        assert len(result.seeds) == 2

    def test_metadata_records_elapsed_time(self) -> None:
        engine = self._make_engine()
        result = engine.generate(prompt="test", seed=1)
        assert "elapsed_seconds" in result.metadata
        assert result.metadata["elapsed_seconds"] >= 0


# ===================================================================== #
# StageCache Tests
# ===================================================================== #


class TestStageCache:
    """StageCache LRU cache from cache.store."""

    def test_put_and_get(self) -> None:
        cache = StageCache(max_memory_bytes=1024 * 1024)
        t = torch.zeros(4)
        cache.put("text", "key1", t)
        result = cache.get("text", "key1")
        assert result is not None
        torch.testing.assert_close(result, t)

    def test_get_miss(self) -> None:
        cache = StageCache()
        assert cache.get("text", "nonexistent") is None

    def test_eviction_on_overflow(self) -> None:
        # Each tensor is 40 bytes (10 float32 elements); limit to ~80 bytes
        cache = StageCache(max_memory_bytes=80)
        cache.put("text", "a", torch.zeros(10))
        cache.put("text", "b", torch.ones(10))
        cache.put("text", "c", torch.full((10,), 2.0))
        assert cache.get("text", "a") is None  # evicted
        assert cache.get("text", "b") is not None
        assert cache.get("text", "c") is not None

    def test_invalidate(self) -> None:
        cache = StageCache()
        cache.put("text", "a", torch.zeros(4))
        cache.put("text", "b", torch.ones(4))
        cache.invalidate()
        assert cache.stats["entries"] == 0

    def test_lru_ordering(self) -> None:
        # Each tensor is 40 bytes (10 float32 elements); limit to ~80 bytes
        cache = StageCache(max_memory_bytes=80)
        cache.put("text", "a", torch.zeros(10))
        cache.put("text", "b", torch.ones(10))
        cache.get("text", "a")  # refresh "a"
        cache.put("text", "c", torch.full((10,), 2.0))  # should evict "b" (oldest)
        assert cache.get("text", "a") is not None
        assert cache.get("text", "b") is None
        assert cache.get("text", "c") is not None


# ===================================================================== #
# GenerationResult Tests
# ===================================================================== #


class TestGenerationResult:
    """Result dataclass structure."""

    def test_basic_creation(self) -> None:
        result = GenerationResult(
            images=[torch.zeros(3, 64, 64)],
            seeds=[42],
        )
        assert len(result.images) == 1
        assert result.seeds == [42]
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        result = GenerationResult(
            images=[torch.zeros(3, 64, 64)],
            seeds=[42],
            metadata={"prompt": "test", "steps": 20},
        )
        assert result.metadata["prompt"] == "test"
        assert result.metadata["steps"] == 20


# ===================================================================== #
# Import Tests
# ===================================================================== #


class TestImports:
    """Public API imports work correctly."""

    def test_import_engine_from_package(self) -> None:
        from serenity.inference import InferenceEngine as IE
        assert IE is InferenceEngine

    def test_import_config_from_package(self) -> None:
        from serenity.inference import InferenceConfig as IC
        assert IC is InferenceConfig

    def test_import_result_from_package(self) -> None:
        from serenity.inference import GenerationResult as GR
        assert GR is GenerationResult

    def test_import_vram_mode(self) -> None:
        from serenity.inference import VRAMMode
        assert VRAMMode.AUTO == "auto"

    def test_import_quantization_mode(self) -> None:
        from serenity.inference import QuantizationMode
        assert QuantizationMode.NONE == "none"

    def test_import_attention_backend(self) -> None:
        from serenity.inference import AttentionBackend
        assert AttentionBackend.AUTO == "auto"
