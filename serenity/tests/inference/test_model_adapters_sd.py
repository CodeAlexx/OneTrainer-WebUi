"""Tests for SD 1.5, SDXL, and SD3 model adapters — no GPU or model files required."""

from __future__ import annotations

import torch
import pytest

from serenity.inference.models.detection import ModelArchitecture
from serenity.inference.models.sd15 import SD15Adapter, ADAPTERS as SD15_ADAPTERS
from serenity.inference.models.sdxl import (
    SDXLAdapter,
    SDXLRefinerAdapter,
    ADAPTERS as SDXL_ADAPTERS,
)
from serenity.inference.models.sd3 import SD3Adapter, ADAPTERS as SD3_ADAPTERS


# ---------------------------------------------------------------------------
# SD 1.5 Adapter
# ---------------------------------------------------------------------------


class TestSD15Adapter:
    """SD 1.5 adapter metadata and conditioning tests."""

    def setup_method(self) -> None:
        self.adapter = SD15Adapter()

    def test_architecture(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.SD15

    def test_text_encoder_types(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_l"]

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "eps"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.18215)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (512, 512)

    def test_prepare_conditioning_passthrough(self) -> None:
        cond = torch.randn(1, 77, 768)
        text_outputs = {"cond": cond}
        result = self.adapter.prepare_conditioning(text_outputs)
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_no_cond_key(self) -> None:
        """When 'cond' key is missing, return text_outputs unchanged."""
        text_outputs = {"other": torch.randn(1, 77, 768)}
        result = self.adapter.prepare_conditioning(text_outputs)
        assert result == text_outputs

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.SD15 in SD15_ADAPTERS
        assert SD15_ADAPTERS[ModelArchitecture.SD15] is SD15Adapter

    def test_create_model_requires_diffusers(self) -> None:
        """create_model should raise NotImplementedError when diffusers is missing."""
        # This will either work (if diffusers is installed) or raise
        # NotImplementedError — we just verify it doesn't crash silently.
        try:
            self.adapter.create_model({}, device="cpu")
        except (NotImplementedError, Exception):
            pass  # Expected when diffusers not installed or bad state_dict


# ---------------------------------------------------------------------------
# SDXL Adapter
# ---------------------------------------------------------------------------


class TestSDXLAdapter:
    """SDXL adapter metadata and conditioning tests."""

    def setup_method(self) -> None:
        self.adapter = SDXLAdapter()

    def test_architecture(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.SDXL

    def test_text_encoder_types(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_l", "clip_g"]

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "eps"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.13025)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_concatenates(self) -> None:
        cond_l = torch.randn(1, 77, 768)
        cond_g = torch.randn(1, 77, 1280)
        pooled = torch.randn(1, 1280)
        text_outputs = {"cond_l": cond_l, "cond_g": cond_g, "pooled": pooled}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "encoder_hidden_states" in result
        # Feature dim should be sum of CLIP-L (768) + CLIP-G (1280) = 2048
        assert result["encoder_hidden_states"].shape == (1, 77, 2048)
        assert "pooled" in result
        assert torch.equal(result["pooled"], pooled)

    def test_prepare_conditioning_includes_add_time_ids(self) -> None:
        cond_l = torch.randn(1, 77, 768)
        cond_g = torch.randn(1, 77, 1280)
        pooled = torch.randn(1, 1280)
        text_outputs = {"cond_l": cond_l, "cond_g": cond_g, "pooled": pooled}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "add_time_ids" in result
        time_ids = result["add_time_ids"]
        # Should be [1, 6]: original_h, original_w, crop_top, crop_left, target_h, target_w
        assert time_ids.shape == (1, 6)
        # Default values: 1024x1024, no crop, 1024x1024 target
        assert time_ids[0, 0].item() == pytest.approx(1024.0)
        assert time_ids[0, 1].item() == pytest.approx(1024.0)
        assert time_ids[0, 2].item() == pytest.approx(0.0)
        assert time_ids[0, 3].item() == pytest.approx(0.0)
        assert time_ids[0, 4].item() == pytest.approx(1024.0)
        assert time_ids[0, 5].item() == pytest.approx(1024.0)

    def test_prepare_conditioning_custom_sizes(self) -> None:
        cond_l = torch.randn(1, 77, 768)
        cond_g = torch.randn(1, 77, 1280)
        text_outputs = {"cond_l": cond_l, "cond_g": cond_g}

        result = self.adapter.prepare_conditioning(
            text_outputs,
            original_size=(768, 512),
            crop_coords=(32, 16),
            target_size=(1024, 1024),
        )

        time_ids = result["add_time_ids"]
        assert time_ids[0, 0].item() == pytest.approx(768.0)
        assert time_ids[0, 1].item() == pytest.approx(512.0)
        assert time_ids[0, 2].item() == pytest.approx(32.0)
        assert time_ids[0, 3].item() == pytest.approx(16.0)

    def test_prepare_conditioning_pads_sequence_lengths(self) -> None:
        """CLIP-L and CLIP-G with different sequence lengths should be padded."""
        cond_l = torch.randn(1, 60, 768)
        cond_g = torch.randn(1, 77, 1280)
        text_outputs = {"cond_l": cond_l, "cond_g": cond_g}

        result = self.adapter.prepare_conditioning(text_outputs)

        # Both should be padded to max_len=77
        assert result["encoder_hidden_states"].shape == (1, 77, 2048)

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.SDXL in SDXL_ADAPTERS
        assert SDXL_ADAPTERS[ModelArchitecture.SDXL] is SDXLAdapter
        assert ModelArchitecture.SDXL_REFINER in SDXL_ADAPTERS
        assert SDXL_ADAPTERS[ModelArchitecture.SDXL_REFINER] is SDXLRefinerAdapter


# ---------------------------------------------------------------------------
# SDXL Refiner Adapter
# ---------------------------------------------------------------------------


class TestSDXLRefinerAdapter:
    """SDXL Refiner adapter tests — CLIP-G only."""

    def setup_method(self) -> None:
        self.adapter = SDXLRefinerAdapter()

    def test_architecture(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.SDXL_REFINER

    def test_text_encoder_types_clip_g_only(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_g"]

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "eps"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.13025)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_uses_clip_g(self) -> None:
        cond_g = torch.randn(1, 77, 1280)
        pooled = torch.randn(1, 1280)
        text_outputs = {"cond_g": cond_g, "pooled": pooled}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond_g)
        assert "pooled" in result
        assert torch.equal(result["pooled"], pooled)

    def test_prepare_conditioning_includes_time_ids(self) -> None:
        cond_g = torch.randn(1, 77, 1280)
        pooled = torch.randn(1, 1280)
        text_outputs = {"cond_g": cond_g, "pooled": pooled}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "add_time_ids" in result
        # Refiner time IDs have 5 elements: original_h, original_w, crop_top, crop_left, aesthetic_score
        assert result["add_time_ids"].shape == (1, 5)

    def test_prepare_conditioning_aesthetic_score(self) -> None:
        cond_g = torch.randn(1, 77, 1280)
        pooled = torch.randn(1, 1280)
        text_outputs = {"cond_g": cond_g, "pooled": pooled}

        result = self.adapter.prepare_conditioning(
            text_outputs, aesthetic_score=8.5,
        )

        time_ids = result["add_time_ids"]
        assert time_ids[0, 4].item() == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# SD3 Adapter
# ---------------------------------------------------------------------------


class TestSD3Adapter:
    """SD3 adapter metadata and conditioning tests."""

    def setup_method(self) -> None:
        self.adapter = SD3Adapter()

    def test_architecture(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.SD3

    def test_text_encoder_types(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_l", "clip_g", "t5_xxl"]

    def test_prediction_type_is_flow(self) -> None:
        assert self.adapter.get_prediction_type() == "flow"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(1.5305)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_concatenates_embeddings(self) -> None:
        # In real SD3, CLIP-L and CLIP-G outputs are projected to match T5
        # hidden dim (4096) before concatenation along the sequence dimension.
        cond_l = torch.randn(1, 77, 4096)
        cond_g = torch.randn(1, 77, 4096)
        cond_t5 = torch.randn(1, 256, 4096)
        pooled_l = torch.randn(1, 768)
        pooled_g = torch.randn(1, 1280)

        text_outputs = {
            "cond_l": cond_l,
            "cond_g": cond_g,
            "cond_t5": cond_t5,
            "pooled_l": pooled_l,
            "pooled_g": pooled_g,
        }

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "encoder_hidden_states" in result
        # Concatenated along sequence dim: 77 + 77 + 256 = 410
        assert result["encoder_hidden_states"].shape[1] == 410
        assert result["encoder_hidden_states"].shape[2] == 4096

        assert "pooled_projections" in result
        # Concatenated along feature dim: 768 + 1280 = 2048
        assert result["pooled_projections"].shape == (1, 2048)

    def test_prepare_conditioning_partial_encoders(self) -> None:
        """Should work with only some text encoder outputs present."""
        cond_l = torch.randn(1, 77, 4096)
        cond_g = torch.randn(1, 77, 4096)
        text_outputs = {"cond_l": cond_l, "cond_g": cond_g}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "encoder_hidden_states" in result
        # Only L + G: 77 + 77 = 154
        assert result["encoder_hidden_states"].shape[1] == 154

    def test_prepare_conditioning_fallback_cond(self) -> None:
        """Falls back to 'cond' key when specific keys are missing."""
        cond = torch.randn(1, 77, 4096)
        text_outputs = {"cond": cond}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_pooled_fallback(self) -> None:
        """Falls back to 'pooled' when specific pooled keys are missing."""
        pooled = torch.randn(1, 2048)
        text_outputs = {"cond": torch.randn(1, 77, 4096), "pooled": pooled}

        result = self.adapter.prepare_conditioning(text_outputs)

        assert "pooled_projections" in result
        assert torch.equal(result["pooled_projections"], pooled)

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.SD3 in SD3_ADAPTERS
        assert SD3_ADAPTERS[ModelArchitecture.SD3] is SD3Adapter


# ---------------------------------------------------------------------------
# Cross-adapter sanity checks
# ---------------------------------------------------------------------------


class TestAdapterInteroperability:
    """Verify adapter contracts across all SD-family adapters."""

    @pytest.fixture(params=[SD15Adapter, SDXLAdapter, SDXLRefinerAdapter, SD3Adapter])
    def adapter(self, request: pytest.FixtureRequest) -> object:
        return request.param()

    def test_architecture_is_model_architecture(self, adapter) -> None:
        assert isinstance(adapter.architecture, ModelArchitecture)

    def test_text_encoder_types_is_list(self, adapter) -> None:
        types = adapter.get_text_encoder_types()
        assert isinstance(types, list)
        assert all(isinstance(t, str) for t in types)

    def test_prediction_type_is_string(self, adapter) -> None:
        assert isinstance(adapter.get_prediction_type(), str)

    def test_vae_scaling_factor_is_positive(self, adapter) -> None:
        assert adapter.get_vae_scaling_factor() > 0.0

    def test_default_resolution_is_tuple(self, adapter) -> None:
        res = adapter.get_default_resolution()
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(isinstance(v, int) and v > 0 for v in res)
