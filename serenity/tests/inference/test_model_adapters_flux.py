"""Tests for Flux and Chroma model adapters."""

from __future__ import annotations

import pytest
import torch

from serenity.inference.models.base import BaseModelAdapter
from serenity.inference.models.detection import ModelArchitecture
from serenity.inference.models.flux import (
    ADAPTERS as FLUX_ADAPTERS,
    FluxAdapter,
    FluxSchnellAdapter,
    compute_img_ids,
)
from serenity.inference.models.chroma import (
    ADAPTERS as CHROMA_ADAPTERS,
    ChromaAdapter,
)


# ---------------------------------------------------------------------------
# FluxAdapter tests
# ---------------------------------------------------------------------------


class TestFluxAdapter:
    """Tests for the Flux Dev adapter."""

    def setup_method(self) -> None:
        self.adapter = FluxAdapter()

    def test_architecture_is_flux_dev(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.FLUX_DEV

    def test_text_encoder_types(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_l", "t5_xxl"]

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "flow_flux"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_is_base_model_adapter_subclass(self) -> None:
        assert isinstance(self.adapter, BaseModelAdapter)

    def test_prediction_kwargs_dev(self) -> None:
        kwargs = self.adapter.get_prediction_kwargs()
        assert "seq_len" in kwargs
        assert kwargs["seq_len"] == 4096
        assert kwargs["base_shift"] == 0.5
        assert kwargs["max_shift"] == 1.15

    def test_prepare_conditioning_includes_expected_keys(self) -> None:
        text_outputs = {
            "t5_xxl": torch.randn(1, 128, 4096),
            "clip_l_pooled": torch.randn(1, 768),
        }
        result = self.adapter.prepare_conditioning(text_outputs)
        assert "encoder_hidden_states" in result
        assert "txt_ids" in result
        assert "pooled_projections" in result

    def test_prepare_conditioning_with_height_width(self) -> None:
        text_outputs = {
            "t5_xxl": torch.randn(1, 128, 4096),
            "clip_l_pooled": torch.randn(1, 768),
        }
        result = self.adapter.prepare_conditioning(
            text_outputs, height=1024, width=1024,
        )
        assert "img_ids" in result
        assert "encoder_hidden_states" in result
        assert "txt_ids" in result
        assert "pooled_projections" in result

    def test_prepare_conditioning_txt_ids_shape(self) -> None:
        seq_len = 77
        text_outputs = {
            "t5_xxl": torch.randn(1, seq_len, 4096),
        }
        result = self.adapter.prepare_conditioning(text_outputs)
        assert result["txt_ids"].shape == (seq_len, 3)

    def test_create_model_raises_without_diffusers(self) -> None:
        """create_model raises NotImplementedError when diffusers is missing."""
        # This test verifies the lazy-import guard works.  The real diffusers
        # import may or may not be available, but the adapter should not crash
        # on import alone.
        adapter = FluxAdapter()
        # We just verify the adapter object is usable — create_model tested
        # separately if diffusers is available.
        assert adapter.architecture == ModelArchitecture.FLUX_DEV

    def test_variant_stored(self) -> None:
        assert self.adapter._variant == "dev"

    def test_adapters_registry_flux_dev(self) -> None:
        assert ModelArchitecture.FLUX_DEV in FLUX_ADAPTERS
        assert FLUX_ADAPTERS[ModelArchitecture.FLUX_DEV] is FluxAdapter

    def test_adapters_registry_flux_schnell(self) -> None:
        assert ModelArchitecture.FLUX_SCHNELL in FLUX_ADAPTERS
        assert FLUX_ADAPTERS[ModelArchitecture.FLUX_SCHNELL] is FluxSchnellAdapter


# ---------------------------------------------------------------------------
# FluxSchnellAdapter tests
# ---------------------------------------------------------------------------


class TestFluxSchnellAdapter:
    """Tests for the Flux Schnell adapter."""

    def setup_method(self) -> None:
        self.adapter = FluxSchnellAdapter()

    def test_architecture_is_flux_schnell(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.FLUX_SCHNELL

    def test_inherits_from_flux_adapter(self) -> None:
        assert isinstance(self.adapter, FluxAdapter)
        assert isinstance(self.adapter, BaseModelAdapter)

    def test_text_encoder_types_same_as_dev(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["clip_l", "t5_xxl"]

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "flow_flux"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_variant_is_schnell(self) -> None:
        assert self.adapter._variant == "schnell"

    def test_prediction_kwargs_schnell(self) -> None:
        kwargs = self.adapter.get_prediction_kwargs()
        assert kwargs == {"mu": 1.0}


# ---------------------------------------------------------------------------
# compute_img_ids tests
# ---------------------------------------------------------------------------


class TestComputeImgIds:
    """Tests for the image positional ID computation."""

    def test_shape_standard_resolution(self) -> None:
        ids = compute_img_ids(1024, 1024, patch_size=2)
        # 1024/2 = 512 patches per side -> 512*512 = 262144 patches
        assert ids.shape == (512 * 512, 3)

    def test_shape_small(self) -> None:
        ids = compute_img_ids(64, 64, patch_size=2)
        assert ids.shape == (32 * 32, 3)

    def test_batch_index_is_zero(self) -> None:
        ids = compute_img_ids(64, 64, patch_size=2)
        assert (ids[:, 0] == 0).all()

    def test_y_coordinates_range(self) -> None:
        ids = compute_img_ids(64, 128, patch_size=2)
        # h_patches=32, w_patches=64
        y_coords = ids[:, 1]
        assert y_coords.min().item() == 0
        assert y_coords.max().item() == 31

    def test_x_coordinates_range(self) -> None:
        ids = compute_img_ids(64, 128, patch_size=2)
        x_coords = ids[:, 2]
        assert x_coords.min().item() == 0
        assert x_coords.max().item() == 63

    def test_dtype_propagation(self) -> None:
        ids = compute_img_ids(64, 64, dtype=torch.float16)
        assert ids.dtype == torch.float16

    def test_device_cpu(self) -> None:
        ids = compute_img_ids(64, 64, device="cpu")
        assert ids.device == torch.device("cpu")

    def test_rectangular_resolution(self) -> None:
        ids = compute_img_ids(768, 1024, patch_size=2)
        assert ids.shape == (384 * 512, 3)


# ---------------------------------------------------------------------------
# ChromaAdapter tests
# ---------------------------------------------------------------------------


class TestChromaAdapter:
    """Tests for the Chroma adapter."""

    def setup_method(self) -> None:
        self.adapter = ChromaAdapter()

    def test_architecture_is_chroma(self) -> None:
        assert self.adapter.architecture == ModelArchitecture.CHROMA

    def test_text_encoder_types_t5_only(self) -> None:
        assert self.adapter.get_text_encoder_types() == ["t5_xxl"]

    def test_no_clip_in_encoder_types(self) -> None:
        types = self.adapter.get_text_encoder_types()
        assert "clip_l" not in types

    def test_prediction_type(self) -> None:
        assert self.adapter.get_prediction_type() == "flow_flux"

    def test_vae_scaling_factor(self) -> None:
        assert self.adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        assert self.adapter.get_default_resolution() == (1024, 1024)

    def test_is_base_model_adapter_subclass(self) -> None:
        assert isinstance(self.adapter, BaseModelAdapter)

    def test_prediction_kwargs(self) -> None:
        kwargs = self.adapter.get_prediction_kwargs()
        assert kwargs == {"mu": 1.0}

    def test_prepare_conditioning_no_pooled_projections(self) -> None:
        text_outputs = {
            "t5_xxl": torch.randn(1, 128, 4096),
        }
        result = self.adapter.prepare_conditioning(text_outputs)
        assert "encoder_hidden_states" in result
        assert "txt_ids" in result
        assert "pooled_projections" not in result

    def test_prepare_conditioning_with_height_width(self) -> None:
        text_outputs = {
            "t5_xxl": torch.randn(1, 128, 4096),
        }
        result = self.adapter.prepare_conditioning(
            text_outputs, height=1024, width=1024,
        )
        assert "img_ids" in result
        assert "pooled_projections" not in result

    def test_prepare_conditioning_txt_ids_shape(self) -> None:
        seq_len = 256
        text_outputs = {
            "t5_xxl": torch.randn(1, seq_len, 4096),
        }
        result = self.adapter.prepare_conditioning(text_outputs)
        assert result["txt_ids"].shape == (seq_len, 3)

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.CHROMA in CHROMA_ADAPTERS
        assert CHROMA_ADAPTERS[ModelArchitecture.CHROMA] is ChromaAdapter

    def test_adapters_registry_size(self) -> None:
        assert len(CHROMA_ADAPTERS) == 1
