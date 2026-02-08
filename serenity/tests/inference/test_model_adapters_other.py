"""Tests for Wan, Lumina, ZImage, and Qwen model adapters."""

from __future__ import annotations

import pytest
import torch

from serenity.inference.models.detection import ModelArchitecture
from serenity.inference.models.wan import ADAPTERS as WAN_ADAPTERS
from serenity.inference.models.wan import WanAdapter
from serenity.inference.models.lumina import ADAPTERS as LUMINA_ADAPTERS
from serenity.inference.models.lumina import LuminaAdapter
from serenity.inference.models.zimage import ADAPTERS as ZIMAGE_ADAPTERS
from serenity.inference.models.zimage import ZImageAdapter
from serenity.inference.models.qwen import ADAPTERS as QWEN_ADAPTERS
from serenity.inference.models.qwen import QwenAdapter


# ---------------------------------------------------------------------------
# WanAdapter
# ---------------------------------------------------------------------------


class TestWanAdapter:
    """Tests for the Wan 2.1 adapter."""

    def test_architecture(self) -> None:
        adapter = WanAdapter()
        assert adapter.architecture == ModelArchitecture.WAN

    def test_text_encoder_types(self) -> None:
        adapter = WanAdapter()
        assert adapter.get_text_encoder_types() == ["t5_xxl"]

    def test_prediction_type(self) -> None:
        adapter = WanAdapter()
        assert adapter.get_prediction_type() == "flow"

    def test_vae_scaling_factor(self) -> None:
        adapter = WanAdapter()
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.13025)

    def test_default_resolution_t2v(self) -> None:
        adapter = WanAdapter(variant="t2v")
        assert adapter.get_default_resolution() == (480, 832)

    def test_default_resolution_i2v(self) -> None:
        adapter = WanAdapter(variant="i2v")
        assert adapter.get_default_resolution() == (480, 832)

    def test_variant_default(self) -> None:
        adapter = WanAdapter()
        assert adapter.variant == "t2v"

    def test_variant_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be 't2v' or 'i2v'"):
            WanAdapter(variant="invalid")

    def test_prepare_conditioning_basic(self) -> None:
        adapter = WanAdapter()
        cond = torch.randn(1, 77, 4096)
        result = adapter.prepare_conditioning({"cond": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_encoder_hidden_states_key(self) -> None:
        adapter = WanAdapter()
        cond = torch.randn(1, 77, 4096)
        result = adapter.prepare_conditioning({"encoder_hidden_states": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_i2v_with_image_latents(self) -> None:
        adapter = WanAdapter(variant="i2v")
        cond = torch.randn(1, 77, 4096)
        img_latents = torch.randn(1, 16, 4, 60, 104)
        result = adapter.prepare_conditioning(
            {"cond": cond}, image_latents=img_latents
        )
        assert "encoder_hidden_states" in result
        assert "image_latents" in result
        assert torch.equal(result["image_latents"], img_latents)

    def test_prepare_conditioning_t2v_ignores_image_latents(self) -> None:
        adapter = WanAdapter(variant="t2v")
        cond = torch.randn(1, 77, 4096)
        img_latents = torch.randn(1, 16, 4, 60, 104)
        result = adapter.prepare_conditioning(
            {"cond": cond}, image_latents=img_latents
        )
        assert "encoder_hidden_states" in result
        assert "image_latents" not in result

    def test_prepare_conditioning_passthrough(self) -> None:
        adapter = WanAdapter()
        inputs = {"other_key": torch.randn(1)}
        result = adapter.prepare_conditioning(inputs)
        assert result is inputs

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.WAN in WAN_ADAPTERS
        assert WAN_ADAPTERS[ModelArchitecture.WAN] is WanAdapter


# ---------------------------------------------------------------------------
# LuminaAdapter
# ---------------------------------------------------------------------------


class TestLuminaAdapter:
    """Tests for the Lumina 2 adapter."""

    def test_architecture(self) -> None:
        adapter = LuminaAdapter()
        assert adapter.architecture == ModelArchitecture.LUMINA

    def test_text_encoder_types(self) -> None:
        adapter = LuminaAdapter()
        assert adapter.get_text_encoder_types() == ["gemma"]

    def test_prediction_type(self) -> None:
        adapter = LuminaAdapter()
        assert adapter.get_prediction_type() == "flow"

    def test_vae_scaling_factor(self) -> None:
        adapter = LuminaAdapter()
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        adapter = LuminaAdapter()
        assert adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_basic(self) -> None:
        adapter = LuminaAdapter()
        cond = torch.randn(1, 77, 2304)
        result = adapter.prepare_conditioning({"cond": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_encoder_hidden_states_key(self) -> None:
        adapter = LuminaAdapter()
        cond = torch.randn(1, 77, 2304)
        result = adapter.prepare_conditioning({"encoder_hidden_states": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_passthrough(self) -> None:
        adapter = LuminaAdapter()
        inputs = {"other_key": torch.randn(1)}
        result = adapter.prepare_conditioning(inputs)
        assert result is inputs

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.LUMINA in LUMINA_ADAPTERS
        assert LUMINA_ADAPTERS[ModelArchitecture.LUMINA] is LuminaAdapter


# ---------------------------------------------------------------------------
# ZImageAdapter
# ---------------------------------------------------------------------------


class TestZImageAdapter:
    """Tests for the Z-Image adapter."""

    def test_architecture(self) -> None:
        adapter = ZImageAdapter()
        assert adapter.architecture == ModelArchitecture.ZIMAGE

    def test_text_encoder_types(self) -> None:
        adapter = ZImageAdapter()
        assert adapter.get_text_encoder_types() == ["gemma"]

    def test_prediction_type(self) -> None:
        adapter = ZImageAdapter()
        assert adapter.get_prediction_type() == "flow"

    def test_vae_scaling_factor(self) -> None:
        adapter = ZImageAdapter()
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        adapter = ZImageAdapter()
        assert adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_basic(self) -> None:
        adapter = ZImageAdapter()
        cond = torch.randn(1, 77, 3840)
        result = adapter.prepare_conditioning({"cond": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_encoder_hidden_states_key(self) -> None:
        adapter = ZImageAdapter()
        cond = torch.randn(1, 77, 3840)
        result = adapter.prepare_conditioning({"encoder_hidden_states": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_passthrough(self) -> None:
        adapter = ZImageAdapter()
        inputs = {"other_key": torch.randn(1)}
        result = adapter.prepare_conditioning(inputs)
        assert result is inputs

    def test_distinct_from_lumina(self) -> None:
        """Lumina and ZImage are structurally similar but distinct architectures."""
        lumina = LuminaAdapter()
        zimage = ZImageAdapter()
        assert lumina.architecture != zimage.architecture
        assert lumina.architecture == ModelArchitecture.LUMINA
        assert zimage.architecture == ModelArchitecture.ZIMAGE

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.ZIMAGE in ZIMAGE_ADAPTERS
        assert ZIMAGE_ADAPTERS[ModelArchitecture.ZIMAGE] is ZImageAdapter


# ---------------------------------------------------------------------------
# QwenAdapter
# ---------------------------------------------------------------------------


class TestQwenAdapter:
    """Tests for the Qwen Image adapter."""

    def test_architecture(self) -> None:
        adapter = QwenAdapter()
        assert adapter.architecture == ModelArchitecture.QWEN

    def test_text_encoder_types(self) -> None:
        adapter = QwenAdapter()
        assert adapter.get_text_encoder_types() == ["qwen"]

    def test_prediction_type(self) -> None:
        adapter = QwenAdapter()
        assert adapter.get_prediction_type() == "flow"

    def test_vae_scaling_factor(self) -> None:
        adapter = QwenAdapter()
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.3611)

    def test_default_resolution(self) -> None:
        adapter = QwenAdapter()
        assert adapter.get_default_resolution() == (1024, 1024)

    def test_prepare_conditioning_basic(self) -> None:
        adapter = QwenAdapter()
        cond = torch.randn(1, 77, 3584)
        result = adapter.prepare_conditioning({"cond": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_encoder_hidden_states_key(self) -> None:
        adapter = QwenAdapter()
        cond = torch.randn(1, 77, 3584)
        result = adapter.prepare_conditioning({"encoder_hidden_states": cond})
        assert "encoder_hidden_states" in result
        assert torch.equal(result["encoder_hidden_states"], cond)

    def test_prepare_conditioning_with_ref_latents(self) -> None:
        adapter = QwenAdapter()
        cond = torch.randn(1, 77, 3584)
        ref = torch.randn(1, 16, 64, 64)
        result = adapter.prepare_conditioning({"cond": cond}, ref_latents=ref)
        assert "encoder_hidden_states" in result
        assert "ref_latents" in result
        assert torch.equal(result["ref_latents"], ref)

    def test_prepare_conditioning_passthrough(self) -> None:
        adapter = QwenAdapter()
        inputs = {"other_key": torch.randn(1)}
        result = adapter.prepare_conditioning(inputs)
        assert result is inputs

    def test_adapters_registry(self) -> None:
        assert ModelArchitecture.QWEN in QWEN_ADAPTERS
        assert QWEN_ADAPTERS[ModelArchitecture.QWEN] is QwenAdapter
