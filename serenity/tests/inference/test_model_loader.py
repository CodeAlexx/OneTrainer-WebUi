"""Tests for model loading utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from serenity.inference.models.base import BaseModelAdapter, ModelAdapter
from serenity.inference.models.detection import ModelArchitecture, ModelConfig
from serenity.inference.models.loader import (
    extract_vae_state_dict,
    load_model,
    load_state_dict,
)


# ---------------------------------------------------------------------------
# load_state_dict tests
# ---------------------------------------------------------------------------


class TestLoadStateDict:
    """Tests for loading state dicts from files."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_state_dict("/nonexistent/model.safetensors")

    def test_safetensors_file(self, tmp_path) -> None:
        """Create a real safetensors file, then load it."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors not installed")

        model_file = tmp_path / "model.safetensors"
        tensors = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        }
        save_file(tensors, str(model_file))

        sd = load_state_dict(str(model_file))
        assert "weight" in sd
        assert "bias" in sd
        assert torch.equal(sd["weight"], tensors["weight"])
        assert torch.equal(sd["bias"], tensors["bias"])

    def test_pt_file(self, tmp_path) -> None:
        """Create a .pt file and load it."""
        model_file = tmp_path / "model.pt"
        tensors = {"layer.weight": torch.randn(3, 3)}
        torch.save(tensors, str(model_file))

        sd = load_state_dict(str(model_file))
        assert "layer.weight" in sd

    def test_nested_state_dict(self, tmp_path) -> None:
        """Handle checkpoints with a nested 'state_dict' key."""
        model_file = tmp_path / "model.pt"
        inner = {"layer.weight": torch.randn(3, 3)}
        torch.save({"state_dict": inner, "epoch": 10}, str(model_file))

        sd = load_state_dict(str(model_file))
        assert "layer.weight" in sd
        assert "epoch" not in sd


# ---------------------------------------------------------------------------
# extract_vae_state_dict tests
# ---------------------------------------------------------------------------


class TestExtractVAEStateDict:
    """Tests for extracting VAE keys from full checkpoints."""

    def test_first_stage_model_prefix(self) -> None:
        full_sd = {
            "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
            "first_stage_model.decoder.conv_out.weight": torch.randn(3, 128, 3, 3),
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3),
        }
        vae_sd = extract_vae_state_dict(full_sd)
        assert "encoder.conv_in.weight" in vae_sd
        assert "decoder.conv_out.weight" in vae_sd
        assert len(vae_sd) == 2

    def test_vae_prefix(self) -> None:
        full_sd = {
            "vae.encoder.weight": torch.randn(4, 4),
            "vae.decoder.weight": torch.randn(4, 4),
            "model.unet.weight": torch.randn(4, 4),
        }
        vae_sd = extract_vae_state_dict(full_sd)
        assert "encoder.weight" in vae_sd
        assert "decoder.weight" in vae_sd
        assert len(vae_sd) == 2

    def test_no_vae_keys_returns_empty(self) -> None:
        full_sd = {
            "model.weight": torch.randn(4, 4),
        }
        vae_sd = extract_vae_state_dict(full_sd)
        assert len(vae_sd) == 0

    def test_mixed_prefixes(self) -> None:
        """If both prefixes exist, both should be extracted."""
        full_sd = {
            "first_stage_model.encoder.weight": torch.randn(4),
            "vae.decoder.weight": torch.randn(4),
            "unet.weight": torch.randn(4),
        }
        vae_sd = extract_vae_state_dict(full_sd)
        assert "encoder.weight" in vae_sd
        assert "decoder.weight" in vae_sd
        assert len(vae_sd) == 2


# ---------------------------------------------------------------------------
# ModelAdapter / BaseModelAdapter tests
# ---------------------------------------------------------------------------


class TestModelAdapter:
    """Tests for the abstract model adapter interface."""

    def test_base_adapter_defaults(self) -> None:
        adapter = BaseModelAdapter()
        assert adapter.architecture == ModelArchitecture.SD15
        assert adapter.get_text_encoder_types() == ["clip_l"]
        assert adapter.get_prediction_type() == "eps"
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.18215)
        assert adapter.get_default_resolution() == (512, 512)

    def test_base_adapter_prepare_conditioning_passthrough(self) -> None:
        adapter = BaseModelAdapter()
        inputs = {"clip_l": torch.randn(1, 77, 768)}
        result = adapter.prepare_conditioning(inputs)
        assert "clip_l" in result
        assert torch.equal(result["clip_l"], inputs["clip_l"])

    def test_base_adapter_create_model_raises(self) -> None:
        adapter = BaseModelAdapter()
        with pytest.raises(NotImplementedError):
            adapter.create_model({})

    def test_concrete_subclass(self) -> None:
        """A concrete adapter can override all abstract methods."""

        class FluxAdapter(ModelAdapter):
            @property
            def architecture(self) -> ModelArchitecture:
                return ModelArchitecture.FLUX_DEV

            def create_model(self, state_dict, device="cpu", dtype=torch.float32, **kw):
                return MagicMock(spec=torch.nn.Module)

            def get_text_encoder_types(self) -> list[str]:
                return ["clip_l", "t5_xxl"]

            def get_prediction_type(self) -> str:
                return "flow"

            def get_vae_scaling_factor(self) -> float:
                return 0.3611

            def get_default_resolution(self) -> tuple[int, int]:
                return (1024, 1024)

            def prepare_conditioning(self, text_outputs, **kw):
                return text_outputs

        adapter = FluxAdapter()
        assert adapter.architecture == ModelArchitecture.FLUX_DEV
        assert adapter.get_prediction_type() == "flow"
        assert adapter.get_vae_scaling_factor() == pytest.approx(0.3611)
        assert adapter.get_default_resolution() == (1024, 1024)
        assert adapter.get_text_encoder_types() == ["clip_l", "t5_xxl"]


# ---------------------------------------------------------------------------
# load_model tests (mocked)
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model with mocked dependencies."""

    def test_load_model_adapter_found(self, tmp_path) -> None:
        """load_model finds the SD15 adapter and returns a model."""
        model_file = tmp_path / "model.pt"
        sd = {k: torch.zeros(1) for k in [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.0.0.bias",
        ]}
        # Add enough keys for prefix detection
        for i in range(10):
            sd[f"model.diffusion_model.extra_{i}"] = torch.zeros(1)
        # Add keys for SD15 detection
        for i in range(1, 12):
            sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.0.weight"] = torch.zeros(1)
        for i in range(12):
            sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.0.weight"] = torch.zeros(1)
        torch.save(sd, str(model_file))

        # Adapter registry is wired up — SD15 adapter found, model loaded (strict=False)
        model = load_model(str(model_file))
        assert model is not None

    def test_load_model_with_explicit_config(self, tmp_path) -> None:
        """When config is provided, detection is skipped."""
        model_file = tmp_path / "model.pt"
        sd = {"some.key": torch.zeros(1)}
        torch.save(sd, str(model_file))

        config = ModelConfig(
            architecture=ModelArchitecture.SD15,
            prediction_type="eps",
        )

        # Adapter is found via explicit config, model loaded (strict=False)
        model = load_model(str(model_file), config=config)
        assert model is not None

    def test_load_model_unrecognised_architecture(self, tmp_path) -> None:
        """Unrecognised state dict raises ValueError."""
        model_file = tmp_path / "model.pt"
        sd = {"random.key.1": torch.zeros(1), "random.key.2": torch.zeros(1)}
        torch.save(sd, str(model_file))

        with pytest.raises(ValueError, match="Cannot detect"):
            load_model(str(model_file))
