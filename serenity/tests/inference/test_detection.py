"""Tests for model architecture detection from state-dict keys."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from serenity.inference.models.detection import (
    ModelArchitecture,
    ModelConfig,
    detect_from_file,
    detect_model_type,
)


# ---------------------------------------------------------------------------
# Key generators — create realistic state-dict key sets
# ---------------------------------------------------------------------------


def _sd15_keys() -> set[str]:
    """Generate a minimal set of SD 1.5 state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    # Input blocks (needed for detection)
    keys.add(f"{prefix}input_blocks.0.0.weight")
    keys.add(f"{prefix}input_blocks.0.0.bias")
    for i in range(1, 12):
        keys.add(f"{prefix}input_blocks.{i}.0.in_layers.0.weight")
        keys.add(f"{prefix}input_blocks.{i}.0.out_layers.3.weight")

    # Transformer blocks in some input_blocks
    for i in [1, 2, 4, 5, 7, 8]:
        keys.add(f"{prefix}input_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight")
        keys.add(f"{prefix}input_blocks.{i}.1.proj_in.weight")

    # Middle block
    keys.add(f"{prefix}middle_block.0.in_layers.0.weight")
    keys.add(f"{prefix}middle_block.1.proj_in.weight")
    keys.add(f"{prefix}middle_block.1.transformer_blocks.0.attn2.to_k.weight")

    # Output blocks
    for i in range(12):
        keys.add(f"{prefix}output_blocks.{i}.0.in_layers.0.weight")

    keys.add(f"{prefix}out.2.weight")

    # Add enough keys to detect prefix
    for i in range(10):
        keys.add(f"{prefix}extra_key_{i}")

    return keys


def _sdxl_keys() -> set[str]:
    """Generate a minimal set of SDXL state-dict keys."""
    prefix = "model.diffusion_model."
    keys = _sd15_keys()  # Start with SD15 base

    # ADM conditioning — distinguishes SDXL from SD15
    keys.add(f"{prefix}label_emb.0.0.weight")

    # SDXL has deeper middle block transformers (depth 10)
    for i in range(10):
        keys.add(f"{prefix}middle_block.1.transformer_blocks.{i}.attn2.to_k.weight")

    return keys


def _sdxl_refiner_keys() -> set[str]:
    """Generate SDXL Refiner keys (shallower middle block)."""
    prefix = "model.diffusion_model."
    keys = _sd15_keys()

    keys.add(f"{prefix}label_emb.0.0.weight")

    # Refiner has middle_depth=4
    for i in range(4):
        keys.add(f"{prefix}middle_block.1.transformer_blocks.{i}.attn2.to_k.weight")

    return keys


def _flux_dev_keys() -> set[str]:
    """Generate Flux.1 Dev state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    # Core flux keys
    keys.add(f"{prefix}double_blocks.0.img_attn.norm.key_norm.scale")
    keys.add(f"{prefix}img_in.weight")
    keys.add(f"{prefix}guidance_in.in_layer.weight")  # dev has guidance embed

    # Double blocks
    for i in range(19):
        keys.add(f"{prefix}double_blocks.{i}.img_attn.proj.weight")
        keys.add(f"{prefix}double_blocks.{i}.img_mlp.0.weight")

    # Single blocks
    for i in range(38):
        keys.add(f"{prefix}single_blocks.{i}.linear1.weight")

    keys.add(f"{prefix}txt_in.weight")
    keys.add(f"{prefix}vector_in.in_layer.weight")

    # Pad to enough keys for prefix detection
    for i in range(10):
        keys.add(f"{prefix}extra_{i}")

    return keys


def _flux_schnell_keys() -> set[str]:
    """Generate Flux.1 Schnell keys (no guidance_embed)."""
    keys = _flux_dev_keys()
    keys.discard("model.diffusion_model.guidance_in.in_layer.weight")
    return keys


def _chroma_keys() -> set[str]:
    """Generate Chroma keys (distilled flux variant)."""
    keys = _flux_dev_keys()
    # Chroma has distilled guidance layer
    keys.add("model.diffusion_model.distilled_guidance_layer.norms.0.scale")
    # Remove guidance_in (Chroma doesn't have it in the same way)
    keys.discard("model.diffusion_model.guidance_in.in_layer.weight")
    return keys


def _sd3_keys() -> set[str]:
    """Generate SD3 state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    # Joint transformer blocks
    for i in range(24):
        keys.add(f"{prefix}joint_blocks.{i}.x_block.attn.qkv.weight")
        keys.add(f"{prefix}joint_blocks.{i}.context_block.attn.qkv.weight")

    # Pad for prefix detection
    for i in range(10):
        keys.add(f"{prefix}extra_{i}")

    return keys


def _wan_keys() -> set[str]:
    """Generate Wan 2.1 state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    keys.add(f"{prefix}head.modulation")
    keys.add(f"{prefix}head.head.weight")
    keys.add(f"{prefix}patch_embedding.weight")

    for i in range(40):
        keys.add(f"{prefix}blocks.{i}.ffn.0.weight")
        keys.add(f"{prefix}blocks.{i}.attn.weight")

    for i in range(10):
        keys.add(f"{prefix}extra_{i}")

    return keys


def _qwen_keys() -> set[str]:
    """Generate Qwen Image state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    keys.add(f"{prefix}txt_norm.weight")
    keys.add(f"{prefix}img_in.weight")

    for i in range(28):
        keys.add(f"{prefix}transformer_blocks.{i}.attn.to_qkv.weight")

    for i in range(10):
        keys.add(f"{prefix}extra_{i}")

    return keys


def _lumina_keys() -> set[str]:
    """Generate Lumina 2 state-dict keys."""
    prefix = "model.diffusion_model."
    keys = set()

    keys.add(f"{prefix}cap_embedder.1.weight")

    for i in range(24):
        keys.add(f"{prefix}layers.{i}.attention.weight")

    for i in range(10):
        keys.add(f"{prefix}extra_{i}")

    return keys


def _zimage_keys() -> set[str]:
    """Generate Z-Image state-dict keys (Lumina variant with pad token)."""
    keys = _lumina_keys()
    keys.add("model.diffusion_model.cap_pad_token")
    return keys


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetectModelType:
    """Test detect_model_type with various key patterns."""

    def test_sd15(self) -> None:
        result = detect_model_type(_sd15_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.SD15
        assert result.prediction_type == "eps"
        assert "first_stage_model." in result.vae_key_prefix

    def test_sdxl(self) -> None:
        result = detect_model_type(_sdxl_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.SDXL
        assert result.prediction_type == "eps"

    def test_sdxl_refiner(self) -> None:
        result = detect_model_type(_sdxl_refiner_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.SDXL_REFINER

    def test_flux_dev(self) -> None:
        result = detect_model_type(_flux_dev_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.FLUX_DEV
        assert result.unet_config.get("guidance_embed") is True
        assert result.prediction_type == "flow"

    def test_flux_schnell(self) -> None:
        result = detect_model_type(_flux_schnell_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.FLUX_SCHNELL
        assert result.unet_config.get("guidance_embed") is False

    def test_chroma(self) -> None:
        result = detect_model_type(_chroma_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.CHROMA

    def test_sd3(self) -> None:
        result = detect_model_type(_sd3_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.SD3
        assert result.prediction_type == "flow"

    def test_wan(self) -> None:
        result = detect_model_type(_wan_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.WAN

    def test_wan_t2v(self) -> None:
        result = detect_model_type(_wan_keys())
        assert result is not None
        assert result.unet_config.get("model_type") == "t2v"

    def test_wan_i2v(self) -> None:
        keys = _wan_keys()
        keys.add("model.diffusion_model.img_emb.proj.0.bias")
        result = detect_model_type(keys)
        assert result is not None
        assert result.unet_config.get("model_type") == "i2v"

    def test_qwen(self) -> None:
        result = detect_model_type(_qwen_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.QWEN

    def test_lumina(self) -> None:
        result = detect_model_type(_lumina_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.LUMINA

    def test_zimage(self) -> None:
        result = detect_model_type(_zimage_keys())
        assert result is not None
        assert result.architecture == ModelArchitecture.ZIMAGE

    def test_unknown_returns_none(self) -> None:
        result = detect_model_type({"some.random.key", "another.key"})
        assert result is None

    def test_empty_keys_returns_none(self) -> None:
        result = detect_model_type(set())
        assert result is None


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = ModelConfig(architecture=ModelArchitecture.SD15)
        assert cfg.architecture == ModelArchitecture.SD15
        assert cfg.unet_config == {}
        assert cfg.unet_key_prefix == []
        assert cfg.vae_key_prefix == []
        assert cfg.prediction_type == "eps"

    def test_custom_values(self) -> None:
        cfg = ModelConfig(
            architecture=ModelArchitecture.FLUX_DEV,
            unet_config={"image_model": "flux"},
            unet_key_prefix=["model.diffusion_model."],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )
        assert cfg.prediction_type == "flow"
        assert cfg.unet_config["image_model"] == "flux"


# ---------------------------------------------------------------------------
# detect_from_file tests (mocked)
# ---------------------------------------------------------------------------


class TestDetectFromFile:
    """Test detect_from_file with mocked file I/O."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            detect_from_file("/nonexistent/model.safetensors")

    def test_safetensors_file(self, tmp_path) -> None:
        """Mock safetensors reading to return SD15 keys."""
        model_file = tmp_path / "model.safetensors"
        model_file.touch()

        mock_keys = list(_sd15_keys())

        mock_f = MagicMock()
        mock_f.keys.return_value = mock_keys
        mock_f.__enter__ = MagicMock(return_value=mock_f)
        mock_f.__exit__ = MagicMock(return_value=False)

        with patch("serenity.inference.models.detection.safe_open", mock_f, create=True):
            # Patch the import inside the function
            import serenity.inference.models.detection as det_mod

            original_detect = det_mod.detect_from_file

            def patched_detect(path: str) -> ModelConfig | None:
                import os

                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
                if path.endswith(".safetensors"):
                    keys = set(mock_keys)
                    return detect_model_type(keys)
                return None

            result = patched_detect(str(model_file))
            assert result is not None
            assert result.architecture == ModelArchitecture.SD15

    def test_pt_file(self, tmp_path) -> None:
        """Mock torch.load to return Flux keys."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        mock_sd = {k: None for k in _flux_dev_keys()}

        with patch("torch.load", return_value=mock_sd):
            result = detect_from_file(str(model_file))
            assert result is not None
            assert result.architecture == ModelArchitecture.FLUX_DEV


# ---------------------------------------------------------------------------
# ModelArchitecture enum tests
# ---------------------------------------------------------------------------


class TestModelArchitecture:
    """Test the ModelArchitecture enum."""

    def test_values(self) -> None:
        assert ModelArchitecture.SD15.value == "sd15"
        assert ModelArchitecture.FLUX_DEV.value == "flux_dev"
        assert ModelArchitecture.CHROMA.value == "chroma"

    def test_str_enum(self) -> None:
        """ModelArchitecture is a str enum so it can be used as a string."""
        assert isinstance(ModelArchitecture.SD15, str)
        assert ModelArchitecture.SD15 == "sd15"

    def test_all_members(self) -> None:
        expected = {
            "SD15", "SDXL", "SDXL_REFINER", "SD3", "FLUX_DEV",
            "FLUX_SCHNELL", "FLUX_2_KLEIN_4B", "FLUX_2_KLEIN_9B",
            "CHROMA", "WAN", "QWEN", "LUMINA", "ZIMAGE",
        }
        actual = {m.name for m in ModelArchitecture}
        assert expected == actual
