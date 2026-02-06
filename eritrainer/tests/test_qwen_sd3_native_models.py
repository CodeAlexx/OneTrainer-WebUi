"""Tests for native Qwen and SD3 model helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eritrainer.cli.native_diffusion import _prepare_training_pairs, _resolve_conditioning_image_path
from eritrainer.core.interfaces import ModelType
from eritrainer.models.qwen import QwenModel, _validate_qwen_model_path
from eritrainer.models.sd3 import SD3Model, _validate_sd3_model_path

import torch

import pytest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def test_validate_qwen_model_path_accepts_complete_layout(tmp_path):
    root = tmp_path / "qwen"
    _touch(root / "scheduler" / "scheduler_config.json")
    _touch(root / "tokenizer" / "tokenizer_config.json")
    _touch(root / "text_encoder" / "config.json")
    _touch(root / "vae" / "config.json")
    _touch(root / "transformer" / "config.json")

    assert _validate_qwen_model_path(str(root)) == root


def test_validate_sd3_model_path_accepts_complete_layout(tmp_path):
    root = tmp_path / "sd3"
    _touch(root / "scheduler" / "scheduler_config.json")
    _touch(root / "tokenizer" / "tokenizer_config.json")
    _touch(root / "tokenizer_2" / "tokenizer_config.json")
    _touch(root / "tokenizer_3" / "tokenizer_config.json")
    _touch(root / "text_encoder" / "config.json")
    _touch(root / "text_encoder_2" / "config.json")
    _touch(root / "text_encoder_3" / "config.json")
    _touch(root / "transformer" / "config.json")
    _touch(root / "vae" / "config.json")

    assert _validate_sd3_model_path(str(root)) == root


def test_prepare_training_pairs_filters_qwen_edit_condlabel(tmp_path):
    target = tmp_path / "image01.jpg"
    cond = tmp_path / "image01-condlabel.png"
    target.write_text("x")
    cond.write_text("x")

    pairs = [(target, "caption"), (cond, "cond caption")]
    prepared = _prepare_training_pairs(pairs, model_type=ModelType.QWEN_IMAGE_EDIT)

    assert len(prepared) == 1
    assert prepared[0][0] == target


def test_resolve_conditioning_image_path_prefers_condlabel(tmp_path):
    target = tmp_path / "image01.jpg"
    cond = tmp_path / "image01-condlabel.png"
    target.write_text("x")
    cond.write_text("x")

    assert _resolve_conditioning_image_path(target) == cond


def test_qwen_encode_prompt_features_passes_conditioning_image():
    class DummyPipeline:
        def __init__(self) -> None:
            self.kwargs = None

        def encode_prompt(self, *, prompt, device, num_images_per_prompt, max_sequence_length, image=None):
            kwargs = {
                "prompt": prompt,
                "device": device,
                "num_images_per_prompt": num_images_per_prompt,
                "max_sequence_length": max_sequence_length,
                "image": image,
            }
            self.kwargs = kwargs
            prompt_embeds = torch.randn(1, 3, 4)
            prompt_mask = torch.ones(1, 3, dtype=torch.long)
            return prompt_embeds, prompt_mask

    pipeline = DummyPipeline()
    model = QwenModel()
    conditioning = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

    prompt_embeds, pooled, prompt_mask = model.encode_prompt_features(
        pipeline,
        "caption",
        device=torch.device("cpu"),
        conditioning_image=conditioning,
    )

    assert prompt_embeds.shape == (1, 3, 4)
    assert pooled is None
    assert prompt_mask.shape == (1, 3)
    assert pipeline.kwargs is not None
    assert pipeline.kwargs["image"].dtype == torch.float32


def test_qwen_encode_latents_accepts_4d_pixels():
    class DummyVAE:
        class _Config:
            z_dim = 2
            latents_mean = [0.0, 0.0]
            latents_std = [1.0, 1.0]

        def __init__(self) -> None:
            self.config = self._Config()
            self.seen_shape = None

        def encode(self, pixel_values):
            self.seen_shape = tuple(pixel_values.shape)
            latents = torch.zeros((pixel_values.shape[0], 2, 1, 8, 8), dtype=pixel_values.dtype)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latents))

    class DummyPipeline:
        def __init__(self) -> None:
            self.vae = DummyVAE()

    pipeline = DummyPipeline()
    model = QwenModel()
    pixel_values = torch.randn(1, 3, 64, 64)
    latents = model.encode_latents(pipeline, pixel_values)

    assert pipeline.vae.seen_shape == (1, 3, 1, 64, 64)
    assert latents.shape == (1, 2, 1, 8, 8)


def test_qwen_quantization_mode_normalization_aliases():
    assert QwenModel.normalize_quantization_mode(None) is None
    assert QwenModel.normalize_quantization_mode("off") is None
    assert QwenModel.normalize_quantization_mode("INT_8") == "int8"
    assert QwenModel.normalize_quantization_mode("w8a8_int") == "int8"
    assert QwenModel.normalize_quantization_mode("FP8") == "fp8"
    assert QwenModel.normalize_quantization_mode("float_w8a8") == "fp8"


def test_qwen_load_pipeline_routes_to_fp8_loader(monkeypatch, tmp_path):
    root = tmp_path / "qwen"
    _touch(root / "scheduler" / "scheduler_config.json")
    _touch(root / "tokenizer" / "tokenizer_config.json")
    _touch(root / "text_encoder" / "config.json")
    _touch(root / "vae" / "config.json")
    _touch(root / "transformer" / "config.json")

    model = QwenModel()
    calls: dict[str, object] = {}

    def _fake_fp8_loader(self, model_root, dtype, train_device, requested_gpu_budget_gib=None):
        calls["model_root"] = model_root
        calls["dtype"] = dtype
        calls["train_device"] = train_device
        calls["requested_gpu_budget_gib"] = requested_gpu_budget_gib
        return "fp8_pipeline"

    monkeypatch.setattr(QwenModel, "_load_pipeline_fp8", _fake_fp8_loader)

    pipeline = model.load_pipeline(
        str(root),
        torch.bfloat16,
        torch.device("cpu"),
        quantization_mode="fp8",
        requested_gpu_budget_gib=10,
    )

    assert pipeline == "fp8_pipeline"
    assert calls["model_root"] == root
    assert calls["dtype"] == torch.bfloat16
    assert calls["train_device"] == torch.device("cpu")
    assert calls["requested_gpu_budget_gib"] == 10


def test_sd3_encode_prompt_features_extracts_pooled():
    class DummyPipeline:
        def encode_prompt(self, **kwargs):
            del kwargs
            prompt_embeds = torch.randn(1, 77, 4096)
            ignored_mask = torch.ones(1, 77, dtype=torch.long)
            pooled = torch.randn(1, 2048)
            return prompt_embeds, ignored_mask, pooled

    model = SD3Model()
    prompt_embeds, pooled, prompt_mask = model.encode_prompt_features(
        DummyPipeline(),
        "caption",
        device=torch.device("cpu"),
    )

    assert prompt_embeds.shape == (1, 77, 4096)
    assert pooled is not None
    assert pooled.shape == (1, 2048)
    assert prompt_mask is None


def test_validate_helpers_raise_for_missing_layout(tmp_path):
    with pytest.raises(FileNotFoundError):
        _validate_qwen_model_path(str(tmp_path / "missing_qwen"))
    with pytest.raises(FileNotFoundError):
        _validate_sd3_model_path(str(tmp_path / "missing_sd3"))
