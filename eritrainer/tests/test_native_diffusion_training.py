"""Tests for native non-bridge diffusion training helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eritrainer.cli.native_diffusion import (
    _compute_vae_loss,
    _build_video_training_pairs,
    _collect_ltx_component_paths,
    _load_media_tensor,
    _maybe_restore_training_state,
    _maybe_load_transformer_override,
    _maybe_sample,
    _normalize_quantization_mode,
    _normalize_training_method,
    _qwen_pack_latents,
    _qwen_unpack_latents,
    _resolve_component_train_flags,
    _resolve_ltx_video_frame_count,
    _save_training_state,
    is_native_diffusion_model_type,
)
from eritrainer.training.ema import EMAModel
from eritrainer.core.interfaces import ModelType

import torch
import torch.nn as nn

import numpy as np


def test_native_diffusion_model_type_aliases():
    assert is_native_diffusion_model_type("flux")
    assert is_native_diffusion_model_type("flux2")
    assert is_native_diffusion_model_type("z_image")
    assert is_native_diffusion_model_type("qwen")
    assert is_native_diffusion_model_type("qwen_image_edit")
    assert is_native_diffusion_model_type("ltx")
    assert is_native_diffusion_model_type("ltx2")
    assert is_native_diffusion_model_type("sdxl")
    assert is_native_diffusion_model_type("sd_15")
    assert is_native_diffusion_model_type("sd3.5")
    assert not is_native_diffusion_model_type("flux_2_klein_4b")


def test_build_video_training_pairs(tmp_path):
    concept_dir = tmp_path / "videos"
    concept_dir.mkdir(parents=True, exist_ok=True)

    video_path = concept_dir / "sample.mp4"
    caption_path = concept_dir / "sample.txt"
    video_path.write_bytes(b"not-a-real-video")
    caption_path.write_text("video caption", encoding="utf-8")

    cfg = {
        "concepts": [{"path": str(concept_dir)}],
    }
    pairs = _build_video_training_pairs(cfg)

    assert len(pairs) == 1
    assert pairs[0][0] == video_path
    assert pairs[0][1] == "video caption"


def test_qwen_pack_unpack_round_trip():
    latents = torch.randn(2, 16, 1, 32, 32)
    packed = _qwen_pack_latents(latents)
    unpacked = _qwen_unpack_latents(packed, 32, 32)

    assert packed.shape == (2, 256, 64)
    assert unpacked.shape == latents.shape
    assert torch.allclose(unpacked, latents, atol=1e-5)


def test_native_quantization_aliases():
    assert _normalize_quantization_mode(None) is None
    assert _normalize_quantization_mode("off") is None
    assert _normalize_quantization_mode("INT_8") == "int8"
    assert _normalize_quantization_mode("int_w8a8") == "int8"
    assert _normalize_quantization_mode("w8a8_int") == "int8"
    assert _normalize_quantization_mode("fp8") == "fp8"


def test_native_training_method_normalization():
    assert _normalize_training_method("LORA") == "lora"
    assert _normalize_training_method("fine_tune") == "fine_tune"
    assert _normalize_training_method("finetune") == "fine_tune"
    assert _normalize_training_method("full_finetune") == "fine_tune"
    assert _normalize_training_method("fine_tune_vae") == "fine_tune_vae"
    assert _normalize_training_method("embedding") == "embedding"


def test_resolve_component_train_flags_defaults():
    train_primary, text_flags, train_vae = _resolve_component_train_flags(
        config={},
        model_block={},
        family="sd15",
        full_finetune=True,
        training_method="fine_tune",
    )
    assert train_primary is True
    assert text_flags["text_encoder"] is False
    assert train_vae is False


def test_compute_vae_loss_with_dummy_vae():
    class _DummyDist:
        def __init__(self, x):
            self._x = x

        def sample(self):
            return self._x

        def kl(self):
            return torch.zeros(1, device=self._x.device, dtype=self._x.dtype)

    class _DummyVAE(torch.nn.Module):
        def encode(self, x):
            return SimpleNamespace(latent_dist=_DummyDist(x))

        def decode(self, z):
            return SimpleNamespace(sample=z)

    vae = _DummyVAE()
    pixel_values = torch.randn(2, 3, 16, 16)
    loss = _compute_vae_loss(vae, pixel_values, kl_weight=1e-6)
    assert torch.is_tensor(loss)
    assert float(loss.detach().cpu()) >= 0.0


def test_save_and_restore_training_state_roundtrip(tmp_path):
    module = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    ema = EMAModel(modules=[module], decay=0.99)
    state_path = tmp_path / "state_step_000005.pt"

    _save_training_state(
        state_path,
        step=5,
        model_checkpoint=None,
        optimizer=optimizer,
        lr_scheduler=None,
        ema_model=ema,
    )

    next_step = _maybe_restore_training_state(
        resume_state_path=state_path,
        train_module=module,
        adapter=None,
        optimizer=optimizer,
        lr_scheduler=None,
        ema_model=ema,
    )
    assert next_step == 6


def test_collect_ltx_component_paths_prefers_model_block_paths():
    config = {
        "vae_path": "/global/vae",
        "text_encoder_path": "/global/text",
        "ltx_template_path": "/global/template",
    }
    model_block = {
        "vae_path": "/model/vae",
        "text_encoder_path": "/model/text",
        "tokenizer_path": "/model/tokenizer",
    }

    component_paths, template_path = _collect_ltx_component_paths(config, model_block)
    assert component_paths["vae"] == "/model/vae"
    assert component_paths["text_encoder"] == "/model/text"
    assert component_paths["tokenizer"] == "/model/tokenizer"
    assert template_path == "/global/template"


def test_collect_ltx_component_paths_reads_top_level_aliases():
    config = {
        "transformer_path": "~/models/ltx.safetensors",
        "clip_path": "~/models/clip",
        "vae": "~/models/VAE",
        "scheduler_dir": "~/models/scheduler",
        "template_path": "~/models/LTX-Video-0.9.7-dev",
    }

    component_paths, template_path = _collect_ltx_component_paths(config, {})
    assert component_paths["transformer"].endswith("/models/ltx.safetensors")
    assert component_paths["text_encoder"].endswith("/models/clip")
    assert component_paths["vae"].endswith("/models/VAE")
    assert component_paths["scheduler"].endswith("/models/scheduler")
    assert template_path and template_path.endswith("/models/LTX-Video-0.9.7-dev")


def test_resolve_ltx_video_frame_count_adjusts_to_valid_stride():
    assert _resolve_ltx_video_frame_count({"frames": 10}, {}) == 9
    assert _resolve_ltx_video_frame_count({"frames": 17}, {}) == 17
    assert _resolve_ltx_video_frame_count({}, {}) == 9


def test_load_media_tensor_repeats_image_for_ltx_video_mode(monkeypatch):
    base = torch.ones(3, 4, 4, dtype=torch.float32)

    monkeypatch.setattr(
        "eritrainer.cli.native_diffusion._load_image_tensor",
        lambda *_args, **_kwargs: base.clone(),
    )

    loaded = _load_media_tensor(
        Path("sample.png"),
        resolution=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        allow_video=True,
        video_frame_count=9,
    )
    assert loaded.shape == (3, 9, 4, 4)
    assert torch.allclose(loaded[:, 0], base)
    assert torch.allclose(loaded[:, -1], base)


def test_load_media_tensor_decodes_and_pads_video_frames(monkeypatch):
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 255, dtype=np.uint8)
    raw = frame_a.tobytes() + frame_b.tobytes()

    monkeypatch.setattr(
        "eritrainer.cli.native_diffusion.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=raw, stderr=b""),
    )

    loaded = _load_media_tensor(
        Path("clip.mp4"),
        resolution=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        allow_video=True,
        video_frame_count=5,
    )
    assert loaded.shape == (3, 5, 2, 2)
    assert torch.allclose(loaded[:, 0], torch.full((3, 2, 2), -1.0))
    assert torch.allclose(loaded[:, 1], torch.full((3, 2, 2), 1.0))
    assert torch.allclose(loaded[:, 4], loaded[:, 1])


def test_maybe_sample_expands_prompts_and_seeds(monkeypatch, tmp_path):
    calls: list[dict] = []

    class _DummySampler:
        def sample(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        "eritrainer.cli.native_diffusion.create_sampler",
        lambda *_args, **_kwargs: _DummySampler(),
    )

    _maybe_sample(
        {
            "seed": 123,
            "sample": {
                "enabled": True,
                "interval": 2,
                "prompts": ["a", "b", "c"],
                "seeds": [11, 22],
            },
        },
        model_type=ModelType.SDXL,
        model_path="/tmp/mock",
        output_dir=tmp_path,
        step=2,
        train_device=torch.device("cpu"),
        train_dtype=torch.float32,
    )

    assert len(calls) == 3
    assert [int(call["seed"]) for call in calls] == [11, 22, 11]
    assert all(str(call["output_path"]).endswith(".png") for call in calls)
    assert calls[-1]["unload"] is True


def test_maybe_sample_uses_default_resolution_when_not_overridden(monkeypatch, tmp_path):
    calls: list[dict] = []

    class _DummySampler:
        def sample(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        "eritrainer.cli.native_diffusion.create_sampler",
        lambda *_args, **_kwargs: _DummySampler(),
    )

    _maybe_sample(
        {
            "seed": 7,
            "sample": {
                "enabled": True,
                "interval": 1,
                "prompts": ["portrait"],
            },
        },
        model_type=ModelType.SDXL,
        model_path="/tmp/mock",
        output_dir=tmp_path,
        step=1,
        train_device=torch.device("cpu"),
        train_dtype=torch.float32,
        default_resolution=512,
    )

    assert len(calls) == 1
    assert int(calls[0]["height"]) == 512
    assert int(calls[0]["width"]) == 512


def test_maybe_sample_ltx2_passes_video_args_and_uses_gif(monkeypatch, tmp_path):
    calls: list[dict] = []

    class _DummySampler:
        def sample(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        "eritrainer.cli.native_diffusion.create_sampler",
        lambda *_args, **_kwargs: _DummySampler(),
    )

    _maybe_sample(
        {
            "seed": 99,
            "sample": {
                "enabled": True,
                "interval": 1,
                "prompts": ["video sample"],
                "num_frames": 17,
                "frame_rate": 12.0,
            },
        },
        model_type=ModelType.LTX2,
        model_path="/tmp/mock",
        output_dir=tmp_path,
        step=1,
        train_device=torch.device("cpu"),
        train_dtype=torch.float32,
    )

    assert len(calls) == 1
    call = calls[0]
    assert int(call["num_frames"]) == 17
    assert float(call["frame_rate"]) == 12.0
    assert str(call["output_path"]).endswith(".gif")


def test_maybe_load_transformer_override_treats_zimage_turbo_as_adapter(monkeypatch, tmp_path):
    adapter_path = tmp_path / "zimage_turbo_training_adapter.safetensors"
    adapter_path.write_bytes(b"fake")

    class _DummyPipeline:
        called: dict | None = None

        def __init__(self, transformer):
            self.transformer = transformer

        @classmethod
        def load_lora_into_transformer(cls, state_dict, transformer, adapter_name=None, **kwargs):
            cls.called = {
                "state_dict": state_dict,
                "transformer": transformer,
                "adapter_name": adapter_name,
                "kwargs": kwargs,
            }

    module = nn.Module()
    module.foo = nn.Module()
    module.foo.assistant = nn.Module()
    module.foo.assistant.lora_A = nn.Parameter(torch.ones(1))
    module.foo.base = nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda *_args, **_kwargs: {"transformer.lora_A.weight": torch.zeros(1)},
    )

    _maybe_load_transformer_override(
        module,
        {"turbo_adapter_path": str(adapter_path), "assistant_lora_strength": 0.7},
        pipeline=_DummyPipeline(module),
        family="zimage",
    )

    assert _DummyPipeline.called is not None
    assert _DummyPipeline.called["adapter_name"] == "assistant"
    assert module.foo.assistant.lora_A.requires_grad is False
