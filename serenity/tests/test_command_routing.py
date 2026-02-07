"""Tests for CLI backend routing between native diffusion and native Flux2 paths."""

from __future__ import annotations

from pathlib import Path

from serenity.cli import commands
from serenity.cli.commands import (
    _convert_legacy_to_serenity,
    _native_diffusion_opt_in,
    _legacy_bridge_opt_in,
    _should_use_native_flux2_backend,
)


def test_native_flux2_routing_for_explicit_klein_type():
    cfg = {
        "model_type": "flux_2_klein_4b",
        "model": {"path": "/tmp/mock-model"},
    }
    assert _should_use_native_flux2_backend(cfg, "flux_2_klein_4b")


def test_native_flux2_routing_for_legacy_flux2_alias_with_klein_path():
    cfg = {
        "model_type": "flux2",
        "model": {"path": "/models/black-forest-labs/FLUX.2-klein-base-9B"},
    }
    assert _should_use_native_flux2_backend(cfg, "flux2")


def test_native_flux2_routing_skips_non_klein_flux2():
    cfg = {
        "model_type": "flux2",
        "model": {"path": "/models/black-forest-labs/FLUX.2-dev"},
    }
    assert not _should_use_native_flux2_backend(cfg, "flux2")


def test_native_flux2_routing_skips_flux1_models():
    cfg = {
        "model_type": "flux_dev",
        "model": {"path": "/models/black-forest-labs/FLUX.1-dev"},
    }
    assert not _should_use_native_flux2_backend(cfg, "flux_dev")


def test_native_diffusion_opt_in_defaults_off():
    cfg = {
        "model_type": "qwen_image_edit",
        "model": {"path": "/models/qwen-image-edit"},
    }
    assert not _native_diffusion_opt_in(cfg)


def test_native_diffusion_opt_in_accepts_explicit_flag():
    cfg = {
        "model_type": "qwen_image_edit",
        "use_native_diffusion": True,
    }
    assert _native_diffusion_opt_in(cfg)


def test_convert_legacy_to_serenity_maps_model_and_paths(tmp_path):
    cfg = {
        "model_type": "FLUX_2_KLEIN_9B_BASE",
        "base_model_name": "/models/black-forest-labs/FLUX.2-klein-base-9B",
        "output_model_destination": str(tmp_path / "weights" / "adapter.safetensors"),
        "concepts": [{"path": "/datasets/disney", "balancing": 2.0}],
        "learning_rate": 3e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "lora_rank": 16,
        "lora_alpha": 16,
    }

    converted = _convert_legacy_to_serenity(cfg, source_path=Path("test.json"))

    assert converted["model_type"] == "flux_2_klein_9b_base"
    assert converted["model"]["path"] == cfg["base_model_name"]
    assert converted["checkpoint"]["output_dir"] == str(tmp_path / "weights")
    assert converted["data"]["concepts"][0]["path"] == "/datasets/disney"
    assert converted["adapter"]["type"] == "lora"


def test_convert_legacy_to_serenity_maps_qwen_edit_flags():
    cfg = {
        "model_type": "QWEN",
        "base_model_name": "/models/Qwen-Image",
        "output_model_destination": "/tmp/out",
        "concepts": [{"path": "/datasets/disney"}],
        "custom_conditioning_image": True,
        "training_method": "LORA",
    }
    converted = _convert_legacy_to_serenity(cfg, source_path=Path("test.json"))
    assert converted["model_type"] == "qwen_image_edit"


def test_convert_legacy_to_serenity_preserves_embedding_method():
    cfg = {
        "model_type": "STABLE_DIFFUSION_15",
        "base_model_name": "/models/sd15",
        "output_model_destination": "/tmp/out",
        "concepts": [{"path": "/datasets/disney"}],
        "training_method": "EMBEDDING",
    }
    converted = _convert_legacy_to_serenity(cfg, source_path=Path("test.json"))
    assert converted["training_method"] == "embedding"
    assert converted["adapter"]["type"] == "lora"


def test_legacy_bridge_opt_in_uses_backend_key():
    cfg = {"backend": "legacy"}
    assert _legacy_bridge_opt_in(cfg)


def test_train_command_defaults_to_native_without_opt_in(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text("{}")

    cfg = {
        "model_type": "qwen",
        "model": {"path": "/models/qwen-image"},
        "concepts": [{"path": "/datasets/disney"}],
    }

    monkeypatch.setattr(commands, "_load_config", lambda _: cfg)
    monkeypatch.setattr(commands, "_should_use_native_flux2_backend", lambda _cfg, _model_type: False)
    monkeypatch.setattr("serenity.cli.native_diffusion.is_native_diffusion_model_type", lambda _model_type: True)

    captured: dict[str, object] = {}

    def _fake_run_native_diffusion(cfg, *, source_path, steps_override):
        captured["cfg"] = cfg
        captured["source_path"] = source_path
        captured["steps_override"] = steps_override
        return 0

    monkeypatch.setattr("serenity.cli.native_diffusion.run_native_diffusion_training", _fake_run_native_diffusion)

    result = commands.train_command(["train", str(config_path)])
    assert result == 0
    assert captured["cfg"]["model_type"] == "qwen"


def test_train_command_converts_legacy_config_to_native(monkeypatch, tmp_path):
    config_path = tmp_path / "ot.json"
    config_path.write_text("{}")

    cfg = {
        "model_type": "QWEN",
        "base_model_name": "/models/qwen-image",
        "output_model_destination": str(tmp_path / "out"),
        "concepts": [{"path": "/datasets/disney"}],
    }

    monkeypatch.setattr(commands, "_load_config", lambda _: cfg)
    monkeypatch.setattr(commands, "_should_use_native_flux2_backend", lambda _cfg, _model_type: False)
    monkeypatch.setattr("serenity.cli.native_diffusion.is_native_diffusion_model_type", lambda _model_type: True)

    captured: dict[str, object] = {}

    def _fake_run_native_diffusion(cfg, *, source_path, steps_override):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr("serenity.cli.native_diffusion.run_native_diffusion_training", _fake_run_native_diffusion)

    result = commands.train_command(["train", str(config_path)])
    assert result == 0
    assert captured["cfg"]["model_type"] == "qwen"
    assert captured["cfg"]["model"]["path"] == "/models/qwen-image"
