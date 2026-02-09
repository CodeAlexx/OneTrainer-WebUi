"""Tests for native LoRA/DoRA adapter implementations.

Covers:
- LoRALinear apply / forward / merge / unmerge roundtrip
- DoRALinear apply / forward / magnitude_vector
- State dict extract / load roundtrip
- LoRAManager and DoRAManager with native adapters
- Adapter factory routing (LoRA/DoRA → native, exotic → LyCORIS)
- Zero PEFT imports in codebase
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_model():
    """A 3-layer model with two nn.Linear layers at positions 0 and 2."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


@pytest.fixture()
def input_tensor():
    return torch.randn(2, 16)


# ---------------------------------------------------------------------------
# Native LoRA tests
# ---------------------------------------------------------------------------


class TestLoRALinear:
    def test_apply_replaces_linear_layers(self, simple_model):
        from serenity.training.adapters.lora import LoRALinear, apply_lora

        created = apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        assert len(created) == 2
        assert isinstance(simple_model[0], LoRALinear)
        assert isinstance(simple_model[2], LoRALinear)

    def test_orig_frozen_lora_trainable(self, simple_model):
        from serenity.training.adapters.lora import apply_lora

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0"])
        lora = simple_model[0]
        for p in lora.orig.parameters():
            assert not p.requires_grad, "orig should be frozen"
        assert lora.lora_down.weight.requires_grad
        assert lora.lora_up.weight.requires_grad

    def test_forward_shape(self, simple_model, input_tensor):
        from serenity.training.adapters.lora import apply_lora

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        y = simple_model(input_tensor)
        assert y.shape == (2, 8)

    def test_get_lora_params(self, simple_model):
        from serenity.training.adapters.lora import apply_lora, get_lora_params

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        params = get_lora_params(simple_model)
        assert len(params) == 4  # 2 layers x (down + up)
        assert all(isinstance(p, nn.Parameter) for p in params)

    def test_merge_unmerge_roundtrip(self, simple_model, input_tensor):
        from serenity.training.adapters.lora import apply_lora, merge_lora, unmerge_lora

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        y_before = simple_model(input_tensor).clone()

        merge_lora(simple_model)
        assert simple_model[0].merged
        y_merged = simple_model(input_tensor)
        assert torch.allclose(y_before, y_merged, atol=1e-5)

        unmerge_lora(simple_model)
        assert not simple_model[0].merged
        y_unmerged = simple_model(input_tensor)
        assert torch.allclose(y_before, y_unmerged, atol=1e-5)

    def test_state_dict_roundtrip(self, simple_model):
        from serenity.training.adapters.lora import (
            apply_lora,
            extract_lora_state_dict,
            load_lora_state_dict,
        )

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        sd = extract_lora_state_dict(simple_model)
        assert "0.lora_down.weight" in sd
        assert "0.lora_up.weight" in sd
        assert "2.lora_down.weight" in sd
        assert "2.lora_up.weight" in sd

        # Load into fresh model
        model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        apply_lora(model2, rank=4, alpha=8.0, target_modules=["0", "2"])
        load_lora_state_dict(model2, sd)
        sd2 = extract_lora_state_dict(model2)
        for key in sd:
            assert torch.equal(sd[key], sd2[key]), f"Key {key} mismatch after load"

    def test_no_match_returns_empty(self, simple_model):
        from serenity.training.adapters.lora import apply_lora

        created = apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["nonexistent"])
        assert len(created) == 0

    def test_dropout_applied(self, simple_model):
        from serenity.training.adapters.lora import apply_lora

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0"], dropout=0.5)
        assert isinstance(simple_model[0].dropout, nn.Dropout)

    def test_no_dropout_identity(self, simple_model):
        from serenity.training.adapters.lora import apply_lora

        apply_lora(simple_model, rank=4, alpha=8.0, target_modules=["0"], dropout=0.0)
        assert isinstance(simple_model[0].dropout, nn.Identity)


# ---------------------------------------------------------------------------
# Native DoRA tests
# ---------------------------------------------------------------------------


class TestDoRALinear:
    def test_apply_creates_dora(self, simple_model):
        from serenity.training.adapters.dora import DoRALinear, apply_dora
        from serenity.training.adapters.lora import LoRALinear

        created = apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        assert len(created) == 2
        assert isinstance(simple_model[0], DoRALinear)
        assert isinstance(simple_model[0], LoRALinear)  # subclass

    def test_magnitude_vector_present(self, simple_model):
        from serenity.training.adapters.dora import apply_dora

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0"])
        assert hasattr(simple_model[0], "magnitude_vector")
        assert isinstance(simple_model[0].magnitude_vector, nn.Parameter)
        assert simple_model[0].magnitude_vector.shape == (32,)

    def test_forward_shape(self, simple_model, input_tensor):
        from serenity.training.adapters.dora import apply_dora

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        y = simple_model(input_tensor)
        assert y.shape == (2, 8)

    def test_get_lora_params_includes_magnitude(self, simple_model):
        from serenity.training.adapters.dora import apply_dora
        from serenity.training.adapters.lora import get_lora_params

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        params = get_lora_params(simple_model)
        assert len(params) == 6  # 2 layers x (down + up + magnitude)

    def test_state_dict_includes_magnitude(self, simple_model):
        from serenity.training.adapters.dora import apply_dora
        from serenity.training.adapters.lora import extract_lora_state_dict

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0", "2"])
        sd = extract_lora_state_dict(simple_model)
        assert "0.magnitude_vector" in sd
        assert "2.magnitude_vector" in sd

    def test_merge_raises(self, simple_model):
        from serenity.training.adapters.dora import apply_dora

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0"])
        with pytest.raises(NotImplementedError):
            simple_model[0].merge_weights()

    def test_unmerge_raises(self, simple_model):
        from serenity.training.adapters.dora import apply_dora

        apply_dora(simple_model, rank=4, alpha=8.0, target_modules=["0"])
        with pytest.raises(NotImplementedError):
            simple_model[0].unmerge_weights()


# ---------------------------------------------------------------------------
# LoRAManager / DoRAManager tests
# ---------------------------------------------------------------------------


class TestLoRAManager:
    def test_apply_and_is_attached(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = LoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        assert manager.is_attached()

    def test_prepare_optimizer_params(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = LoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        groups = manager.prepare_optimizer_params(lr=1e-4)
        assert len(groups) == 1
        assert "params" in groups[0]
        assert len(groups[0]["params"]) == 4
        assert groups[0]["lr"] == 1e-4

    def test_state_dict_and_load(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = LoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        sd = manager.state_dict()
        assert len(sd) == 4

        model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        manager2 = LoRAManager(cfg, model_type="flux_2")
        manager2.apply(model2)
        manager2.load_state_dict(sd)
        sd2 = manager2.state_dict()
        for key in sd:
            assert torch.equal(sd[key], sd2[key])

    def test_merge_and_restore(self, simple_model, input_tensor):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = LoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        y_before = simple_model(input_tensor).clone()

        manager.merge_to()
        y_merged = simple_model(input_tensor)
        assert torch.allclose(y_before, y_merged, atol=1e-5)

        manager.restore()
        y_restored = simple_model(input_tensor)
        assert torch.allclose(y_before, y_restored, atol=1e-5)

    def test_save_and_load_weights(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = LoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        sd_before = manager.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lora.safetensors"
            manager.save_weights(path)
            assert path.exists()

            model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
            manager2 = LoRAManager(cfg, model_type="flux_2")
            manager2.apply(model2)
            manager2.load_weights(str(path))
            sd_after = manager2.state_dict()
            for key in sd_before:
                assert torch.equal(sd_before[key], sd_after[key])

    def test_not_attached_raises(self):
        from serenity.training.lora_manager import AdapterConfig, LoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0)
        manager = LoRAManager(cfg, model_type="flux_2")
        with pytest.raises(RuntimeError):
            manager.prepare_optimizer_params()
        with pytest.raises(RuntimeError):
            manager.merge_to()


class TestDoRAManager:
    def test_apply_creates_dora_layers(self, simple_model):
        from serenity.training.adapters.dora import DoRALinear
        from serenity.training.lora_manager import AdapterConfig, DoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = DoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        assert manager.is_attached()
        assert isinstance(simple_model[0], DoRALinear)

    def test_merge_raises(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, DoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = DoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        with pytest.raises(NotImplementedError):
            manager.merge_to()

    def test_restore_raises(self, simple_model):
        from serenity.training.lora_manager import AdapterConfig, DoRAManager

        cfg = AdapterConfig(rank=4, alpha=8.0, target_modules=["0", "2"])
        manager = DoRAManager(cfg, model_type="flux_2")
        manager.apply(simple_model)
        with pytest.raises(NotImplementedError):
            manager.restore()


# ---------------------------------------------------------------------------
# Adapter factory routing tests
# ---------------------------------------------------------------------------


class TestAdapterFactoryRouting:
    def test_lora_routes_to_native(self):
        from serenity.adapters import create_adapter

        adapter = create_adapter("lora", rank=4, alpha=8.0, model_type="flux_2")
        manager = adapter._ensure_manager()
        from serenity.training.lora_manager import LoRAManager

        assert isinstance(manager, LoRAManager)

    def test_dora_routes_to_native(self):
        from serenity.adapters import create_adapter

        adapter = create_adapter("dora", rank=4, alpha=8.0, model_type="flux_2")
        manager = adapter._ensure_manager()
        from serenity.training.lora_manager import DoRAManager

        assert isinstance(manager, DoRAManager)

    def test_exotic_routes_to_lycoris(self):
        from serenity.adapters import create_adapter

        adapter = create_adapter("lokr", rank=4, alpha=8.0, model_type="flux_2")
        manager = adapter._ensure_manager()
        from serenity.training.lycoris_manager import LyCORISManager

        assert isinstance(manager, LyCORISManager)


# ---------------------------------------------------------------------------
# PEFT eradication test
# ---------------------------------------------------------------------------


class TestNoPEFT:
    def test_no_peft_imports_in_codebase(self):
        """Verify no PEFT imports remain anywhere in the serenity package."""
        import os

        _PEFT_MARKER_FROM = "from pef" + "t"
        _PEFT_MARKER_IMPORT = "import pef" + "t"
        serenity_root = Path(__file__).parent.parent
        violations: list[str] = []
        for dirpath, _dirs, files in os.walk(serenity_root):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = Path(dirpath) / fname
                if fpath == Path(__file__):
                    continue
                try:
                    text = fpath.read_text()
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if _PEFT_MARKER_FROM in stripped or _PEFT_MARKER_IMPORT in stripped:
                        violations.append(f"{fpath}:{i}: {stripped}")
        assert violations == [], f"PEFT imports found:\n" + "\n".join(violations)
