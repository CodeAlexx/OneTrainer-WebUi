"""Tests for serenity.inference.lora."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from serenity.inference.lora.loader import detect_lora_type, normalize_lora_keys
from serenity.inference.lora.merge import (
    merge_lora_into_model,
    merge_lora_to_weight,
    unmerge_lora_from_model,
    weight_decompose,
)
from serenity.inference.lora.online import apply_online_lora, remove_online_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_model() -> nn.Module:
    """Create a small 2-layer MLP for testing."""
    model = nn.Sequential(
        nn.Linear(16, 32, bias=True),
        nn.ReLU(),
        nn.Linear(32, 8, bias=True),
    )
    # Ensure deterministic weights
    torch.manual_seed(123)
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    return model


def _make_lora_state_dict(
    prefix_0: str = "0",
    prefix_1: str = "2",
    rank: int = 4,
) -> dict[str, torch.Tensor]:
    """Create a LoRA state dict targeting a 2-layer MLP (layers 0 and 2)."""
    torch.manual_seed(99)
    sd: dict[str, torch.Tensor] = {}

    # LoRA for first linear (16 -> 32)
    sd[f"{prefix_0}.lora_down.weight"] = torch.randn(rank, 16)
    sd[f"{prefix_0}.lora_up.weight"] = torch.randn(32, rank)

    # LoRA for second linear (32 -> 8)
    sd[f"{prefix_1}.lora_down.weight"] = torch.randn(rank, 32)
    sd[f"{prefix_1}.lora_up.weight"] = torch.randn(8, rank)

    return sd


# ---------------------------------------------------------------------------
# merge_lora_to_weight
# ---------------------------------------------------------------------------


class TestMergeLoraToWeight:
    def test_diff_patch(self):
        """Additive diff should add strength * diff to weight."""
        weight = torch.zeros(4, 8)
        diff = torch.ones(4, 8) * 2.0
        patches = [{"type": "diff", "diff": diff}]
        result = merge_lora_to_weight(weight, patches, strength=0.5)
        expected = torch.ones(4, 8) * 1.0
        torch.testing.assert_close(result, expected)

    def test_set_patch(self):
        """Set patch should replace the weight entirely."""
        weight = torch.zeros(4, 8)
        replacement = torch.ones(4, 8) * 3.0
        patches = [{"type": "set", "weight": replacement}]
        result = merge_lora_to_weight(weight, patches, strength=1.0)
        torch.testing.assert_close(result, replacement)

    def test_lora_up_down(self):
        """Standard LoRA: weight += strength * (alpha/rank) * (up @ down)."""
        weight = torch.zeros(8, 16)
        rank = 4
        up = torch.ones(8, rank)
        down = torch.ones(rank, 16)
        alpha = 4.0
        patches = [{"up": up, "down": down, "alpha": alpha}]
        result = merge_lora_to_weight(weight, patches, strength=1.0)

        # (alpha/rank) * strength = 1.0, up @ down = 4 * ones(8, 16)
        expected = torch.ones(8, 16) * rank
        torch.testing.assert_close(result, expected)

    def test_lora_strength_scaling(self):
        """Strength < 1 should scale the diff."""
        weight = torch.zeros(8, 16)
        up = torch.ones(8, 2)
        down = torch.ones(2, 16)
        patches = [{"up": up, "down": down, "alpha": 2.0}]
        result = merge_lora_to_weight(weight, patches, strength=0.5)
        # scale = (2/2) * 0.5 = 0.5, up @ down = 2 * ones(8,16)
        expected = torch.ones(8, 16) * 1.0
        torch.testing.assert_close(result, expected)

    def test_multiple_patches(self):
        """Multiple patches should be applied sequentially."""
        weight = torch.zeros(4, 4)
        patches = [
            {"type": "diff", "diff": torch.ones(4, 4)},
            {"type": "diff", "diff": torch.ones(4, 4) * 2},
        ]
        result = merge_lora_to_weight(weight, patches, strength=1.0)
        expected = torch.ones(4, 4) * 3.0
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# weight_decompose (DoRA)
# ---------------------------------------------------------------------------


class TestWeightDecompose:
    def test_dora_basic(self):
        """DoRA should modify weight direction while respecting scale."""
        torch.manual_seed(42)
        weight = torch.randn(4, 8)
        lora_diff = torch.randn(4, 8) * 0.1
        dora_scale = torch.ones(4)

        result = weight_decompose(
            dora_scale, weight.clone(), lora_diff, alpha=1.0, strength=1.0
        )
        # Result should differ from input
        assert not torch.allclose(result, weight)
        # Shape should be preserved
        assert result.shape == weight.shape

    def test_dora_zero_diff(self):
        """Zero diff with dora_scale=norm should roughly preserve weight."""
        torch.manual_seed(42)
        weight = torch.randn(4, 8)
        lora_diff = torch.zeros(4, 8)
        # dora_scale = column norms of weight
        dora_scale = weight.reshape(4, -1).norm(dim=1)

        result = weight_decompose(
            dora_scale, weight.clone(), lora_diff, alpha=1.0, strength=1.0
        )
        torch.testing.assert_close(result, weight, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# merge / unmerge model
# ---------------------------------------------------------------------------


class TestMergeUnmergeModel:
    def test_merge_changes_weights(self):
        """Merging LoRA should change model weights."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        original_w0 = model[0].weight.data.clone()
        merge_lora_into_model(model, lora_sd, strength=1.0)

        assert not torch.equal(model[0].weight.data, original_w0)

    def test_unmerge_restores_weights(self):
        """Unmerging should restore original weights exactly."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        original_w0 = model[0].weight.data.clone()
        original_w2 = model[2].weight.data.clone()

        merge_lora_into_model(model, lora_sd, strength=1.0)
        unmerge_lora_from_model(model)

        torch.testing.assert_close(model[0].weight.data, original_w0)
        torch.testing.assert_close(model[2].weight.data, original_w2)

    def test_merge_strength_zero(self):
        """Strength 0 should leave weights unchanged."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        original_w0 = model[0].weight.data.clone()
        merge_lora_into_model(model, lora_sd, strength=0.0)

        torch.testing.assert_close(model[0].weight.data, original_w0)

    def test_empty_state_dict(self):
        """Empty state dict should be a no-op."""
        model = _make_small_model()
        original_w0 = model[0].weight.data.clone()
        merge_lora_into_model(model, {}, strength=1.0)
        torch.testing.assert_close(model[0].weight.data, original_w0)

    def test_unmerge_without_merge(self):
        """Unmerge without prior merge should be a no-op."""
        model = _make_small_model()
        unmerge_lora_from_model(model)  # Should not raise


# ---------------------------------------------------------------------------
# detect_lora_type
# ---------------------------------------------------------------------------


class TestDetectLoraType:
    def test_standard(self):
        sd = {"layer.lora_up.weight": torch.zeros(1), "layer.lora_down.weight": torch.zeros(1)}
        assert detect_lora_type(sd) == "standard"

    def test_kohya(self):
        sd = {"layer.lora_A.weight": torch.zeros(1), "layer.lora_B.weight": torch.zeros(1)}
        assert detect_lora_type(sd) == "kohya"

    def test_diffusers(self):
        sd = {"base_model.model.layer.lora_A.weight": torch.zeros(1)}
        assert detect_lora_type(sd) == "diffusers"

    def test_empty(self):
        assert detect_lora_type({}) == "standard"


# ---------------------------------------------------------------------------
# normalize_lora_keys
# ---------------------------------------------------------------------------


class TestNormalizeLoraKeys:
    def test_kohya_normalisation(self):
        sd = {
            "layer.lora_A.weight": torch.zeros(4, 8),
            "layer.lora_B.weight": torch.zeros(16, 4),
        }
        normalised = normalize_lora_keys(sd, "kohya")
        assert "layer.lora_down.weight" in normalised
        assert "layer.lora_up.weight" in normalised

    def test_diffusers_normalisation(self):
        sd = {
            "base_model.model.layer.lora_A.weight": torch.zeros(4, 8),
            "base_model.model.layer.lora_B.weight": torch.zeros(16, 4),
        }
        normalised = normalize_lora_keys(sd, "diffusers")
        assert "layer.lora_down.weight" in normalised
        assert "layer.lora_up.weight" in normalised

    def test_standard_passthrough(self):
        sd = {
            "layer.lora_up.weight": torch.zeros(16, 4),
            "layer.lora_down.weight": torch.zeros(4, 8),
        }
        normalised = normalize_lora_keys(sd, "standard")
        assert normalised.keys() == sd.keys()


# ---------------------------------------------------------------------------
# Online LoRA
# ---------------------------------------------------------------------------


class TestOnlineLoRA:
    def test_apply_stores_patches(self):
        """apply_online_lora should attach _online_lora_patches to modules."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        apply_online_lora(model, lora_sd, strength=1.0)

        # Check that at least one module has the attribute
        found = False
        for _name, mod in model.named_modules():
            if hasattr(mod, "_online_lora_patches"):
                found = True
                loras = mod._online_lora_patches
                assert "weight" in loras
                assert len(loras["weight"]) > 0
        assert found

    def test_remove_clears_patches(self):
        """remove_online_lora should delete all _online_lora_patches."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        apply_online_lora(model, lora_sd, strength=1.0)
        remove_online_lora(model)

        for _name, mod in model.named_modules():
            assert not hasattr(mod, "_online_lora_patches")

    def test_apply_does_not_modify_weights(self):
        """Online LoRA should NOT change the actual weight parameters."""
        model = _make_small_model()
        lora_sd = _make_lora_state_dict()

        original_w0 = model[0].weight.data.clone()
        apply_online_lora(model, lora_sd, strength=1.0)

        torch.testing.assert_close(model[0].weight.data, original_w0)

    def test_remove_without_apply(self):
        """remove_online_lora without prior apply should be a no-op."""
        model = _make_small_model()
        remove_online_lora(model)  # Should not raise
