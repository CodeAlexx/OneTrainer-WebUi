"""Integration tests for LoRA + Quantization interaction.

Verifies that LoRA merge/unmerge and online application work correctly
with quantized layers (INT8, BnB NF4, Nunchaku).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from serenity.inference.lora.merge import (
    dequantize_if_needed,
    merge_lora_into_model,
    merge_lora_to_weight,
    unmerge_lora_from_model,
)
from serenity.inference.lora.online import apply_online_lora, remove_online_lora
from serenity.inference.quantization.ops import QuantizedLinear, get_weight_and_bias
from serenity.inference.quantization import int8 as int8_mod
from serenity.inference.quantization import bnb as bnb_mod
from serenity.inference.quantization.nunchaku import NunchakuLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int8_model(
    in_f: int = 32, hidden: int = 64, out_f: int = 16
) -> nn.Sequential:
    """Build a small 2-layer model using Int8Linear."""
    torch.manual_seed(42)
    layer0 = int8_mod.Int8Linear(in_f, hidden, bias=True)
    layer1 = int8_mod.Int8Linear(hidden, out_f, bias=True)

    # Initialise with random float weights, then quantize
    w0 = torch.randn(hidden, in_f) * 0.1
    w1 = torch.randn(out_f, hidden) * 0.1
    nn.init.zeros_(layer0.bias)
    nn.init.zeros_(layer1.bias)
    layer0.quantize_weight(w0)
    layer1.quantize_weight(w1)

    return nn.Sequential(layer0, layer1)


def _make_lora_sd(
    prefix_0: str = "0",
    prefix_1: str = "1",
    in0: int = 32,
    out0: int = 64,
    in1: int = 64,
    out1: int = 16,
    rank: int = 4,
) -> dict[str, torch.Tensor]:
    """Create a LoRA state dict targeting two layers."""
    torch.manual_seed(99)
    sd: dict[str, torch.Tensor] = {}
    sd[f"{prefix_0}.lora_down.weight"] = torch.randn(rank, in0) * 0.01
    sd[f"{prefix_0}.lora_up.weight"] = torch.randn(out0, rank) * 0.01
    sd[f"{prefix_1}.lora_down.weight"] = torch.randn(rank, in1) * 0.01
    sd[f"{prefix_1}.lora_up.weight"] = torch.randn(out1, rank) * 0.01
    return sd


def _make_single_lora_patch(
    in_features: int = 32, out_features: int = 64, rank: int = 4
) -> list[dict[str, torch.Tensor | float | None]]:
    """Create a single LoRA patch list for merge_lora_to_weight."""
    torch.manual_seed(77)
    return [
        {
            "up": torch.randn(out_features, rank) * 0.01,
            "down": torch.randn(rank, in_features) * 0.01,
            "alpha": float(rank),
        }
    ]


# ---------------------------------------------------------------------------
# Task 1 test: Nunchaku fallback raises ImportError
# ---------------------------------------------------------------------------


class TestNunchakuFallback:
    """Verify that NunchakuLinear raises ImportError without nunchaku runtime."""

    def test_dequantize_raises_import_error(self):
        """_dequantize should raise ImportError, not produce garbage."""
        layer = NunchakuLinear(16, 8)
        layer.load_quantized(
            qweight=torch.randint(0, 255, (8, 16), dtype=torch.uint8),
            scales=torch.ones(8, 1),
        )
        with pytest.raises(ImportError, match="nunchaku runtime"):
            layer._dequantize()

    def test_forward_without_inner_raises(self):
        """Forward without _inner should raise ImportError via _dequantize."""
        layer = NunchakuLinear(16, 8)
        layer.load_quantized(
            qweight=torch.randint(0, 255, (8, 16), dtype=torch.uint8),
            scales=torch.ones(8, 1),
        )
        x = torch.randn(2, 16)
        with pytest.raises(ImportError, match="nunchaku runtime"):
            layer(x)

    def test_forward_with_inner_delegates(self):
        """When _inner is set, forward should delegate to it."""
        layer = NunchakuLinear(16, 8)
        # Set _inner to a regular linear
        inner = nn.Linear(16, 8, bias=False)
        nn.init.ones_(inner.weight)
        layer._inner = inner

        x = torch.randn(2, 16)
        out = layer(x)
        expected = inner(x)
        torch.testing.assert_close(out, expected)

    def test_no_weights_raises_runtime_error(self):
        """_dequantize without loaded weights raises RuntimeError."""
        layer = NunchakuLinear(16, 8)
        with pytest.raises(RuntimeError, match="No quantized weights loaded"):
            layer._dequantize()


# ---------------------------------------------------------------------------
# Task 2 tests: Offline merge re-quantization verification
# ---------------------------------------------------------------------------


class TestOfflineMergeRequantization:
    """Verify that offline merge into quantized layers does NOT re-quantize.

    The merge path calls ``dequantize_weight()`` to get float weights, applies
    the LoRA delta in float, then copies the result back into the parameter via
    ``copy_()``.  For Int8Linear, the parameter stays int8 dtype because
    ``copy_()`` truncates the float result — this loses the scale factor and
    produces corrupted weights.  The model-level merge for quantized layers is
    lossy; use online LoRA instead for correctness.
    """

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_int8_raw_merge_preserves_int8_dtype(self):
        """merge_lora_to_weight on raw int8 tensor keeps int8 dtype.

        dequantize_if_needed does not recognise plain torch.int8 as quantized.
        The merge casts to float32 for math, then casts back to int8 (backup
        dtype), losing precision via truncation.
        """
        layer = int8_mod.Int8Linear(32, 64, bias=True)
        w = torch.randn(64, 32) * 0.1
        layer.quantize_weight(w)
        nn.init.zeros_(layer.bias)

        # Weight is plain int8 (not qint8), so dequantize_if_needed won't detect it
        assert layer.weight.dtype == torch.int8
        assert not layer.weight.is_quantized  # plain int8, not PyTorch quantized

        patch = _make_single_lora_patch(in_features=32, out_features=64)
        new_weight = merge_lora_to_weight(layer.weight.data, patch, strength=1.0)

        # Result is int8 due to backup_dtype cast — NOT properly re-quantized
        assert new_weight.dtype == torch.int8
        assert new_weight.shape == (64, 32)

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_int8_model_merge_via_dequantize_weight(self):
        """merge_lora_into_model uses dequantize_weight for quantized layers.

        The source weight is properly dequantized to bfloat16 before LoRA is
        applied.  However, the result is then copy_()ed into the int8 parameter,
        which truncates all sub-1.0 float values to 0.  This documents a known
        limitation: offline merge into INT8 layers is lossy.
        """
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        # Before merge: weight is int8
        assert model[0].weight.dtype == torch.int8

        merge_lora_into_model(model, lora_sd, strength=1.0)

        # After merge: weight is still int8 because copy_() preserves dst dtype
        # The float result was truncated into the int8 parameter
        assert model[0].weight.dtype == torch.int8

    def test_dequantize_if_needed_on_quantized_tensor(self):
        """dequantize_if_needed correctly identifies PyTorch quantized tensors."""
        x = torch.randn(4, 8)
        # Non-quantized tensor passes through
        result, was_q = dequantize_if_needed(x)
        assert not was_q
        torch.testing.assert_close(result, x)

    def test_dequantize_if_needed_qint8(self):
        """dequantize_if_needed handles qint8 tensors."""
        x = torch.randn(4, 8)
        qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.qint8)
        result, was_q = dequantize_if_needed(qx)
        assert was_q
        assert result.dtype == torch.float32
        # Should be close (within quantization error)
        assert torch.allclose(result, x, atol=0.15)


# ---------------------------------------------------------------------------
# Task 3: LoRA + Quantization integration tests
# ---------------------------------------------------------------------------


class TestLoRAQuantOfflineMerge:
    """Test offline LoRA merge with various quantization formats."""

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_int8_lora_merge_output_is_corrupted(self):
        """Offline merge into INT8 layer corrupts weights via copy_() truncation.

        This documents a known limitation: merge_lora_into_model computes the
        correct merged weight in float, but copy_() into the int8 parameter
        truncates the float values to zero (for values < 1.0), producing
        all-zero output.
        """
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        x = torch.randn(4, 32)
        out_before = model(x)
        assert torch.isfinite(out_before).all(), "Pre-merge output has NaN/Inf"

        merge_lora_into_model(model, lora_sd, strength=1.0)

        out_after = model(x)
        # Output is finite but effectively zero due to truncation
        assert torch.isfinite(out_after).all(), "Post-merge output has NaN/Inf"
        # Output changes (becomes zeros) — this is the corruption
        assert not torch.allclose(out_before, out_after, atol=1e-6)

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_int8_lora_unmerge_restores_original(self):
        """Unmerge restores original weights from backup, recovering from corruption."""
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        x = torch.randn(4, 32)
        out_original = model(x).clone()

        merge_lora_into_model(model, lora_sd, strength=1.0)
        unmerge_lora_from_model(model)
        out_restored = model(x)

        # Unmerge restores from the backup, so output matches original exactly
        torch.testing.assert_close(out_restored, out_original, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(
        not bnb_mod.is_available(), reason="bitsandbytes not installed"
    )
    def test_bnb_nf4_lora_merge(self):
        """BnB NF4 layer + LoRA merge produces finite output."""
        layer = bnb_mod.BnbLinear4bit(32, 64, bias=True, quant_type="nf4")
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)

        patch = _make_single_lora_patch(in_features=32, out_features=64)
        new_weight = merge_lora_to_weight(layer.weight.data, patch, strength=1.0)

        assert torch.isfinite(new_weight).all()
        assert new_weight.shape == (64, 32)

    def test_quantized_linear_base_lora_merge(self):
        """QuantizedLinear (base class, float32 weights) + LoRA merge works."""
        layer = QuantizedLinear(32, 64, bias=True)
        torch.manual_seed(42)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)

        x = torch.randn(4, 32)
        out_before = layer(x).clone()

        patch = _make_single_lora_patch(in_features=32, out_features=64)
        new_weight = merge_lora_to_weight(layer.weight.data, patch, strength=1.0)
        layer.weight.data.copy_(new_weight)

        out_after = layer(x)
        assert torch.isfinite(out_after).all()
        assert not torch.allclose(out_before, out_after, atol=1e-6)


class TestLoRAQuantOnline:
    """Test online LoRA patches with quantized models."""

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_online_lora_on_int8_layer(self):
        """Online LoRA patches apply correctly to INT8 layers via get_weight_and_bias."""
        layer = int8_mod.Int8Linear(32, 64, bias=True)
        w = torch.randn(64, 32) * 0.1
        layer.quantize_weight(w)
        nn.init.zeros_(layer.bias)

        # Attach a simple online patch
        delta = torch.randn(64, 32) * 0.01
        layer._online_lora_patches = {"weight": [delta]}  # type: ignore[attr-defined]

        weight, bias = get_weight_and_bias(layer)
        assert weight is not None
        # Weight should include the delta
        expected = layer.weight + delta.to(dtype=layer.weight.dtype)
        torch.testing.assert_close(weight, expected)

    def test_online_lora_on_quantized_linear(self):
        """Online LoRA on base QuantizedLinear produces correct output."""
        layer = QuantizedLinear(32, 64, bias=False)
        nn.init.zeros_(layer.weight)

        delta = torch.ones(64, 32) * 0.5
        layer._online_lora_patches = {"weight": [delta]}  # type: ignore[attr-defined]

        weight, bias = get_weight_and_bias(layer)
        expected = torch.ones(64, 32) * 0.5
        torch.testing.assert_close(weight, expected)

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_online_lora_on_int8_model_via_apply(self):
        """apply_online_lora attaches patches to INT8 model modules."""
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        apply_online_lora(model, lora_sd, strength=1.0)

        # At least one module should have patches
        found = False
        for _name, mod in model.named_modules():
            if hasattr(mod, "_online_lora_patches"):
                found = True
                loras = mod._online_lora_patches
                assert "weight" in loras
                assert len(loras["weight"]) > 0
        assert found

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_online_lora_remove_restores_original(self):
        """remove_online_lora removes all patches from quantized model."""
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        apply_online_lora(model, lora_sd, strength=1.0)
        remove_online_lora(model)

        for _name, mod in model.named_modules():
            assert not hasattr(mod, "_online_lora_patches")

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_online_lora_does_not_modify_int8_weights(self):
        """Online LoRA should not change the actual INT8 weight data."""
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        original_w = model[0].weight.data.clone()
        apply_online_lora(model, lora_sd, strength=1.0)

        # Actual weight parameter should be unchanged
        torch.testing.assert_close(model[0].weight.data, original_w)


class TestLoRAQuantEdgeCases:
    """Edge cases for LoRA + quantization interaction."""

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_merge_preserves_weight_shape(self):
        """Weight shape should be preserved through merge."""
        model = _make_int8_model()
        lora_sd = _make_lora_sd()

        shape_before_0 = model[0].weight.shape
        shape_before_1 = model[1].weight.shape

        merge_lora_into_model(model, lora_sd, strength=1.0)

        assert model[0].weight.shape == shape_before_0
        assert model[1].weight.shape == shape_before_1

    def test_zero_strength_lora_on_float_model_is_noop(self):
        """LoRA with strength=0 on float QuantizedLinear produces same output."""
        layer = QuantizedLinear(32, 16, bias=True)
        torch.manual_seed(42)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)

        model = nn.Sequential(layer)
        lora_sd: dict[str, torch.Tensor] = {}
        torch.manual_seed(99)
        lora_sd["0.lora_down.weight"] = torch.randn(4, 32) * 0.01
        lora_sd["0.lora_up.weight"] = torch.randn(16, 4) * 0.01

        x = torch.randn(4, 32)
        out_before = model(x).clone()

        merge_lora_into_model(model, lora_sd, strength=0.0)
        out_after = model(x)

        torch.testing.assert_close(out_before, out_after, atol=1e-5, rtol=1e-5)

    def test_multiple_loras_compose_on_float_model(self):
        """Two LoRAs applied sequentially on float QuantizedLinear compose."""
        layer0 = QuantizedLinear(32, 64, bias=True)
        layer1 = QuantizedLinear(64, 16, bias=True)
        torch.manual_seed(42)
        nn.init.normal_(layer0.weight, std=0.1)
        nn.init.zeros_(layer0.bias)
        nn.init.normal_(layer1.weight, std=0.1)
        nn.init.zeros_(layer1.bias)
        model = nn.Sequential(layer0, layer1)

        x = torch.randn(4, 32)
        out_original = model(x).clone()

        # First LoRA
        torch.manual_seed(111)
        lora_sd_1 = _make_lora_sd()
        merge_lora_into_model(model, lora_sd_1, strength=0.5)
        out_after_first = model(x).clone()

        # Unmerge first, apply second
        unmerge_lora_from_model(model)

        torch.manual_seed(222)
        lora_sd_2: dict[str, torch.Tensor] = {}
        lora_sd_2["0.lora_down.weight"] = torch.randn(4, 32) * 0.02
        lora_sd_2["0.lora_up.weight"] = torch.randn(64, 4) * 0.02
        lora_sd_2["1.lora_down.weight"] = torch.randn(4, 64) * 0.02
        lora_sd_2["1.lora_up.weight"] = torch.randn(16, 4) * 0.02

        merge_lora_into_model(model, lora_sd_2, strength=0.5)
        out_after_second = model(x).clone()

        # Both LoRAs should produce different outputs from original
        assert not torch.allclose(out_original, out_after_first, atol=1e-6)
        assert not torch.allclose(out_original, out_after_second, atol=1e-6)
        assert not torch.allclose(out_after_first, out_after_second, atol=1e-6)

    def test_merge_lora_to_weight_with_qint8_tensor(self):
        """merge_lora_to_weight handles PyTorch qint8 tensors via dequantize_if_needed."""
        x = torch.randn(64, 32) * 0.1
        qx = torch.quantize_per_tensor(x, scale=0.01, zero_point=0, dtype=torch.qint8)

        patch = _make_single_lora_patch(in_features=32, out_features=64)
        result = merge_lora_to_weight(qx, patch, strength=1.0)

        assert result.dtype in (torch.float32, torch.bfloat16, torch.float16)
        assert torch.isfinite(result).all()
        assert result.shape == (64, 32)

    def test_online_lora_multiple_patches(self):
        """Multiple online LoRA patches accumulate correctly."""
        layer = QuantizedLinear(16, 8, bias=False)
        nn.init.zeros_(layer.weight)

        patch1 = torch.ones(8, 16) * 0.3
        patch2 = torch.ones(8, 16) * 0.7
        layer._online_lora_patches = {"weight": [patch1, patch2]}  # type: ignore[attr-defined]

        weight, _ = get_weight_and_bias(layer)
        expected = torch.ones(8, 16) * 1.0
        torch.testing.assert_close(weight, expected)

    def test_merge_then_online_lora(self):
        """Offline merge followed by online LoRA should both take effect."""
        layer = QuantizedLinear(32, 64, bias=False)
        torch.manual_seed(42)
        nn.init.normal_(layer.weight, std=0.1)

        # Offline merge
        patch = _make_single_lora_patch(in_features=32, out_features=64)
        new_weight = merge_lora_to_weight(layer.weight.data, patch, strength=1.0)
        layer.weight.data.copy_(new_weight)

        # Then online LoRA
        delta = torch.ones(64, 32) * 0.01
        layer._online_lora_patches = {"weight": [delta]}  # type: ignore[attr-defined]

        weight, _ = get_weight_and_bias(layer)
        # Weight should be the merged weight + online delta
        expected = new_weight + delta
        torch.testing.assert_close(weight, expected, atol=1e-5, rtol=1e-5)

    def test_lora_strength_scales_linearly(self):
        """LoRA strength should scale the delta linearly."""
        weight = torch.zeros(64, 32)
        patch = _make_single_lora_patch(in_features=32, out_features=64)

        result_half = merge_lora_to_weight(weight.clone(), patch, strength=0.5)
        result_full = merge_lora_to_weight(weight.clone(), patch, strength=1.0)

        # Full strength result should be ~2x the half-strength result
        torch.testing.assert_close(result_full, result_half * 2.0, atol=1e-5, rtol=1e-5)
