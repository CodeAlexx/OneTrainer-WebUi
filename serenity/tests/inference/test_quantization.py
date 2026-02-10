"""Tests for serenity.inference.quantization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from serenity.inference.quantization.ops import (
    ManualCastLinear,
    OperationContext,
    QuantizedLinear,
    get_weight_and_bias,
)
from serenity.inference.quantization import int8 as int8_mod
from serenity.inference.quantization import bnb as bnb_mod
from serenity.inference.quantization import fp8 as fp8_mod
from serenity.inference.quantization import gguf as gguf_mod


# ---------------------------------------------------------------------------
# ManualCastLinear
# ---------------------------------------------------------------------------


class TestManualCastLinear:
    def test_output_matches_regular_linear(self):
        """ManualCastLinear should produce the same output as nn.Linear."""
        torch.manual_seed(42)
        ref = nn.Linear(16, 8, bias=True)
        mcl = ManualCastLinear(16, 8, bias=True, manual_cast=True)

        # Copy weights
        mcl.weight.data.copy_(ref.weight.data)
        mcl.bias.data.copy_(ref.bias.data)

        x = torch.randn(4, 16)
        out_ref = ref(x)
        out_mcl = mcl(x)

        torch.testing.assert_close(out_ref, out_mcl, atol=1e-5, rtol=1e-5)

    def test_manual_cast_dtype(self):
        """Weight should be cast to input dtype when manual_cast is True."""
        mcl = ManualCastLinear(8, 4, bias=False, manual_cast=True)
        mcl.weight.data.fill_(1.0)
        x = torch.randn(2, 8, dtype=torch.float16)
        out = mcl(x)
        assert out.dtype == torch.float16

    def test_no_manual_cast(self):
        """When manual_cast is False, plain forward works."""
        mcl = ManualCastLinear(8, 4, bias=True, manual_cast=False)
        x = torch.randn(2, 8)
        out = mcl(x)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# OperationContext
# ---------------------------------------------------------------------------


class TestOperationContext:
    def test_create_linear_default(self):
        """Default context creates nn.Linear."""
        ctx = OperationContext()
        layer = ctx.create_linear(16, 8)
        assert isinstance(layer, nn.Linear)

    def test_create_linear_manual_cast(self):
        """Context with ManualCastLinear cls creates ManualCastLinear."""
        ctx = OperationContext(linear_cls=ManualCastLinear)
        layer = ctx.create_linear(16, 8, bias=True)
        assert isinstance(layer, ManualCastLinear)

    def test_create_conv2d(self):
        ctx = OperationContext()
        layer = ctx.create_conv2d(3, 16, 3, padding=1)
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == 3
        assert layer.out_channels == 16


# ---------------------------------------------------------------------------
# get_weight_and_bias
# ---------------------------------------------------------------------------


class TestGetWeightAndBias:
    def test_no_lora(self):
        """Without online LoRA, returns plain weight and bias."""
        layer = nn.Linear(8, 4)
        w, b = get_weight_and_bias(layer)
        torch.testing.assert_close(w, layer.weight)
        torch.testing.assert_close(b, layer.bias)

    def test_with_scale_weight(self):
        """scale_weight should multiply the weight."""
        layer = nn.Linear(8, 4, bias=False)
        nn.init.ones_(layer.weight)
        layer.scale_weight = torch.tensor(2.0)  # type: ignore[attr-defined]
        w, b = get_weight_and_bias(layer)
        expected = layer.weight * 2.0
        torch.testing.assert_close(w, expected)

    def test_with_online_lora(self):
        """Online LoRA patches should be added to the weight."""
        layer = nn.Linear(8, 4, bias=False)
        nn.init.zeros_(layer.weight)

        patch = torch.ones(4, 8) * 0.5
        layer._online_lora_patches = {"weight": [patch]}  # type: ignore[attr-defined]

        w, b = get_weight_and_bias(layer)
        expected = torch.ones(4, 8) * 0.5
        torch.testing.assert_close(w, expected)

    def test_with_online_lora_bias(self):
        """Online LoRA patches on bias."""
        layer = nn.Linear(8, 4, bias=True)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

        bias_patch = torch.ones(4) * 0.1
        layer._online_lora_patches = {"bias": [bias_patch]}  # type: ignore[attr-defined]

        w, b = get_weight_and_bias(layer)
        torch.testing.assert_close(b, torch.ones(4) * 0.1, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# INT8
# ---------------------------------------------------------------------------


class TestInt8:
    def test_is_available(self):
        result = int8_mod.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not int8_mod.is_available(), reason="torch._int_mm not available"
    )
    def test_int8_linear_forward(self):
        """INT8 forward should produce a result close to float forward."""
        torch.manual_seed(42)
        ref = nn.Linear(64, 32, bias=True)
        q = int8_mod.Int8Linear(64, 32, bias=True)
        q.bias.data.copy_(ref.bias.data)
        q.quantize_weight(ref.weight.data.clone())

        x = torch.randn(8, 64)
        out = q(x)
        out_ref = ref(x)

        # INT8 quantization introduces error, but should be reasonable
        assert out.shape == out_ref.shape
        # Relaxed tolerance for quantization error
        assert torch.allclose(out, out_ref, atol=0.5, rtol=0.2)

    def test_int8_quantize_dequantize(self):
        """Round-trip quantization should be close to original."""
        x = torch.randn(32, 64)
        q, scale = int8_mod.quantize_int8_tensorwise(x)
        x_hat = int8_mod.dequantize_int8(q, scale)
        assert torch.allclose(x, x_hat, atol=0.1)

    def test_int8_linear_not_quantized(self):
        """Unquantized Int8Linear should behave like regular linear."""
        q = int8_mod.Int8Linear(8, 4, bias=True)
        nn.init.ones_(q.weight)
        nn.init.zeros_(q.bias)
        x = torch.randn(2, 8)
        out = q(x)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------


class TestFp8:
    def test_is_available(self):
        result = fp8_mod.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not fp8_mod.is_available() or not torch.cuda.is_available(),
        reason="torch._scaled_mm or CUDA not available",
    )
    def test_fp8_linear_forward(self):
        """FP8 forward should produce a reasonable result."""
        torch.manual_seed(42)
        fp8_layer = fp8_mod.Fp8Linear(64, 32, bias=False)
        # Store weight as fp8
        w = torch.randn(32, 64)
        fp8_layer.weight = nn.Parameter(
            w.to(torch.float8_e4m3fn), requires_grad=False
        )
        fp8_layer.scale_weight = torch.tensor(1.0)

        x = torch.randn(4, 1, 64).cuda()
        fp8_layer = fp8_layer.cuda()
        try:
            out = fp8_layer(x)
        except RuntimeError as e:
            if "compute capability" in str(e) or "MI300" in str(e):
                pytest.skip(f"GPU does not support FP8: {e}")
            raise
        assert out.shape[-1] == 32

    def test_fp8_fallback_on_non_fp8_weight(self):
        """Non-FP8 weight should fall back to regular dequantize path."""
        fp8 = fp8_mod.Fp8Linear(8, 4, bias=True)
        nn.init.ones_(fp8.weight)
        nn.init.zeros_(fp8.bias)
        x = torch.randn(2, 8)
        out = fp8(x)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# BnB
# ---------------------------------------------------------------------------


class TestBnb:
    def test_is_available(self):
        result = bnb_mod.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not bnb_mod.is_available(), reason="bitsandbytes not installed")
    def test_bnb_linear_construction(self):
        """BnbLinear4bit should construct without error."""
        layer = bnb_mod.BnbLinear4bit(16, 8, quant_type="nf4")
        assert layer.quant_type == "nf4"

    def test_bnb_unavailable_raises(self):
        """When bnb is not installed, construction should raise."""
        if bnb_mod.is_available():
            pytest.skip("bitsandbytes IS available")
        with pytest.raises(RuntimeError, match="bitsandbytes"):
            bnb_mod.BnbLinear4bit(16, 8)


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


class TestGGUF:
    def test_is_available(self):
        result = gguf_mod.is_available()
        assert isinstance(result, bool)

    def test_dequantize_tensor_plain(self):
        """Plain tensor (no gguf_cls) should pass through."""
        t = torch.randn(4, 4)
        result = gguf_mod.dequantize_tensor(t)
        torch.testing.assert_close(result, t)

    def test_dequantize_tensor_with_dtype(self):
        """Plain tensor with dtype cast."""
        t = torch.randn(4, 4)
        result = gguf_mod.dequantize_tensor(t, dtype=torch.float16)
        assert result.dtype == torch.float16

    def test_dequantize_tensor_none_raises(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError):
            gguf_mod.dequantize_tensor(None)

    def test_gguf_linear_fallback(self):
        """GGUFLinear with normal weight should work as regular linear."""
        layer = gguf_mod.GGUFLinear(8, 4, bias=True)
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 4)

    def test_bake_gguf_model_no_gguf(self):
        """bake_gguf_model on a regular model should be a no-op."""
        model = nn.Linear(8, 4)
        gguf_mod.bake_gguf_model(model)  # Should not raise
