"""Tests for the attention backend system."""

from __future__ import annotations

import pytest
import torch

from serenity.inference.attention.backends import (
    AttentionType,
    attention,
    detect_available_backends,
    get_attention_fn,
    select_best_backend,
)
from serenity.inference.attention.sdp import attention_einsum, attention_sdp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qkv(
    batch: int = 2,
    seq_len: int = 16,
    heads: int = 4,
    dim_head: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random q/k/v tensors of shape (batch, seq_len, heads * dim_head)."""
    inner_dim = heads * dim_head
    q = torch.randn(batch, seq_len, inner_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, inner_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, inner_dim, dtype=dtype, device=device)
    return q, k, v


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetection:
    """Test backend detection."""

    def test_detect_returns_nonempty(self) -> None:
        """At least SDP and EINSUM should always be available."""
        available = detect_available_backends()
        assert len(available) >= 2
        assert AttentionType.SDP in available
        assert AttentionType.EINSUM in available

    def test_detect_order(self) -> None:
        """Available backends should be in priority order."""
        available = detect_available_backends()
        priority = [
            AttentionType.SAGE,
            AttentionType.FLASH,
            AttentionType.XFORMERS,
            AttentionType.SDP,
            AttentionType.EINSUM,
        ]
        filtered_priority = [b for b in priority if b in available]
        assert available == filtered_priority


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------


class TestSelection:
    """Test backend selection logic."""

    def test_auto_returns_valid(self) -> None:
        best = select_best_backend("auto")
        assert isinstance(best, AttentionType)
        assert best in detect_available_backends()

    def test_force_sdp(self) -> None:
        result = select_best_backend("sdp")
        assert result == AttentionType.SDP

    def test_force_einsum(self) -> None:
        result = select_best_backend("einsum")
        assert result == AttentionType.EINSUM

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown attention backend"):
            select_best_backend("nonexistent")


# ---------------------------------------------------------------------------
# get_attention_fn tests
# ---------------------------------------------------------------------------


class TestGetAttentionFn:
    """Test function lookup."""

    def test_sdp_fn(self) -> None:
        fn = get_attention_fn(AttentionType.SDP)
        assert callable(fn)

    def test_einsum_fn(self) -> None:
        fn = get_attention_fn(AttentionType.EINSUM)
        assert callable(fn)

    def test_all_available_fns_callable(self) -> None:
        for backend in detect_available_backends():
            fn = get_attention_fn(backend)
            assert callable(fn)


# ---------------------------------------------------------------------------
# Functional SDP tests
# ---------------------------------------------------------------------------


class TestSDP:
    """Test the SDP attention function."""

    def test_output_shape(self) -> None:
        q, k, v = _make_qkv()
        out = attention_sdp(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_different_dims(self) -> None:
        """Test with various head/dim configurations."""
        for heads, dim_head in [(1, 64), (8, 16), (4, 32), (2, 128)]:
            q, k, v = _make_qkv(heads=heads, dim_head=dim_head)
            out = attention_sdp(q, k, v, heads=heads)
            assert out.shape == q.shape

    def test_single_batch(self) -> None:
        q, k, v = _make_qkv(batch=1)
        out = attention_sdp(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_deterministic(self) -> None:
        """Same input should produce same output."""
        q, k, v = _make_qkv()
        out1 = attention_sdp(q, k, v, heads=4)
        out2 = attention_sdp(q, k, v, heads=4)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Functional einsum tests
# ---------------------------------------------------------------------------


class TestEinsum:
    """Test the einsum attention fallback."""

    def test_output_shape(self) -> None:
        q, k, v = _make_qkv()
        out = attention_einsum(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_different_dims(self) -> None:
        for heads, dim_head in [(1, 64), (8, 16), (4, 32)]:
            q, k, v = _make_qkv(heads=heads, dim_head=dim_head)
            out = attention_einsum(q, k, v, heads=heads)
            assert out.shape == q.shape

    def test_deterministic(self) -> None:
        q, k, v = _make_qkv()
        out1 = attention_einsum(q, k, v, heads=4)
        out2 = attention_einsum(q, k, v, heads=4)
        torch.testing.assert_close(out1, out2)

    def test_agrees_with_sdp(self) -> None:
        """SDP and einsum should produce approximately the same output."""
        q, k, v = _make_qkv(batch=1, seq_len=8, heads=2, dim_head=16)
        out_sdp = attention_sdp(q, k, v, heads=2)
        out_ein = attention_einsum(q, k, v, heads=2)
        torch.testing.assert_close(out_sdp, out_ein, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Reshape logic test
# ---------------------------------------------------------------------------


class TestReshapeLogic:
    """Verify internal reshape: (b, seq, heads*dim) -> (b, heads, seq, dim) -> back."""

    def test_reshape_roundtrip(self) -> None:
        batch, seq_len, heads, dim_head = 2, 8, 4, 32
        inner_dim = heads * dim_head

        t = torch.randn(batch, seq_len, inner_dim)

        # Forward reshape (what SDP does internally)
        reshaped = t.view(batch, seq_len, heads, dim_head).transpose(1, 2)
        assert reshaped.shape == (batch, heads, seq_len, dim_head)

        # Reverse reshape
        restored = reshaped.transpose(1, 2).reshape(batch, seq_len, inner_dim)
        torch.testing.assert_close(t, restored)


# ---------------------------------------------------------------------------
# Main dispatcher test
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Test the main attention() entry point."""

    def test_auto_dispatch(self) -> None:
        q, k, v = _make_qkv()
        out = attention(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_explicit_sdp(self) -> None:
        q, k, v = _make_qkv()
        out = attention(q, k, v, heads=4, backend="sdp")
        assert out.shape == q.shape

    def test_explicit_einsum(self) -> None:
        q, k, v = _make_qkv()
        out = attention(q, k, v, heads=4, backend="einsum")
        assert out.shape == q.shape


# ---------------------------------------------------------------------------
# Optional backend tests (skip if unavailable)
# ---------------------------------------------------------------------------


class TestOptionalBackends:
    """Test optional backends if they are installed."""

    def test_sage_if_available(self) -> None:
        from serenity.inference.attention.sage import is_available

        if not is_available():
            pytest.skip("sageattention not installed")
        from serenity.inference.attention.sage import attention_sage

        q, k, v = _make_qkv(dim_head=64)
        out = attention_sage(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_flash_if_available(self) -> None:
        from serenity.inference.attention.flash import is_available

        if not is_available():
            pytest.skip("flash_attn not installed")
        from serenity.inference.attention.flash import attention_flash

        q, k, v = _make_qkv()
        out = attention_flash(q, k, v, heads=4)
        assert out.shape == q.shape

    def test_xformers_if_available(self) -> None:
        from serenity.inference.attention.xformers_attn import is_available

        if not is_available():
            pytest.skip("xformers not installed")
        from serenity.inference.attention.xformers_attn import attention_xformers

        q, k, v = _make_qkv()
        out = attention_xformers(q, k, v, heads=4)
        assert out.shape == q.shape
