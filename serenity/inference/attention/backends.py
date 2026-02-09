"""Attention backend detection, selection, and dispatch."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "AttentionType",
    "attention",
    "detect_available_backends",
    "get_attention_fn",
    "select_best_backend",
    "vae_attention",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class AttentionType(str, Enum):
    """Available attention backend types, ordered by priority."""

    SAGE = "sage"
    FLASH = "flash"
    XFORMERS = "xformers"
    SDP = "sdp"
    EINSUM = "einsum"


# Priority order for automatic selection (best first).
_PRIORITY: list[AttentionType] = [
    AttentionType.SAGE,
    AttentionType.FLASH,
    AttentionType.XFORMERS,
    AttentionType.SDP,
    AttentionType.EINSUM,
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_available_backends() -> list[AttentionType]:
    """Probe which attention backends are usable on this system.

    Returns:
        Sorted list (best-first) of available backends.
        Always includes at least SDP and EINSUM.
    """
    from serenity.inference.attention import flash as _flash
    from serenity.inference.attention import sage as _sage
    from serenity.inference.attention import sdp as _sdp
    from serenity.inference.attention import xformers_attn as _xf

    checkers: dict[AttentionType, Callable[[], bool]] = {
        AttentionType.SAGE: _sage.is_available,
        AttentionType.FLASH: _flash.is_available,
        AttentionType.XFORMERS: _xf.is_available,
        AttentionType.SDP: _sdp.is_available,
        AttentionType.EINSUM: lambda: True,  # always available
    }

    available: list[AttentionType] = []
    for backend in _PRIORITY:
        try:
            if checkers[backend]():
                available.append(backend)
        except (ImportError, RuntimeError):
            # Skip backends that can't be imported or have CUDA initialization errors
            pass

    return available


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_best_backend(preferred: str = "auto") -> AttentionType:
    """Select the best available attention backend.

    Args:
        preferred: ``"auto"`` to pick the highest-priority backend, or
            one of ``"sage"``, ``"flash"``, ``"xformers"``, ``"sdp"``,
            ``"einsum"`` to force a specific backend (must be available).

    Returns:
        The selected :class:`AttentionType`.

    Raises:
        ValueError: If *preferred* is not ``"auto"`` and is unavailable.
    """
    available = detect_available_backends()

    if preferred == "auto":
        best = available[0] if available else AttentionType.EINSUM
        logger.info("Auto-selected attention backend: %s", best.value)
        return best

    # Explicit selection
    try:
        requested = AttentionType(preferred)
    except ValueError:
        raise ValueError(
            f"Unknown attention backend {preferred!r}. "
            f"Choose from: {', '.join(t.value for t in AttentionType)}"
        ) from None

    if requested in available:
        logger.info("Using requested attention backend: %s", requested.value)
        return requested

    raise ValueError(
        f"Attention backend {preferred!r} is not available. "
        f"Available: {[b.value for b in available]}"
    )


# ---------------------------------------------------------------------------
# Function lookup
# ---------------------------------------------------------------------------


def get_attention_fn(backend: AttentionType) -> Callable[..., Tensor]:
    """Return the attention callable for a given backend type.

    Args:
        backend: The backend to retrieve.

    Returns:
        A function with signature
        ``(q, k, v, heads, mask=None) -> Tensor``.
    """
    from serenity.inference.attention.flash import attention_flash
    from serenity.inference.attention.sage import attention_sage
    from serenity.inference.attention.sdp import attention_einsum, attention_sdp
    from serenity.inference.attention.xformers_attn import attention_xformers

    _FN_MAP: dict[AttentionType, Callable[..., Tensor]] = {
        AttentionType.SAGE: attention_sage,
        AttentionType.FLASH: attention_flash,
        AttentionType.XFORMERS: attention_xformers,
        AttentionType.SDP: attention_sdp,
        AttentionType.EINSUM: attention_einsum,
    }

    fn = _FN_MAP.get(backend)
    if fn is None:
        raise ValueError(f"No function registered for backend {backend!r}")
    return fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _resolve_auto_backend() -> tuple[AttentionType, Callable[..., Tensor]]:
    """Select and cache the best auto-detected backend (lazy, once)."""
    selected = select_best_backend("auto")
    return selected, get_attention_fn(selected)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
    *,
    backend: str = "auto",
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    attn_precision: Any = None,
) -> Tensor:
    """Unified attention entry point -- dispatches to the best available backend.

    Args:
        q: Query tensor ``(batch, seq_len, heads * dim_head)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.
        mask: Optional attention mask.
        backend: ``"auto"`` or a specific backend name.
        skip_reshape: Skip the initial Q/K/V reshape to multi-head format
            (caller has already reshaped).
        skip_output_reshape: Skip reshaping the output back to
            ``(batch, seq_len, heads * dim_head)``.
        attn_precision: Optional ``torch.dtype`` to cast Q/K/V before
            computation. The output is cast back to the original dtype.

    Returns:
        Output tensor ``(batch, seq_len, heads * dim_head)``.
    """
    # Precision casting
    orig_dtype = q.dtype
    if attn_precision is not None:
        q = q.to(attn_precision)
        k = k.to(attn_precision)
        v = v.to(attn_precision)

    if backend == "auto":
        _selected, fn = _resolve_auto_backend()
        result = fn(
            q, k, v, heads, mask=mask,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
        )
    else:
        selected = select_best_backend(backend)
        fn = get_attention_fn(selected)
        result = fn(
            q, k, v, heads, mask=mask,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
        )

    # Cast back to original dtype if needed
    if attn_precision is not None and result.dtype != orig_dtype:
        result = result.to(orig_dtype)

    return result


# ---------------------------------------------------------------------------
# VAE-specific attention
# ---------------------------------------------------------------------------


def vae_attention(q: Tensor, k: Tensor, v: Tensor, heads: int = 1) -> Tensor:
    """Attention for VAE with ``(B, C, H, W)`` spatial tensors.

    Reshapes internally to ``(B*heads, H*W, C//heads)``, computes
    attention via SDP, and reshapes back to ``(B, C, H, W)``.
    Avoids unnecessary memory overhead during VAE decoding.

    Args:
        q: Query tensor ``(B, C, H, W)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.

    Returns:
        Output tensor ``(B, C, H, W)``.
    """
    B, C, H, W = q.shape
    head_dim = C // heads
    seq_len = H * W

    # Reshape: (B, C, H, W) -> (B*heads, H*W, head_dim)
    q = q.reshape(B * heads, head_dim, seq_len).transpose(1, 2)
    k = k.reshape(B * heads, head_dim, seq_len).transpose(1, 2)
    v = v.reshape(B * heads, head_dim, seq_len).transpose(1, 2)

    # Use SDP (always available, handles VAE well)
    out = F.scaled_dot_product_attention(q, k, v)

    # Reshape back: (B*heads, H*W, head_dim) -> (B, C, H, W)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out
