"""FlashAttention backend — optional CUDA-accelerated attention."""

from __future__ import annotations

import logging

import torch
from torch import Tensor

__all__ = ["attention_flash", "is_available"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_FLASH_AVAILABLE: bool | None = None
_flash_attn_func = None


def _probe() -> bool:
    """Probe for flash_attn at import time (lazy, runs once)."""
    global _FLASH_AVAILABLE, _flash_attn_func

    if _FLASH_AVAILABLE is not None:
        return _FLASH_AVAILABLE

    try:
        from flash_attn import flash_attn_func  # type: ignore[import-untyped]

        _flash_attn_func = flash_attn_func
        _FLASH_AVAILABLE = True
    except Exception:
        _FLASH_AVAILABLE = False

    return _FLASH_AVAILABLE


def is_available() -> bool:
    """Return True if flash_attn is installed and importable."""
    return _probe()


# ---------------------------------------------------------------------------
# Attention function
# ---------------------------------------------------------------------------


def attention_flash(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute attention via FlashAttention.

    Falls back to SDP on error or when a mask is provided (flash_attn does
    not support arbitrary attention masks).

    Args:
        q: Query tensor ``(batch, seq_len, heads * dim_head)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.
        mask: Optional attention mask.

    Returns:
        Output tensor ``(batch, seq_len, heads * dim_head)``.
    """
    from serenity.inference.attention.sdp import attention_sdp

    b, seq_len, inner_dim = q.shape
    dim_head = inner_dim // heads

    # flash_attn expects (batch, seq, heads, dim_head) — no mask support
    if mask is not None:
        logger.debug("FlashAttention does not support masks, falling back to SDP")
        return attention_sdp(q, k, v, heads, mask=mask)

    # Reshape to (batch, seq, heads, dim_head)
    q_f = q.view(b, seq_len, heads, dim_head)
    k_f = k.view(b, seq_len, heads, dim_head)
    v_f = v.view(b, seq_len, heads, dim_head)

    try:
        assert _flash_attn_func is not None
        out = _flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, causal=False)
    except Exception as exc:
        logger.warning("FlashAttention error: %s — falling back to SDP", exc)
        return attention_sdp(q, k, v, heads, mask=mask)

    # Reshape back to (batch, seq, heads * dim_head)
    out = out.reshape(b, seq_len, inner_dim)
    return out
