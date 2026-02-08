"""SageAttention backend — optional high-performance attention."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass

__all__ = ["attention_sage", "is_available"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_SAGE_AVAILABLE: bool | None = None
_IS_SAGE_2: bool = False
_sageattn = None


def _probe() -> bool:
    """Probe for sageattention at import time (lazy, runs once)."""
    global _SAGE_AVAILABLE, _IS_SAGE_2, _sageattn

    if _SAGE_AVAILABLE is not None:
        return _SAGE_AVAILABLE

    try:
        import importlib.metadata as _meta

        version = _meta.version("sageattention")
        if version.startswith("1"):
            _IS_SAGE_2 = False
        else:
            _IS_SAGE_2 = True

        from sageattention import sageattn  # type: ignore[import-untyped]

        _sageattn = sageattn
        _SAGE_AVAILABLE = True
    except Exception:
        _SAGE_AVAILABLE = False

    return _SAGE_AVAILABLE


def is_available() -> bool:
    """Return True if sageattention is installed and importable."""
    return _probe()


# ---------------------------------------------------------------------------
# Attention function
# ---------------------------------------------------------------------------


def attention_sage(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute attention via SageAttention.

    Falls back to SDP when dim_head is unsupported by the installed version.

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

    # Check dim_head restrictions
    if _IS_SAGE_2:
        unsupported = (dim_head % 8 != 0) or (dim_head > 128)
    else:
        unsupported = dim_head not in (64, 96, 128)

    if unsupported:
        logger.debug(
            "SageAttention: dim_head=%d unsupported (sage_v2=%s), falling back to SDP",
            dim_head,
            _IS_SAGE_2,
        )
        return attention_sdp(q, k, v, heads, mask=mask)

    # Reshape to (batch, seq, heads, dim_head) — NHD layout
    q = q.view(b, seq_len, heads, dim_head)
    k = k.view(b, seq_len, heads, dim_head)
    v = v.view(b, seq_len, heads, dim_head)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        assert _sageattn is not None
        out = _sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout="NHD")
    except Exception as exc:
        logger.warning("SageAttention error: %s — falling back to SDP", exc)
        # Undo reshape for SDP fallback
        q = q.reshape(b, seq_len, inner_dim)
        k = k.reshape(b, seq_len, inner_dim)
        v = v.reshape(b, seq_len, inner_dim)
        return attention_sdp(q, k, v, heads, mask=mask)

    # Reshape back to (batch, seq, heads * dim_head)
    out = out.reshape(b, seq_len, inner_dim)
    return out
