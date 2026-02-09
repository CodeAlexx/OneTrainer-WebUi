"""xformers memory-efficient attention backend."""

from __future__ import annotations

import logging

import torch
from torch import Tensor

__all__ = ["attention_xformers", "is_available"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_XFORMERS_AVAILABLE: bool | None = None
_xformers_ops = None


def _probe() -> bool:
    """Probe for xformers at import time (lazy, runs once)."""
    global _XFORMERS_AVAILABLE, _xformers_ops

    if _XFORMERS_AVAILABLE is not None:
        return _XFORMERS_AVAILABLE

    try:
        import xformers.ops  # type: ignore[import-untyped]

        _xformers_ops = xformers.ops
        _XFORMERS_AVAILABLE = True
    except ImportError:
        _XFORMERS_AVAILABLE = False

    return _XFORMERS_AVAILABLE


def is_available() -> bool:
    """Return True if xformers is installed and importable."""
    return _probe()


# ---------------------------------------------------------------------------
# Attention function
# ---------------------------------------------------------------------------


def attention_xformers(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
    *,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
) -> Tensor:
    """Compute attention via xformers memory-efficient attention.

    Falls back to SDP on error (e.g. RTX 50 series incompatibility).

    Args:
        q: Query tensor ``(batch, seq_len, heads * dim_head)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.
        mask: Optional attention mask.
        skip_reshape: Skip initial Q/K/V reshape to multi-head format.
        skip_output_reshape: Skip reshaping output back.

    Returns:
        Output tensor ``(batch, seq_len, heads * dim_head)``.
    """
    from serenity.inference.attention.sdp import attention_sdp

    b, seq_len, inner_dim = q.shape
    dim_head = inner_dim // heads

    # Reshape to (batch, seq, heads, dim_head) -- xformers BMHK format
    if not skip_reshape:
        q_x = q.reshape(b, seq_len, heads, dim_head)
        k_x = k.reshape(b, seq_len, heads, dim_head)
        v_x = v.reshape(b, seq_len, heads, dim_head)
    else:
        q_x, k_x, v_x = q, k, v

    # Prepare mask if provided
    attn_bias: Tensor | None = None
    if mask is not None:
        attn_bias = mask
        if attn_bias.ndim == 2:
            attn_bias = attn_bias.unsqueeze(0)
        if attn_bias.ndim == 3:
            attn_bias = attn_bias.unsqueeze(1)
        # Pad to 8-byte alignment for xformers
        pad = 8 - attn_bias.shape[-1] % 8
        if pad < 8:
            padded = torch.empty(
                [attn_bias.shape[0], attn_bias.shape[1], q_x.shape[1], attn_bias.shape[-1] + pad],
                dtype=q.dtype,
                device=q.device,
            )
            padded[..., : attn_bias.shape[-1]] = attn_bias
            attn_bias = padded[..., : mask.shape[-1]]
        attn_bias = attn_bias.expand(b, heads, -1, -1)

    try:
        assert _xformers_ops is not None
        out = _xformers_ops.memory_efficient_attention(q_x, k_x, v_x, attn_bias=attn_bias)
        fallback = False
    except Exception as exc:
        if "(too new)" in str(exc):
            logger.warning("xformers does not support this GPU (RTX 50 series?)")
        else:
            logger.warning("xformers error: %s -- falling back to SDP", exc)
        fallback = True

    if fallback:
        return attention_sdp(
            q, k, v, heads, mask=mask,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
        )

    # Reshape back to (batch, seq, heads * dim_head)
    if not skip_output_reshape:
        out = out.reshape(b, seq_len, inner_dim)
    return out
