"""PyTorch scaled dot-product attention (SDP) backend and einsum fallback."""

from __future__ import annotations

import logging
import math
import platform
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    pass

__all__ = ["attention_sdp", "attention_einsum", "is_available"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NVIDIA GPUs have a batch limit for SDP; non-NVIDIA can go higher.
_SDP_BATCH_LIMIT: int = 2**15 if torch.cuda.is_available() else 2**31

# ---------------------------------------------------------------------------
# Windows SDP backend priority
# ---------------------------------------------------------------------------

_sdp_context = None

try:
    if torch.cuda.is_available() and platform.system() == "Windows":
        import inspect

        from torch.nn.attention import SDPBackend, sdpa_kernel

        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            _priority = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
            if hasattr(SDPBackend, "CUDNN_ATTENTION"):
                _priority.insert(0, SDPBackend.CUDNN_ATTENTION)
            _sdp_context = lambda: sdpa_kernel(_priority, set_priority=True)  # noqa: E731
except Exception:
    pass


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """SDP is always available with modern PyTorch (>= 2.0)."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# SDP attention
# ---------------------------------------------------------------------------


def _sdpa(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    """Thin wrapper around F.scaled_dot_product_attention with optional context."""
    if _sdp_context is not None:
        with _sdp_context():
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
            )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
    )


def attention_sdp(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute attention via PyTorch scaled dot-product attention.

    Handles batching for NVIDIA GPUs where batch must stay under 2^15.

    Args:
        q: Query tensor ``(batch, seq_len, heads * dim_head)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.
        mask: Optional attention mask.

    Returns:
        Output tensor ``(batch, seq_len, heads * dim_head)``.
    """
    b, seq_len, inner_dim = q.shape
    dim_head = inner_dim // heads

    # Reshape to (batch, heads, seq, dim_head)
    q = q.view(b, seq_len, heads, dim_head).transpose(1, 2)
    k = k.view(b, seq_len, heads, dim_head).transpose(1, 2)
    v = v.view(b, seq_len, heads, dim_head).transpose(1, 2)

    # Prepare mask
    attn_mask: Tensor | None = None
    if mask is not None:
        attn_mask = mask
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

    if _SDP_BATCH_LIMIT >= b:
        out = _sdpa(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, seq_len, inner_dim)
    else:
        # Chunk along batch dimension to respect NVIDIA limits
        out = torch.empty(
            (b, seq_len, inner_dim), dtype=q.dtype, layout=q.layout, device=q.device,
        )
        for i in range(0, b, _SDP_BATCH_LIMIT):
            end = min(i + _SDP_BATCH_LIMIT, b)
            m = attn_mask
            if m is not None and m.shape[0] > 1:
                m = m[i:end]
            chunk_out = _sdpa(q[i:end], k[i:end], v[i:end], attn_mask=m)
            out[i:end] = chunk_out.transpose(1, 2).reshape(-1, seq_len, inner_dim)

    return out


# ---------------------------------------------------------------------------
# Einsum fallback (pure PyTorch, works everywhere)
# ---------------------------------------------------------------------------


def attention_einsum(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Pure einsum attention — ultimate fallback, no external deps.

    Uses chunked computation to prevent OOM on large sequences.

    Args:
        q: Query tensor ``(batch, seq_len, heads * dim_head)``.
        k: Key tensor, same shape.
        v: Value tensor, same shape.
        heads: Number of attention heads.
        mask: Optional attention mask.

    Returns:
        Output tensor ``(batch, seq_len, heads * dim_head)``.
    """
    b, seq_len, inner_dim = q.shape
    dim_head = inner_dim // heads
    scale = dim_head ** -0.5

    # Reshape to (batch * heads, seq, dim_head)
    q = (
        q.unsqueeze(3)
        .reshape(b, seq_len, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, seq_len, dim_head)
        .contiguous()
    )
    k = (
        k.unsqueeze(3)
        .reshape(b, seq_len, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, seq_len, dim_head)
        .contiguous()
    )
    v = (
        v.unsqueeze(3)
        .reshape(b, seq_len, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, seq_len, dim_head)
        .contiguous()
    )

    # Compute attention scores via einsum
    sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

    del q, k

    # Apply mask
    if mask is not None:
        if mask.dtype == torch.bool:
            mask_flat = mask.reshape(mask.shape[0], -1) if mask.ndim > 2 else mask
            max_neg = -torch.finfo(sim.dtype).max
            mask_expanded = mask_flat.unsqueeze(1).expand(-1, heads, -1).reshape(b * heads, 1, -1)
            sim.masked_fill_(~mask_expanded, max_neg)
        else:
            if mask.ndim == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            m = (
                mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
                .expand(b, heads, -1, -1)
                .reshape(b * heads, mask.shape[-2], mask.shape[-1])
            )
            sim = sim + m

    sim = sim.softmax(dim=-1)

    # Weighted sum
    out = torch.einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)

    # Reshape back to (batch, seq, heads * dim_head)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, seq_len, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, seq_len, inner_dim)
    )

    return out
