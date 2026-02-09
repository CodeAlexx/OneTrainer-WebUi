"""SageAttention backend — optional high-performance attention."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    pass

__all__ = [
    "SageAttnMode",
    "attention_sage",
    "get_sage_mode",
    "is_available",
    "set_sage_mode",
]

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
    except ImportError:
        _SAGE_AVAILABLE = False

    return _SAGE_AVAILABLE


def is_available() -> bool:
    """Return True if sageattention is installed and importable."""
    return _probe()


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


class SageAttnMode(str, Enum):
    """SageAttention implementation mode."""

    AUTO = "auto"
    FP16_TRITON = "fp16_triton"
    FP16_CUDA = "fp16_cuda"
    FP8_CUDA = "fp8_cuda"


# Module-level mode setting
_sage_mode: SageAttnMode = SageAttnMode.AUTO


def set_sage_mode(mode: SageAttnMode | str) -> None:
    """Set the SageAttention implementation mode."""
    global _sage_mode
    _sage_mode = SageAttnMode(mode)


def get_sage_mode() -> SageAttnMode:
    """Get the current SageAttention mode."""
    return _sage_mode


def _get_sage_fn() -> Callable:
    """Get the appropriate SageAttention function for current mode."""
    mode = _sage_mode
    if mode == SageAttnMode.AUTO:
        from sageattention import sageattn  # type: ignore[import-untyped]

        return sageattn

    try:
        if mode == SageAttnMode.FP16_TRITON:
            from sageattention import sageattn_qk_int8_pv_fp16_triton  # type: ignore[import-untyped]

            return sageattn_qk_int8_pv_fp16_triton
        elif mode == SageAttnMode.FP16_CUDA:
            from sageattention import sageattn_qk_int8_pv_fp16_cuda  # type: ignore[import-untyped]

            return sageattn_qk_int8_pv_fp16_cuda
        elif mode == SageAttnMode.FP8_CUDA:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda  # type: ignore[import-untyped]

            return sageattn_qk_int8_pv_fp8_cuda
    except ImportError:
        logger.warning("SageAttention mode %s not available, falling back to auto", mode)
        from sageattention import sageattn  # type: ignore[import-untyped]

        return sageattn

    from sageattention import sageattn  # type: ignore[import-untyped]

    return sageattn


# ---------------------------------------------------------------------------
# Attention function
# ---------------------------------------------------------------------------


def attention_sage(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
    *,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
) -> Tensor:
    """Compute attention via SageAttention.

    Falls back to SDP when dim_head is unsupported by the installed version.

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
        return attention_sdp(
            q, k, v, heads, mask=mask,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
        )

    # Reshape to (batch, seq, heads, dim_head) -- NHD layout
    if not skip_reshape:
        q = q.view(b, seq_len, heads, dim_head)
        k = k.view(b, seq_len, heads, dim_head)
        v = v.view(b, seq_len, heads, dim_head)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        # Use mode-selected function when a non-auto mode is set,
        # otherwise fall back to the probed default for performance.
        if _sage_mode != SageAttnMode.AUTO:
            sage_fn = _get_sage_fn()
        else:
            assert _sageattn is not None
            sage_fn = _sageattn
        out = sage_fn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout="NHD")
    except Exception as exc:
        logger.warning("SageAttention error: %s -- falling back to SDP", exc)
        # Undo reshape for SDP fallback
        if not skip_reshape:
            q = q.reshape(b, seq_len, inner_dim)
            k = k.reshape(b, seq_len, inner_dim)
            v = v.reshape(b, seq_len, inner_dim)
        return attention_sdp(
            q, k, v, heads, mask=mask,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
        )

    # Reshape back to (batch, seq, heads * dim_head)
    if not skip_output_reshape:
        out = out.reshape(b, seq_len, inner_dim)
    return out
