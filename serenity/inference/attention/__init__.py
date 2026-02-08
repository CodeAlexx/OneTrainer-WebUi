"""Attention backend system for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.attention.backends import (
    AttentionType,
    attention,
    detect_available_backends,
    get_attention_fn,
    select_best_backend,
    vae_attention,
)
from serenity.inference.attention.sage import (
    SageAttnMode,
    get_sage_mode,
    set_sage_mode,
)
from serenity.inference.attention.token_merging import (
    bipartite_soft_matching,
    merge_tokens,
    unmerge_tokens,
)

__all__ = [
    "AttentionType",
    "SageAttnMode",
    "attention",
    "bipartite_soft_matching",
    "detect_available_backends",
    "get_attention_fn",
    "get_sage_mode",
    "merge_tokens",
    "select_best_backend",
    "set_sage_mode",
    "unmerge_tokens",
    "vae_attention",
]
