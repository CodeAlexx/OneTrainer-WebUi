"""Attention backend system for the Serenity inference engine."""

from __future__ import annotations

from serenity.inference.attention.backends import (
    AttentionType,
    attention,
    detect_available_backends,
    get_attention_fn,
    select_best_backend,
)

__all__ = [
    "AttentionType",
    "attention",
    "detect_available_backends",
    "get_attention_fn",
    "select_best_backend",
]
