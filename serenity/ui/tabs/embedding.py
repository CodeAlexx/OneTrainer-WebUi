"""Embedding / textual-inversion settings tab for the Serenity training UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_float,
    labeled_separator,
)

__all__ = ["build_embedding_tab"]

_EMBEDDING_WEIGHT_DTYPES = ["FLOAT_32", "BFLOAT_16"]


def build_embedding_tab(ui_state: UIState) -> None:
    """Build the Embedding / textual-inversion settings tab."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Embedding Training -------------------------------------------------
    labeled_separator("Embedding Training")

    labeled_float(
        "Embedding Learning Rate",
        tag="emb_tab_learning_rate",
        default_value=cfg.embedding_learning_rate or 0.0,
        callback=cb("emb_tab_learning_rate"),
        tip="Learning rate for embedding / textual-inversion training",
        format_str="%.2e",
    )
    labeled_checkbox(
        "Preserve Embedding Norm",
        tag="emb_tab_preserve_norm",
        default_value=cfg.preserve_embedding_norm,
        callback=cb("emb_tab_preserve_norm"),
        tip="Normalize embeddings to preserve their original magnitude",
    )
    labeled_combo(
        "Embedding Weight Data Type",
        items=_EMBEDDING_WEIGHT_DTYPES,
        tag="embedding_weight_dtype",
        default_value=cfg.embedding_weight_dtype.value
        if hasattr(cfg.embedding_weight_dtype, "value")
        else str(cfg.embedding_weight_dtype),
        callback=cb("embedding_weight_dtype"),
        tip="Data type for trained embedding weights",
    )

    # -- Register all bindings with UIState ---------------------------------
    ui_state.register("emb_tab_learning_rate", field_path="embedding_learning_rate")
    ui_state.register("emb_tab_preserve_norm", field_path="preserve_embedding_norm")
    ui_state.register("embedding_weight_dtype")
