"""LoRA / LoKr adapter settings tab for the Serenity training UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_file,
    labeled_float,
    labeled_int,
    labeled_separator,
)

__all__ = ["build_lora_tab"]

_LORA_WEIGHT_DTYPES = ["FLOAT_32", "BFLOAT_16"]


def build_lora_tab(ui_state: UIState) -> None:
    """Build the LoRA / adapter settings tab with peft_type selector."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- PEFT type selector -------------------------------------------------
    labeled_combo(
        "PEFT Type",
        items=["LORA", "LOKR"],
        tag="peft_type",
        default_value=cfg.peft_type.value
        if hasattr(cfg.peft_type, "value")
        else str(cfg.peft_type),
        callback=_make_peft_switch(ui_state),
        tip="Adapter type: LoRA or LoKr",
    )

    # -- LoRA Settings (shown for all peft types) ---------------------------
    labeled_separator("LoRA Settings")

    labeled_file(
        "LoRA Model",
        tag="lora_model_name",
        default_value=cfg.lora_model_name,
        callback=cb("lora_model_name"),
        tip="Path to an existing LoRA to resume or merge",
    )
    labeled_int(
        "Rank",
        tag="lora_rank",
        default_value=cfg.lora_rank,
        callback=cb("lora_rank"),
        tip="LoRA rank (dimensionality of low-rank matrices)",
        min_value=1,
        max_value=1024,
    )
    labeled_float(
        "Alpha",
        tag="lora_alpha",
        default_value=cfg.lora_alpha,
        callback=cb("lora_alpha"),
        tip="LoRA alpha scaling factor",
        format_str="%.2f",
    )
    labeled_float(
        "Dropout",
        tag="dropout_probability",
        default_value=cfg.dropout_probability,
        callback=cb("dropout_probability"),
        tip="Dropout probability applied to LoRA layers",
        format_str="%.3f",
        min_value=0.0,
        max_value=1.0,
    )
    labeled_combo(
        "LoRA Weight Data Type",
        items=_LORA_WEIGHT_DTYPES,
        tag="lora_weight_dtype",
        default_value=cfg.lora_weight_dtype.value
        if hasattr(cfg.lora_weight_dtype, "value")
        else str(cfg.lora_weight_dtype),
        callback=cb("lora_weight_dtype"),
        tip="Data type for LoRA adapter weights",
    )
    labeled_checkbox(
        "Bundle Additional Embeddings",
        tag="bundle_additional_embeddings",
        default_value=cfg.bundle_additional_embeddings,
        callback=cb("bundle_additional_embeddings"),
        tip="Include trained embeddings inside the LoRA file",
    )

    # -- DoRA (LoRA-specific) -----------------------------------------------
    labeled_separator("DoRA")

    labeled_checkbox(
        "Decompose Weights (DoRA)",
        tag="lora_decompose",
        default_value=cfg.lora_decompose,
        callback=cb("lora_decompose"),
        tip="Enable weight decomposition (DoRA) for improved training",
    )
    labeled_checkbox(
        "Norm Epsilon",
        tag="lora_decompose_norm_epsilon",
        default_value=cfg.lora_decompose_norm_epsilon,
        callback=cb("lora_decompose_norm_epsilon"),
        tip="Add epsilon to norm computation for numerical stability",
    )
    labeled_checkbox(
        "Output Axis Decomposition",
        tag="lora_decompose_output_axis",
        default_value=cfg.lora_decompose_output_axis,
        callback=cb("lora_decompose_output_axis"),
        tip="Decompose along the output axis instead of input",
    )

    # -- LoKr Settings (LoKr-specific) --------------------------------------
    labeled_separator("LoKr Settings")

    _lokr_group = dpg.add_group(tag="__lokr_settings_group")

    labeled_int(
        "LoKr Dim",
        tag="lokr_dim",
        default_value=cfg.lokr_dim,
        callback=cb("lokr_dim"),
        parent=_lokr_group,
        tip="LoKr decomposition dimension",
        min_value=1,
        max_value=1024,
    )
    labeled_float(
        "LoKr Alpha",
        tag="lokr_alpha",
        default_value=cfg.lokr_alpha,
        callback=cb("lokr_alpha"),
        parent=_lokr_group,
        tip="LoKr alpha scaling factor",
        format_str="%.2f",
    )
    labeled_checkbox(
        "Decompose Both",
        tag="lokr_decompose_both",
        default_value=cfg.lokr_decompose_both,
        callback=cb("lokr_decompose_both"),
        parent=_lokr_group,
        tip="Apply Kronecker decomposition to both weight matrices",
    )
    labeled_int(
        "Decompose Factor",
        tag="lokr_decompose_factor",
        default_value=cfg.lokr_decompose_factor,
        callback=cb("lokr_decompose_factor"),
        parent=_lokr_group,
        tip="Factor for LoKr decomposition (-1 for auto)",
    )
    labeled_checkbox(
        "Use Tucker",
        tag="lokr_use_tucker",
        default_value=cfg.lokr_use_tucker,
        callback=cb("lokr_use_tucker"),
        parent=_lokr_group,
        tip="Use Tucker decomposition in LoKr",
    )
    labeled_checkbox(
        "Full Matrix",
        tag="lokr_full_matrix",
        default_value=cfg.lokr_full_matrix,
        callback=cb("lokr_full_matrix"),
        parent=_lokr_group,
        tip="Use full-rank matrix for one of the Kronecker factors",
    )
    labeled_checkbox(
        "RS-LoRA",
        tag="lokr_rs_lora",
        default_value=cfg.lokr_rs_lora,
        callback=cb("lokr_rs_lora"),
        parent=_lokr_group,
        tip="Apply rank-stabilized LoRA scaling",
    )
    labeled_checkbox(
        "Weight Decompose",
        tag="lokr_weight_decompose",
        default_value=cfg.lokr_weight_decompose,
        callback=cb("lokr_weight_decompose"),
        parent=_lokr_group,
        tip="Enable weight decomposition (DoRA-style) for LoKr",
    )
    labeled_checkbox(
        "DoRA on Output",
        tag="lokr_dora_on_output",
        default_value=cfg.lokr_dora_on_output,
        callback=cb("lokr_dora_on_output"),
        parent=_lokr_group,
        tip="Apply DoRA decomposition on the output axis",
    )

    # Apply initial visibility based on current peft_type
    _update_peft_visibility(cfg.peft_type)

    # -- Register all bindings with UIState ---------------------------------
    for tag in [
        "peft_type",
        "lora_model_name",
        "lora_rank",
        "lora_alpha",
        "dropout_probability",
        "lora_weight_dtype",
        "bundle_additional_embeddings",
        "lora_decompose",
        "lora_decompose_norm_epsilon",
        "lora_decompose_output_axis",
        "lokr_dim",
        "lokr_alpha",
        "lokr_decompose_both",
        "lokr_decompose_factor",
        "lokr_use_tucker",
        "lokr_full_matrix",
        "lokr_rs_lora",
        "lokr_weight_decompose",
        "lokr_dora_on_output",
    ]:
        ui_state.register(tag)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_peft_visibility(peft_value) -> None:
    """Show/hide LoKr settings group based on the selected PEFT type."""
    val = peft_value.value if hasattr(peft_value, "value") else str(peft_value)
    is_lokr = val.upper() == "LOKR"
    try:
        dpg.configure_item("__lokr_settings_group", show=is_lokr)
    except Exception:
        pass  # Widget may not exist yet during initial build


def _make_peft_switch(ui_state: UIState):
    """Return a DPG callback that updates config and toggles LoKr visibility."""
    def _cb(sender, app_data, user_data):
        ui_state.widget_to_config("peft_type")
        _update_peft_visibility(app_data)
    return _cb
