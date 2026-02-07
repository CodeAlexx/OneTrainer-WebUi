"""Training tab -- optimizer, LR, noise, masking, and loss settings.

Three-column layout mirroring OneTrainer's training tab.  Column 0 covers
optimizer / LR / text-encoder / embedding settings.  Column 1 has EMA,
precision, transformer, and noise.  Column 2 provides masked training,
loss functions, and layer filtering.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    enum_values,
    labeled_checkbox,
    labeled_combo,
    labeled_float,
    labeled_input,
    labeled_int,
    labeled_separator,
    time_entry,
)
from serenity.core.enums import (
    DataType,
    EMAMode,
    GradientCheckpointingMethod,
    LearningRateScaler,
    LearningRateScheduler,
    LossScaler,
    LossWeight,
    Optimizer,
    TimeUnit,
    TimestepDistribution,
)

__all__ = ["build_training_tab"]

# Column widths for the 3-column layout
_COL_W = 420


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum_default(obj: object, attr: str) -> str:
    """Get the ``.value`` of an enum attribute, safely falling back to str."""
    val = getattr(obj, attr, None)
    if val is None:
        return ""
    return val.value if hasattr(val, "value") else str(val)


def _nested(obj: object, *parts: str) -> object:
    """Walk a dotted attribute chain."""
    for p in parts:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _reg(ui: UIState, tag: str) -> None:
    """Shorthand: register a tag with the same field path."""
    ui.register(tag)


# ---------------------------------------------------------------------------
# Column builders
# ---------------------------------------------------------------------------

def _build_col0(parent: int | str, ui: UIState) -> None:
    """Column 0 -- Optimizer, LR, Text Encoder, Embedding."""
    cfg = ui.config

    # ---- Optimizer section ----
    labeled_separator("Optimizer", parent=parent)

    tag = "optimizer.optimizer"
    labeled_combo(
        "Optimizer", enum_values(Optimizer),
        tag=tag,
        default_value=_enum_default(cfg.optimizer, "optimizer"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "learning_rate_scheduler"
    labeled_combo(
        "LR Scheduler", enum_values(LearningRateScheduler),
        tag=tag,
        default_value=_enum_default(cfg, "learning_rate_scheduler"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "learning_rate"
    labeled_float(
        "Learning Rate",
        tag=tag,
        default_value=cfg.learning_rate,
        callback=ui.make_callback(tag),
        parent=parent,
        format_str="%.2e",
    )
    _reg(ui, tag)

    tag = "learning_rate_warmup_steps"
    labeled_float(
        "LR Warmup Steps",
        tag=tag,
        default_value=cfg.learning_rate_warmup_steps,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "learning_rate_min_factor"
    labeled_float(
        "LR Min Factor",
        tag=tag,
        default_value=cfg.learning_rate_min_factor,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "learning_rate_cycles"
    labeled_float(
        "LR Cycles",
        tag=tag,
        default_value=cfg.learning_rate_cycles,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "epochs"
    labeled_int(
        "Epochs",
        tag=tag,
        default_value=cfg.epochs,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "batch_size"
    labeled_int(
        "Batch Size",
        tag=tag,
        default_value=cfg.batch_size,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "gradient_accumulation_steps"
    labeled_int(
        "Gradient Accum Steps",
        tag=tag,
        default_value=cfg.gradient_accumulation_steps,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "learning_rate_scaler"
    labeled_combo(
        "LR Scaler", enum_values(LearningRateScaler),
        tag=tag,
        default_value=_enum_default(cfg, "learning_rate_scaler"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "clip_grad_norm"
    labeled_float(
        "Clip Grad Norm",
        tag=tag,
        default_value=cfg.clip_grad_norm or 1.0,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Text Encoder section ----
    labeled_separator("Text Encoder", parent=parent)

    tag = "text_encoder.train"
    labeled_checkbox(
        "Train Text Encoder",
        tag=tag,
        default_value=cfg.text_encoder.train,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "text_encoder.dropout_probability"
    labeled_float(
        "TE Dropout",
        tag=tag,
        default_value=cfg.text_encoder.dropout_probability,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "text_encoder.stop_training_after"
    te_stop = cfg.text_encoder.stop_training_after or 0
    tag_unit = "text_encoder.stop_training_after_unit"
    time_entry(
        "TE Stop After",
        value_tag=tag,
        unit_tag=tag_unit,
        default_value=float(te_stop),
        default_unit=_enum_default(cfg.text_encoder, "stop_training_after_unit"),
        unit_items=enum_values(TimeUnit),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)
    _reg(ui, tag_unit)

    tag = "text_encoder.learning_rate"
    labeled_float(
        "TE Learning Rate",
        tag=tag,
        default_value=cfg.text_encoder.learning_rate or 0.0,
        callback=ui.make_callback(tag),
        parent=parent,
        format_str="%.2e",
    )
    _reg(ui, tag)

    # ---- Embedding section ----
    labeled_separator("Embedding", parent=parent)

    tag = "embedding_learning_rate"
    labeled_float(
        "Embedding LR",
        tag=tag,
        default_value=cfg.embedding_learning_rate or 0.0,
        callback=ui.make_callback(tag),
        parent=parent,
        format_str="%.2e",
    )
    _reg(ui, tag)

    tag = "preserve_embedding_norm"
    labeled_checkbox(
        "Preserve Embedding Norm",
        tag=tag,
        default_value=cfg.preserve_embedding_norm,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)


def _build_col1(parent: int | str, ui: UIState) -> None:
    """Column 1 -- EMA, Precision/Memory, Transformer, Noise."""
    cfg = ui.config

    # ---- EMA section ----
    labeled_separator("EMA", parent=parent)

    tag = "ema"
    labeled_combo(
        "EMA Mode", enum_values(EMAMode),
        tag=tag,
        default_value=_enum_default(cfg, "ema"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "ema_decay"
    labeled_float(
        "EMA Decay",
        tag=tag,
        default_value=cfg.ema_decay,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "ema_update_step_interval"
    labeled_int(
        "EMA Update Interval",
        tag=tag,
        default_value=cfg.ema_update_step_interval,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Precision & Memory section ----
    labeled_separator("Precision & Memory", parent=parent)

    tag = "gradient_checkpointing"
    labeled_combo(
        "Gradient Checkpointing", enum_values(GradientCheckpointingMethod),
        tag=tag,
        default_value=_enum_default(cfg, "gradient_checkpointing"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "layer_offload_fraction"
    labeled_float(
        "Layer Offload Fraction",
        tag=tag,
        default_value=cfg.layer_offload_fraction,
        callback=ui.make_callback(tag),
        parent=parent,
        min_value=0.0,
        max_value=1.0,
    )
    _reg(ui, tag)

    # Train dtype -- subset of DataType relevant for training
    _train_dtypes = [
        DataType.FLOAT_32.value,
        DataType.FLOAT_16.value,
        DataType.BFLOAT_16.value,
        DataType.TFLOAT_32.value,
    ]
    tag = "train_dtype"
    labeled_combo(
        "Train Dtype", _train_dtypes,
        tag=tag,
        default_value=_enum_default(cfg, "train_dtype"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    _fallback_dtypes = [DataType.FLOAT_32.value, DataType.BFLOAT_16.value]
    tag = "fallback_train_dtype"
    labeled_combo(
        "Fallback Dtype", _fallback_dtypes,
        tag=tag,
        default_value=_enum_default(cfg, "fallback_train_dtype"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "enable_autocast_cache"
    labeled_checkbox(
        "Autocast Cache",
        tag=tag,
        default_value=cfg.enable_autocast_cache,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "resolution"
    labeled_input(
        "Resolution",
        tag=tag,
        default_value=cfg.resolution,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "force_circular_padding"
    labeled_checkbox(
        "Force Circular Padding",
        tag=tag,
        default_value=cfg.force_circular_padding,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Transformer section ----
    labeled_separator("Transformer", parent=parent)

    tag = "transformer.train"
    labeled_checkbox(
        "Train Transformer",
        tag=tag,
        default_value=cfg.transformer.train,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "transformer.stop_training_after"
    tf_stop = cfg.transformer.stop_training_after or 0
    tag_unit = "transformer.stop_training_after_unit"
    time_entry(
        "TF Stop After",
        value_tag=tag,
        unit_tag=tag_unit,
        default_value=float(tf_stop),
        default_unit=_enum_default(cfg.transformer, "stop_training_after_unit"),
        unit_items=enum_values(TimeUnit),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)
    _reg(ui, tag_unit)

    tag = "transformer.learning_rate"
    labeled_float(
        "TF Learning Rate",
        tag=tag,
        default_value=cfg.transformer.learning_rate or 0.0,
        callback=ui.make_callback(tag),
        parent=parent,
        format_str="%.2e",
    )
    _reg(ui, tag)

    tag = "transformer.attention_mask"
    labeled_checkbox(
        "Attention Mask",
        tag=tag,
        default_value=cfg.transformer.attention_mask,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "transformer.guidance_scale"
    labeled_float(
        "Guidance Scale",
        tag=tag,
        default_value=cfg.transformer.guidance_scale,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Noise section ----
    labeled_separator("Noise", parent=parent)

    tag = "offset_noise_weight"
    labeled_float(
        "Offset Noise Weight",
        tag=tag,
        default_value=cfg.offset_noise_weight,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "perturbation_noise_weight"
    labeled_float(
        "Perturbation Noise",
        tag=tag,
        default_value=cfg.perturbation_noise_weight,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "timestep_distribution"
    labeled_combo(
        "Timestep Distribution", enum_values(TimestepDistribution),
        tag=tag,
        default_value=_enum_default(cfg, "timestep_distribution"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "min_noising_strength"
    labeled_float(
        "Min Noising Strength",
        tag=tag,
        default_value=cfg.min_noising_strength,
        callback=ui.make_callback(tag),
        parent=parent,
        min_value=0.0,
        max_value=1.0,
    )
    _reg(ui, tag)

    tag = "max_noising_strength"
    labeled_float(
        "Max Noising Strength",
        tag=tag,
        default_value=cfg.max_noising_strength,
        callback=ui.make_callback(tag),
        parent=parent,
        min_value=0.0,
        max_value=1.0,
    )
    _reg(ui, tag)

    tag = "noising_weight"
    labeled_float(
        "Noising Weight",
        tag=tag,
        default_value=cfg.noising_weight,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "noising_bias"
    labeled_float(
        "Noising Bias",
        tag=tag,
        default_value=cfg.noising_bias,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "timestep_shift"
    labeled_float(
        "Timestep Shift",
        tag=tag,
        default_value=cfg.timestep_shift,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "dynamic_timestep_shifting"
    labeled_checkbox(
        "Dynamic Timestep Shifting",
        tag=tag,
        default_value=cfg.dynamic_timestep_shifting,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)


def _build_col2(parent: int | str, ui: UIState) -> None:
    """Column 2 -- Masked Training, Loss, Layer Filter."""
    cfg = ui.config

    # ---- Masked Training section ----
    labeled_separator("Masked Training", parent=parent)

    tag = "masked_training"
    labeled_checkbox(
        "Masked Training",
        tag=tag,
        default_value=cfg.masked_training,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "unmasked_probability"
    labeled_float(
        "Unmasked Probability",
        tag=tag,
        default_value=cfg.unmasked_probability,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "unmasked_weight"
    labeled_float(
        "Unmasked Weight",
        tag=tag,
        default_value=cfg.unmasked_weight,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "normalize_masked_area_loss"
    labeled_checkbox(
        "Normalize Masked Loss",
        tag=tag,
        default_value=cfg.normalize_masked_area_loss,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "masked_prior_preservation_weight"
    labeled_float(
        "Masked Prior Weight",
        tag=tag,
        default_value=cfg.masked_prior_preservation_weight,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "custom_conditioning_image"
    labeled_checkbox(
        "Custom Conditioning Image",
        tag=tag,
        default_value=cfg.custom_conditioning_image,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Loss section ----
    labeled_separator("Loss", parent=parent)

    tag = "mse_strength"
    labeled_float(
        "MSE Strength",
        tag=tag,
        default_value=cfg.mse_strength,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "mae_strength"
    labeled_float(
        "MAE Strength",
        tag=tag,
        default_value=cfg.mae_strength,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "log_cosh_strength"
    labeled_float(
        "Log-Cosh Strength",
        tag=tag,
        default_value=cfg.log_cosh_strength,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "huber_strength"
    labeled_float(
        "Huber Strength",
        tag=tag,
        default_value=cfg.huber_strength,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "huber_delta"
    labeled_float(
        "Huber Delta",
        tag=tag,
        default_value=cfg.huber_delta,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "loss_weight_fn"
    labeled_combo(
        "Loss Weight Function", enum_values(LossWeight),
        tag=tag,
        default_value=_enum_default(cfg, "loss_weight_fn"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "loss_weight_strength"
    labeled_float(
        "Loss Weight Strength",
        tag=tag,
        default_value=cfg.loss_weight_strength,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "loss_scaler"
    labeled_combo(
        "Loss Scaler", enum_values(LossScaler),
        tag=tag,
        default_value=_enum_default(cfg, "loss_scaler"),
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "dropout_probability"
    labeled_float(
        "Dropout Probability",
        tag=tag,
        default_value=cfg.dropout_probability,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    # ---- Layer Filter section ----
    labeled_separator("Layer Filter", parent=parent)

    tag = "layer_filter_preset"
    labeled_input(
        "Filter Preset",
        tag=tag,
        default_value=cfg.layer_filter_preset,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "layer_filter"
    labeled_input(
        "Layer Filter",
        tag=tag,
        default_value=cfg.layer_filter,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)

    tag = "layer_filter_regex"
    labeled_checkbox(
        "Regex Filter",
        tag=tag,
        default_value=cfg.layer_filter_regex,
        callback=ui.make_callback(tag),
        parent=parent,
    )
    _reg(ui, tag)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_tab(ui_state: UIState) -> None:
    """Build the Training settings tab inside the current DPG parent."""
    with dpg.group(horizontal=True):
        with dpg.child_window(width=_COL_W, border=False):
            _build_col0(dpg.last_container(), ui_state)
        with dpg.child_window(width=_COL_W, border=False):
            _build_col1(dpg.last_container(), ui_state)
        with dpg.child_window(width=_COL_W, border=False):
            _build_col2(dpg.last_container(), ui_state)
