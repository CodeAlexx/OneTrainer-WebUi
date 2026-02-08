"""Training tab -- optimizer, LR, noise, masking, and loss settings.

Two-column layout with collapsible sections.  Left column covers
optimizer / LR / training / text-encoder / embedding settings.  Right column
has precision & memory, noise, loss, transformer, masked training, and layer
filtering.
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
    section,
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

# Column width for the 2-column layout (sized for 4K readability)
_COL_W = 780


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

def _build_left(parent: int | str, ui: UIState) -> None:
    """Left column -- Optimizer & LR, Training, Text Encoder, Embedding."""
    cfg = ui.config

    # ---- Optimizer & LR section ----
    with section("Optimizer & LR", parent=parent):
        labeled_combo(
            "Optimizer", enum_values(Optimizer),
            tag="optimizer.optimizer",
            default_value=_enum_default(cfg.optimizer, "optimizer"),
            callback=ui.make_callback("optimizer.optimizer"),
        )
        labeled_combo(
            "LR Scheduler", enum_values(LearningRateScheduler),
            tag="learning_rate_scheduler",
            default_value=_enum_default(cfg, "learning_rate_scheduler"),
            callback=ui.make_callback("learning_rate_scheduler"),
        )
        labeled_float(
            "Learning Rate",
            tag="learning_rate",
            default_value=cfg.learning_rate,
            callback=ui.make_callback("learning_rate"),
            format_str="%.2e",
        )
        labeled_float(
            "LR Warmup Steps",
            tag="learning_rate_warmup_steps",
            default_value=cfg.learning_rate_warmup_steps,
            callback=ui.make_callback("learning_rate_warmup_steps"),
        )
        labeled_float(
            "LR Min Factor",
            tag="learning_rate_min_factor",
            default_value=cfg.learning_rate_min_factor,
            callback=ui.make_callback("learning_rate_min_factor"),
        )
        labeled_float(
            "LR Cycles",
            tag="learning_rate_cycles",
            default_value=cfg.learning_rate_cycles,
            callback=ui.make_callback("learning_rate_cycles"),
        )
        labeled_combo(
            "LR Scaler", enum_values(LearningRateScaler),
            tag="learning_rate_scaler",
            default_value=_enum_default(cfg, "learning_rate_scaler"),
            callback=ui.make_callback("learning_rate_scaler"),
        )
        labeled_float(
            "Clip Grad Norm",
            tag="clip_grad_norm",
            default_value=cfg.clip_grad_norm or 1.0,
            callback=ui.make_callback("clip_grad_norm"),
        )
    _reg(ui, "optimizer.optimizer")
    _reg(ui, "learning_rate_scheduler")
    _reg(ui, "learning_rate")
    _reg(ui, "learning_rate_warmup_steps")
    _reg(ui, "learning_rate_min_factor")
    _reg(ui, "learning_rate_cycles")
    _reg(ui, "learning_rate_scaler")
    _reg(ui, "clip_grad_norm")

    # ---- Training section ----
    with section("Training", parent=parent):
        labeled_int(
            "Epochs",
            tag="epochs",
            default_value=cfg.epochs,
            callback=ui.make_callback("epochs"),
        )
        labeled_int(
            "Batch Size",
            tag="batch_size",
            default_value=cfg.batch_size,
            callback=ui.make_callback("batch_size"),
        )
        labeled_int(
            "Gradient Accum Steps",
            tag="gradient_accumulation_steps",
            default_value=cfg.gradient_accumulation_steps,
            callback=ui.make_callback("gradient_accumulation_steps"),
        )
        labeled_input(
            "Resolution",
            tag="resolution",
            default_value=cfg.resolution,
            callback=ui.make_callback("resolution"),
        )
    _reg(ui, "epochs")
    _reg(ui, "batch_size")
    _reg(ui, "gradient_accumulation_steps")
    _reg(ui, "resolution")

    # ---- Text Encoder section ----
    with section("Text Encoder", parent=parent, default_open=False):
        labeled_checkbox(
            "Train Text Encoder",
            tag="text_encoder.train",
            default_value=cfg.text_encoder.train,
            callback=ui.make_callback("text_encoder.train"),
        )
        labeled_float(
            "TE Dropout",
            tag="text_encoder.dropout_probability",
            default_value=cfg.text_encoder.dropout_probability,
            callback=ui.make_callback("text_encoder.dropout_probability"),
        )
        te_stop = cfg.text_encoder.stop_training_after or 0
        time_entry(
            "TE Stop After",
            value_tag="text_encoder.stop_training_after",
            unit_tag="text_encoder.stop_training_after_unit",
            default_value=float(te_stop),
            default_unit=_enum_default(cfg.text_encoder, "stop_training_after_unit"),
            unit_items=enum_values(TimeUnit),
            callback=ui.make_callback("text_encoder.stop_training_after"),
        )
        labeled_float(
            "TE Learning Rate",
            tag="text_encoder.learning_rate",
            default_value=cfg.text_encoder.learning_rate or 0.0,
            callback=ui.make_callback("text_encoder.learning_rate"),
            format_str="%.2e",
        )
    _reg(ui, "text_encoder.train")
    _reg(ui, "text_encoder.dropout_probability")
    _reg(ui, "text_encoder.stop_training_after")
    _reg(ui, "text_encoder.stop_training_after_unit")
    _reg(ui, "text_encoder.learning_rate")

    # ---- Embedding section ----
    with section("Embedding", parent=parent, default_open=False):
        labeled_float(
            "Embedding LR",
            tag="embedding_learning_rate",
            default_value=cfg.embedding_learning_rate or 0.0,
            callback=ui.make_callback("embedding_learning_rate"),
            format_str="%.2e",
        )
        labeled_checkbox(
            "Preserve Embedding Norm",
            tag="preserve_embedding_norm",
            default_value=cfg.preserve_embedding_norm,
            callback=ui.make_callback("preserve_embedding_norm"),
        )
    _reg(ui, "embedding_learning_rate")
    _reg(ui, "preserve_embedding_norm")


def _build_right(parent: int | str, ui: UIState) -> None:
    """Right column -- Precision & Memory, Noise, Loss, Transformer, Masked Training, Layer Filter."""
    cfg = ui.config

    # ---- Precision & Memory section ----
    with section("Precision & Memory", parent=parent):
        labeled_combo(
            "Gradient Checkpointing", enum_values(GradientCheckpointingMethod),
            tag="gradient_checkpointing",
            default_value=_enum_default(cfg, "gradient_checkpointing"),
            callback=ui.make_callback("gradient_checkpointing"),
        )
        labeled_float(
            "Layer Offload Fraction",
            tag="layer_offload_fraction",
            default_value=cfg.layer_offload_fraction,
            callback=ui.make_callback("layer_offload_fraction"),
            min_value=0.0,
            max_value=1.0,
        )
        _train_dtypes = [
            DataType.FLOAT_32.value,
            DataType.FLOAT_16.value,
            DataType.BFLOAT_16.value,
            DataType.TFLOAT_32.value,
        ]
        labeled_combo(
            "Train Dtype", _train_dtypes,
            tag="train_dtype",
            default_value=_enum_default(cfg, "train_dtype"),
            callback=ui.make_callback("train_dtype"),
        )
        _fallback_dtypes = [DataType.FLOAT_32.value, DataType.BFLOAT_16.value]
        labeled_combo(
            "Fallback Dtype", _fallback_dtypes,
            tag="fallback_train_dtype",
            default_value=_enum_default(cfg, "fallback_train_dtype"),
            callback=ui.make_callback("fallback_train_dtype"),
        )
        labeled_checkbox(
            "Autocast Cache",
            tag="enable_autocast_cache",
            default_value=cfg.enable_autocast_cache,
            callback=ui.make_callback("enable_autocast_cache"),
        )
        labeled_checkbox(
            "Force Circular Padding",
            tag="force_circular_padding",
            default_value=cfg.force_circular_padding,
            callback=ui.make_callback("force_circular_padding"),
        )
    _reg(ui, "gradient_checkpointing")
    _reg(ui, "layer_offload_fraction")
    _reg(ui, "train_dtype")
    _reg(ui, "fallback_train_dtype")
    _reg(ui, "enable_autocast_cache")
    _reg(ui, "force_circular_padding")

    # ---- EMA section ----
    with section("EMA", parent=parent):
        labeled_combo(
            "EMA Mode", enum_values(EMAMode),
            tag="ema",
            default_value=_enum_default(cfg, "ema"),
            callback=ui.make_callback("ema"),
        )
        labeled_float(
            "EMA Decay",
            tag="ema_decay",
            default_value=cfg.ema_decay,
            callback=ui.make_callback("ema_decay"),
        )
        labeled_int(
            "EMA Update Interval",
            tag="ema_update_step_interval",
            default_value=cfg.ema_update_step_interval,
            callback=ui.make_callback("ema_update_step_interval"),
        )
    _reg(ui, "ema")
    _reg(ui, "ema_decay")
    _reg(ui, "ema_update_step_interval")

    # ---- Noise section ----
    with section("Noise", parent=parent):
        labeled_float(
            "Offset Noise Weight",
            tag="offset_noise_weight",
            default_value=cfg.offset_noise_weight,
            callback=ui.make_callback("offset_noise_weight"),
        )
        labeled_float(
            "Perturbation Noise",
            tag="perturbation_noise_weight",
            default_value=cfg.perturbation_noise_weight,
            callback=ui.make_callback("perturbation_noise_weight"),
        )
        labeled_combo(
            "Timestep Distribution", enum_values(TimestepDistribution),
            tag="timestep_distribution",
            default_value=_enum_default(cfg, "timestep_distribution"),
            callback=ui.make_callback("timestep_distribution"),
        )
        labeled_float(
            "Min Noising Strength",
            tag="min_noising_strength",
            default_value=cfg.min_noising_strength,
            callback=ui.make_callback("min_noising_strength"),
            min_value=0.0,
            max_value=1.0,
        )
        labeled_float(
            "Max Noising Strength",
            tag="max_noising_strength",
            default_value=cfg.max_noising_strength,
            callback=ui.make_callback("max_noising_strength"),
            min_value=0.0,
            max_value=1.0,
        )
        labeled_float(
            "Noising Weight",
            tag="noising_weight",
            default_value=cfg.noising_weight,
            callback=ui.make_callback("noising_weight"),
        )
        labeled_float(
            "Noising Bias",
            tag="noising_bias",
            default_value=cfg.noising_bias,
            callback=ui.make_callback("noising_bias"),
        )
        labeled_float(
            "Timestep Shift",
            tag="timestep_shift",
            default_value=cfg.timestep_shift,
            callback=ui.make_callback("timestep_shift"),
        )
        labeled_checkbox(
            "Dynamic Timestep Shifting",
            tag="dynamic_timestep_shifting",
            default_value=cfg.dynamic_timestep_shifting,
            callback=ui.make_callback("dynamic_timestep_shifting"),
        )
    _reg(ui, "offset_noise_weight")
    _reg(ui, "perturbation_noise_weight")
    _reg(ui, "timestep_distribution")
    _reg(ui, "min_noising_strength")
    _reg(ui, "max_noising_strength")
    _reg(ui, "noising_weight")
    _reg(ui, "noising_bias")
    _reg(ui, "timestep_shift")
    _reg(ui, "dynamic_timestep_shifting")

    # ---- Loss section ----
    with section("Loss", parent=parent):
        labeled_float(
            "MSE Strength",
            tag="mse_strength",
            default_value=cfg.mse_strength,
            callback=ui.make_callback("mse_strength"),
        )
        labeled_float(
            "MAE Strength",
            tag="mae_strength",
            default_value=cfg.mae_strength,
            callback=ui.make_callback("mae_strength"),
        )
        labeled_float(
            "Log-Cosh Strength",
            tag="log_cosh_strength",
            default_value=cfg.log_cosh_strength,
            callback=ui.make_callback("log_cosh_strength"),
        )
        labeled_float(
            "Huber Strength",
            tag="huber_strength",
            default_value=cfg.huber_strength,
            callback=ui.make_callback("huber_strength"),
        )
        labeled_float(
            "Huber Delta",
            tag="huber_delta",
            default_value=cfg.huber_delta,
            callback=ui.make_callback("huber_delta"),
        )
        labeled_combo(
            "Loss Weight Function", enum_values(LossWeight),
            tag="loss_weight_fn",
            default_value=_enum_default(cfg, "loss_weight_fn"),
            callback=ui.make_callback("loss_weight_fn"),
        )
        labeled_float(
            "Loss Weight Strength",
            tag="loss_weight_strength",
            default_value=cfg.loss_weight_strength,
            callback=ui.make_callback("loss_weight_strength"),
        )
        labeled_combo(
            "Loss Scaler", enum_values(LossScaler),
            tag="loss_scaler",
            default_value=_enum_default(cfg, "loss_scaler"),
            callback=ui.make_callback("loss_scaler"),
        )
        labeled_float(
            "Dropout Probability",
            tag="dropout_probability",
            default_value=cfg.dropout_probability,
            callback=ui.make_callback("dropout_probability"),
        )
    _reg(ui, "mse_strength")
    _reg(ui, "mae_strength")
    _reg(ui, "log_cosh_strength")
    _reg(ui, "huber_strength")
    _reg(ui, "huber_delta")
    _reg(ui, "loss_weight_fn")
    _reg(ui, "loss_weight_strength")
    _reg(ui, "loss_scaler")
    _reg(ui, "dropout_probability")

    # ---- Transformer section ----
    with section("Transformer", parent=parent, default_open=False):
        labeled_checkbox(
            "Train Transformer",
            tag="transformer.train",
            default_value=cfg.transformer.train,
            callback=ui.make_callback("transformer.train"),
        )
        tf_stop = cfg.transformer.stop_training_after or 0
        time_entry(
            "TF Stop After",
            value_tag="transformer.stop_training_after",
            unit_tag="transformer.stop_training_after_unit",
            default_value=float(tf_stop),
            default_unit=_enum_default(cfg.transformer, "stop_training_after_unit"),
            unit_items=enum_values(TimeUnit),
            callback=ui.make_callback("transformer.stop_training_after"),
        )
        labeled_float(
            "TF Learning Rate",
            tag="transformer.learning_rate",
            default_value=cfg.transformer.learning_rate or 0.0,
            callback=ui.make_callback("transformer.learning_rate"),
            format_str="%.2e",
        )
        labeled_checkbox(
            "Attention Mask",
            tag="transformer.attention_mask",
            default_value=cfg.transformer.attention_mask,
            callback=ui.make_callback("transformer.attention_mask"),
        )
        labeled_float(
            "Guidance Scale",
            tag="transformer.guidance_scale",
            default_value=cfg.transformer.guidance_scale,
            callback=ui.make_callback("transformer.guidance_scale"),
        )
    _reg(ui, "transformer.train")
    _reg(ui, "transformer.stop_training_after")
    _reg(ui, "transformer.stop_training_after_unit")
    _reg(ui, "transformer.learning_rate")
    _reg(ui, "transformer.attention_mask")
    _reg(ui, "transformer.guidance_scale")

    # ---- Masked Training section ----
    with section("Masked Training", parent=parent, default_open=False):
        labeled_checkbox(
            "Masked Training",
            tag="masked_training",
            default_value=cfg.masked_training,
            callback=ui.make_callback("masked_training"),
        )
        labeled_float(
            "Unmasked Probability",
            tag="unmasked_probability",
            default_value=cfg.unmasked_probability,
            callback=ui.make_callback("unmasked_probability"),
        )
        labeled_float(
            "Unmasked Weight",
            tag="unmasked_weight",
            default_value=cfg.unmasked_weight,
            callback=ui.make_callback("unmasked_weight"),
        )
        labeled_checkbox(
            "Normalize Masked Loss",
            tag="normalize_masked_area_loss",
            default_value=cfg.normalize_masked_area_loss,
            callback=ui.make_callback("normalize_masked_area_loss"),
        )
        labeled_float(
            "Masked Prior Weight",
            tag="masked_prior_preservation_weight",
            default_value=cfg.masked_prior_preservation_weight,
            callback=ui.make_callback("masked_prior_preservation_weight"),
        )
        labeled_checkbox(
            "Custom Conditioning Image",
            tag="custom_conditioning_image",
            default_value=cfg.custom_conditioning_image,
            callback=ui.make_callback("custom_conditioning_image"),
        )
    _reg(ui, "masked_training")
    _reg(ui, "unmasked_probability")
    _reg(ui, "unmasked_weight")
    _reg(ui, "normalize_masked_area_loss")
    _reg(ui, "masked_prior_preservation_weight")
    _reg(ui, "custom_conditioning_image")

    # ---- Layer Filter section ----
    with section("Layer Filter", parent=parent, default_open=False):
        labeled_input(
            "Filter Preset",
            tag="layer_filter_preset",
            default_value=cfg.layer_filter_preset,
            callback=ui.make_callback("layer_filter_preset"),
        )
        labeled_input(
            "Layer Filter",
            tag="layer_filter",
            default_value=cfg.layer_filter,
            callback=ui.make_callback("layer_filter"),
        )
        labeled_checkbox(
            "Regex Filter",
            tag="layer_filter_regex",
            default_value=cfg.layer_filter_regex,
            callback=ui.make_callback("layer_filter_regex"),
        )
    _reg(ui, "layer_filter_preset")
    _reg(ui, "layer_filter")
    _reg(ui, "layer_filter_regex")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_tab(ui_state: UIState) -> None:
    """Build the Training settings tab inside the current DPG parent."""
    with dpg.group(horizontal=True):
        col_left = dpg.add_child_window(width=_COL_W, border=False)
        col_right = dpg.add_child_window(width=_COL_W, border=False)
    _build_left(col_left, ui_state)
    _build_right(col_right, ui_state)
