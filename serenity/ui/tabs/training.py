"""Training tab -- optimizer, LR, noise, masking, and loss settings.

Two-column layout with collapsible sections.  Left column covers
optimizer / LR / training / text-encoder / embedding settings.  Right column
has precision & memory, noise, loss, transformer, masked training, and layer
filtering.
"""

from __future__ import annotations

import dataclasses
from typing import Any, get_args, get_origin

import dearpygui.dearpygui as dpg

from serenity.core.config import TrainOptimizerConfig
from serenity.ui.state import UIState
from serenity.ui.theme import scaled
from serenity.ui.widgets import (
    LABEL_WIDTH,
    INPUT_WIDTH,
    enum_values,
    labeled_checkbox,
    labeled_combo,
    labeled_float,
    labeled_input,
    labeled_int,
    section,
    time_entry,
    tooltip,
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

# Column width for the 2-column layout -- auto-scaled
_COL_W = scaled(560)
_OPTIMIZER_STATUS_TAG = "training_optimizer_status"
_OPTIMIZER_WINDOW_TAG = "training_optimizer_window"
_OPTIMIZER_WINDOW_LIST_TAG = "training_optimizer_window_list"
_OPTIMIZER_WINDOW_STATUS_TAG = "training_optimizer_window_status"
_SCHEDULER_ADV_BUTTON_TAG = "training_scheduler_adv_button"
_SCHEDULER_WINDOW_TAG = "training_scheduler_window"
_SCHEDULER_WINDOW_CONTENT_TAG = "training_scheduler_window_content"
_SCHEDULER_PARAMS_LIST_TAG = "training_scheduler_params_list"
_SCHEDULER_CUSTOM_CLASS_TAG = "training_scheduler_custom_class"
_OFFLOADING_WINDOW_TAG = "training_offloading_window"
_OFFLOAD_GC_TAG = "training_offload_gc"
_OFFLOAD_ASYNC_TAG = "training_offload_async"
_OFFLOAD_ACTIVATION_TAG = "training_offload_activation"
_OFFLOAD_LAYER_FRAC_TAG = "training_offload_layer_frac"
_TIMESTEP_WINDOW_TAG = "training_timestep_window"
_TIMESTEP_DIST_TAG = "training_timestep_dist"
_TIMESTEP_MIN_TAG = "training_timestep_min"
_TIMESTEP_MAX_TAG = "training_timestep_max"
_TIMESTEP_WEIGHT_TAG = "training_timestep_weight"
_TIMESTEP_BIAS_TAG = "training_timestep_bias"
_TIMESTEP_SHIFT_TAG = "training_timestep_shift"
_TIMESTEP_DYNAMIC_TAG = "training_timestep_dynamic"

_OPTIMIZER_HIDDEN_FIELDS = {"optimizer", "muon_adam_config"}
_OPTIMIZER_COMMON_FIELDS = (
    "weight_decay",
    "beta1",
    "beta2",
    "eps",
    "momentum",
    "dampening",
    "clip_threshold",
    "decay_rate",
    "fused",
    "fused_back_pass",
    "foreach",
    "relative_step",
    "scale_parameter",
    "stochastic_rounding",
    "warmup_init",
)


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


def _set_nested_attr(obj: object, path: str, value: object) -> None:
    """Set a dotted config path on a dataclass-style object."""
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _safe_set_item_value(tag: str, value: object) -> None:
    """Set a DPG item value if it exists."""
    if not dpg.does_item_exist(tag):
        return
    try:
        dpg.set_value(tag, value)
    except SystemError:
        pass


def _is_custom_scheduler(cfg: object) -> bool:
    """True when the selected LR scheduler is CUSTOM."""
    scheduler = _nested(cfg, "learning_rate_scheduler")
    if scheduler is None:
        return False
    if hasattr(scheduler, "value"):
        return str(scheduler.value).upper() == LearningRateScheduler.CUSTOM.value
    return str(scheduler).upper() == LearningRateScheduler.CUSTOM.value


def _update_scheduler_adv_button_state(ui: UIState) -> None:
    """Enable scheduler `...` only for CUSTOM scheduler, matching OneTrainer."""
    if not dpg.does_item_exist(_SCHEDULER_ADV_BUTTON_TAG):
        return
    enabled = _is_custom_scheduler(ui.config)
    dpg.configure_item(_SCHEDULER_ADV_BUTTON_TAG, enabled=enabled)


def _on_scheduler_changed(ui: UIState) -> None:
    """Persist scheduler and refresh related advanced UI state."""
    ui.widget_to_config("learning_rate_scheduler")
    _update_scheduler_adv_button_state(ui)
    if dpg.does_item_exist(_SCHEDULER_WINDOW_TAG):
        _refresh_scheduler_window(ui)


def _ensure_scheduler_params(cfg: object) -> list[dict[str, str]]:
    """Normalize scheduler_params to a mutable list[dict[str, str]]."""
    params = getattr(cfg, "scheduler_params", None)
    if not isinstance(params, list):
        params = []
        setattr(cfg, "scheduler_params", params)

    normalized: list[dict[str, str]] = []
    for entry in params:
        if isinstance(entry, dict):
            normalized.append(
                {
                    "key": str(entry.get("key", "")),
                    "value": str(entry.get("value", "")),
                }
            )
    setattr(cfg, "scheduler_params", normalized)
    return normalized


def _update_scheduler_param(ui: UIState, index: int, field: str, value: object) -> None:
    """Update one scheduler key/value row."""
    params = _ensure_scheduler_params(ui.config)
    if index < 0 or index >= len(params):
        return
    params[index][field] = str(value)


def _remove_scheduler_param(ui: UIState, index: int) -> None:
    """Delete one scheduler key/value row."""
    params = _ensure_scheduler_params(ui.config)
    if 0 <= index < len(params):
        params.pop(index)
        _rebuild_scheduler_params_list(ui)


def _add_scheduler_param(ui: UIState) -> None:
    """Append a scheduler key/value row."""
    params = _ensure_scheduler_params(ui.config)
    params.append({"key": "", "value": ""})
    _rebuild_scheduler_params_list(ui)


def _rebuild_scheduler_params_list(ui: UIState) -> None:
    """Rebuild scheduler key/value editor rows."""
    if not dpg.does_item_exist(_SCHEDULER_PARAMS_LIST_TAG):
        return

    dpg.delete_item(_SCHEDULER_PARAMS_LIST_TAG, children_only=True)
    params = _ensure_scheduler_params(ui.config)
    for idx, entry in enumerate(params):
        with dpg.group(horizontal=True, parent=_SCHEDULER_PARAMS_LIST_TAG):
            dpg.add_button(
                label="X",
                width=scaled(26),
                callback=lambda _s, _a, i=idx: _remove_scheduler_param(ui, i),
            )
            dpg.add_input_text(
                default_value=entry.get("key", ""),
                width=scaled(180),
                hint="key",
                callback=lambda _s, v, i=idx: _update_scheduler_param(ui, i, "key", v),
            )
            dpg.add_input_text(
                default_value=entry.get("value", ""),
                width=scaled(300),
                hint="value",
                callback=lambda _s, v, i=idx: _update_scheduler_param(ui, i, "value", v),
            )


def _refresh_scheduler_window(ui: UIState) -> None:
    """Refresh scheduler popup content to match current scheduler selection."""
    if not dpg.does_item_exist(_SCHEDULER_WINDOW_CONTENT_TAG):
        return

    dpg.delete_item(_SCHEDULER_WINDOW_CONTENT_TAG, children_only=True)
    cfg = ui.config
    if _is_custom_scheduler(cfg):
        ui.register(
            _SCHEDULER_CUSTOM_CLASS_TAG,
            field_path="custom_learning_rate_scheduler",
            type_hint=str,
        )
        labeled_input(
            "Class Name",
            tag=_SCHEDULER_CUSTOM_CLASS_TAG,
            default_value=str(cfg.custom_learning_rate_scheduler or ""),
            callback=ui.make_callback(_SCHEDULER_CUSTOM_CLASS_TAG),
            parent=_SCHEDULER_WINDOW_CONTENT_TAG,
            tip="Python import path for custom scheduler class",
        )
        dpg.add_spacer(height=6, parent=_SCHEDULER_WINDOW_CONTENT_TAG)

    with dpg.group(horizontal=True, parent=_SCHEDULER_WINDOW_CONTENT_TAG):
        dpg.add_button(
            label="Add Parameter",
            callback=lambda _s, _a, _u: _add_scheduler_param(ui),
        )

    dpg.add_child_window(
        tag=_SCHEDULER_PARAMS_LIST_TAG,
        autosize_x=True,
        height=scaled(300),
        border=False,
        parent=_SCHEDULER_WINDOW_CONTENT_TAG,
    )
    _rebuild_scheduler_params_list(ui)


def _open_scheduler_window(ui: UIState) -> None:
    """Open (or focus) scheduler advanced settings popup."""
    if dpg.does_item_exist(_SCHEDULER_WINDOW_TAG):
        dpg.show_item(_SCHEDULER_WINDOW_TAG)
        dpg.focus_item(_SCHEDULER_WINDOW_TAG)
        _refresh_scheduler_window(ui)
        return

    with dpg.window(
        tag=_SCHEDULER_WINDOW_TAG,
        label="Learning Rate Scheduler Settings",
        width=scaled(760),
        height=scaled(520),
    ):
        dpg.add_child_window(
            tag=_SCHEDULER_WINDOW_CONTENT_TAG,
            autosize_x=True,
            height=-1,
            border=False,
        )
    _refresh_scheduler_window(ui)


def _offloading_apply(
    ui: UIState,
    popup_tag: str,
    path: str,
    value: object,
    main_tag: str | None = None,
) -> None:
    """Apply offloading popup edits to config and mirror into main control."""
    if path == "gradient_checkpointing" and not isinstance(value, GradientCheckpointingMethod):
        try:
            value = GradientCheckpointingMethod(str(value))
        except ValueError:
            value = GradientCheckpointingMethod.OFF
    _set_nested_attr(ui.config, path, value)
    if main_tag is not None:
        shown = value.value if hasattr(value, "value") else value
        _safe_set_item_value(main_tag, shown)


def _sync_offloading_popup(ui: UIState) -> None:
    """Sync offloading popup controls from current config."""
    cfg = ui.config
    _safe_set_item_value(_OFFLOAD_GC_TAG, _enum_default(cfg, "gradient_checkpointing"))
    _safe_set_item_value(_OFFLOAD_ASYNC_TAG, bool(cfg.enable_async_offloading))
    _safe_set_item_value(_OFFLOAD_ACTIVATION_TAG, bool(cfg.enable_activation_offloading))
    _safe_set_item_value(_OFFLOAD_LAYER_FRAC_TAG, float(cfg.layer_offload_fraction))


def _open_offloading_window(ui: UIState) -> None:
    """Open (or focus) offloading popup."""
    if dpg.does_item_exist(_OFFLOADING_WINDOW_TAG):
        dpg.show_item(_OFFLOADING_WINDOW_TAG)
        dpg.focus_item(_OFFLOADING_WINDOW_TAG)
        _sync_offloading_popup(ui)
        return

    cfg = ui.config
    with dpg.window(
        tag=_OFFLOADING_WINDOW_TAG,
        label="Offloading",
        width=scaled(760),
        height=scaled(460),
    ):
        labeled_combo(
            "Gradient Checkpointing",
            enum_values(GradientCheckpointingMethod),
            tag=_OFFLOAD_GC_TAG,
            default_value=_enum_default(cfg, "gradient_checkpointing"),
            callback=lambda _s, v: _offloading_apply(
                ui, _OFFLOAD_GC_TAG, "gradient_checkpointing", v, "gradient_checkpointing"
            ),
            tip="Checkpointing strategy for memory/speed tradeoff",
        )
        labeled_checkbox(
            "Async Offloading",
            tag=_OFFLOAD_ASYNC_TAG,
            default_value=cfg.enable_async_offloading,
            callback=lambda _s, v: _offloading_apply(
                ui, _OFFLOAD_ASYNC_TAG, "enable_async_offloading", bool(v)
            ),
        )
        labeled_checkbox(
            "Offload Activations",
            tag=_OFFLOAD_ACTIVATION_TAG,
            default_value=cfg.enable_activation_offloading,
            callback=lambda _s, v: _offloading_apply(
                ui, _OFFLOAD_ACTIVATION_TAG, "enable_activation_offloading", bool(v)
            ),
        )
        labeled_float(
            "Layer Offload Fraction",
            tag=_OFFLOAD_LAYER_FRAC_TAG,
            default_value=cfg.layer_offload_fraction,
            callback=lambda _s, v: _offloading_apply(
                ui, _OFFLOAD_LAYER_FRAC_TAG, "layer_offload_fraction", float(v), "layer_offload_fraction"
            ),
            min_value=0.0,
            max_value=1.0,
        )


def _timestep_apply(
    ui: UIState,
    path: str,
    value: object,
    main_tag: str | None = None,
) -> None:
    """Apply timestep popup value to config and mirror to main control."""
    if path == "timestep_distribution" and not isinstance(value, TimestepDistribution):
        try:
            value = TimestepDistribution(str(value))
        except ValueError:
            value = TimestepDistribution.UNIFORM
    _set_nested_attr(ui.config, path, value)
    if main_tag is not None:
        shown = value.value if hasattr(value, "value") else value
        _safe_set_item_value(main_tag, shown)


def _sync_timestep_popup(ui: UIState) -> None:
    """Sync timestep popup controls from current config."""
    cfg = ui.config
    _safe_set_item_value(_TIMESTEP_DIST_TAG, _enum_default(cfg, "timestep_distribution"))
    _safe_set_item_value(_TIMESTEP_MIN_TAG, float(cfg.min_noising_strength))
    _safe_set_item_value(_TIMESTEP_MAX_TAG, float(cfg.max_noising_strength))
    _safe_set_item_value(_TIMESTEP_WEIGHT_TAG, float(cfg.noising_weight))
    _safe_set_item_value(_TIMESTEP_BIAS_TAG, float(cfg.noising_bias))
    _safe_set_item_value(_TIMESTEP_SHIFT_TAG, float(cfg.timestep_shift))
    _safe_set_item_value(_TIMESTEP_DYNAMIC_TAG, bool(cfg.dynamic_timestep_shifting))


def _open_timestep_window(ui: UIState) -> None:
    """Open (or focus) timestep distribution popup."""
    if dpg.does_item_exist(_TIMESTEP_WINDOW_TAG):
        dpg.show_item(_TIMESTEP_WINDOW_TAG)
        dpg.focus_item(_TIMESTEP_WINDOW_TAG)
        _sync_timestep_popup(ui)
        return

    cfg = ui.config
    with dpg.window(
        tag=_TIMESTEP_WINDOW_TAG,
        label="Timestep Distribution",
        width=scaled(780),
        height=scaled(560),
    ):
        labeled_combo(
            "Timestep Distribution",
            enum_values(TimestepDistribution),
            tag=_TIMESTEP_DIST_TAG,
            default_value=_enum_default(cfg, "timestep_distribution"),
            callback=lambda _s, v: _timestep_apply(
                ui, "timestep_distribution", v, "timestep_distribution"
            ),
        )
        labeled_float(
            "Min Noising Strength",
            tag=_TIMESTEP_MIN_TAG,
            default_value=cfg.min_noising_strength,
            callback=lambda _s, v: _timestep_apply(
                ui, "min_noising_strength", float(v), "min_noising_strength"
            ),
            min_value=0.0,
            max_value=1.0,
        )
        labeled_float(
            "Max Noising Strength",
            tag=_TIMESTEP_MAX_TAG,
            default_value=cfg.max_noising_strength,
            callback=lambda _s, v: _timestep_apply(
                ui, "max_noising_strength", float(v), "max_noising_strength"
            ),
            min_value=0.0,
            max_value=1.0,
        )
        labeled_float(
            "Noising Weight",
            tag=_TIMESTEP_WEIGHT_TAG,
            default_value=cfg.noising_weight,
            callback=lambda _s, v: _timestep_apply(
                ui, "noising_weight", float(v), "noising_weight"
            ),
        )
        labeled_float(
            "Noising Bias",
            tag=_TIMESTEP_BIAS_TAG,
            default_value=cfg.noising_bias,
            callback=lambda _s, v: _timestep_apply(
                ui, "noising_bias", float(v), "noising_bias"
            ),
        )
        labeled_float(
            "Timestep Shift",
            tag=_TIMESTEP_SHIFT_TAG,
            default_value=cfg.timestep_shift,
            callback=lambda _s, v: _timestep_apply(
                ui, "timestep_shift", float(v), "timestep_shift"
            ),
        )
        labeled_checkbox(
            "Dynamic Timestep Shifting",
            tag=_TIMESTEP_DYNAMIC_TAG,
            default_value=cfg.dynamic_timestep_shifting,
            callback=lambda _s, v: _timestep_apply(
                ui, "dynamic_timestep_shifting", bool(v), "dynamic_timestep_shifting"
            ),
        )


def _optimizer_name(cfg: object) -> str:
    """Return the selected optimizer name as a string value."""
    value = _nested(cfg, "optimizer", "optimizer")
    if value is None:
        return ""
    return value.value if hasattr(value, "value") else str(value)


def _selected_optimizer_defaults(cfg: object) -> dict[str, Any]:
    """Get defaults for the currently selected optimizer, if present."""
    opt_name = _optimizer_name(cfg)
    defaults = getattr(cfg, "optimizer_defaults", {})
    if not isinstance(defaults, dict) or not opt_name:
        return {}

    for key, value in defaults.items():
        if isinstance(key, str) and key.upper() == opt_name.upper() and isinstance(value, dict):
            return value
    return {}


def _optimizer_label(name: str) -> str:
    """Human-friendly labels for optimizer parameter keys."""
    aliases = {
        "beta1": "Beta1",
        "beta2": "Beta2",
        "beta3": "Beta3",
        "eps": "Epsilon",
        "eps2": "Epsilon 2",
        "lr_decay": "LR Decay",
        "min_8bit_size": "Min 8-bit Size",
        "fused_back_pass": "Fused Back Pass",
        "muon_te1_adam_lr": "Muon TE1 Adam LR",
        "muon_te2_adam_lr": "Muon TE2 Adam LR",
        "muon_adam_lr": "Muon Adam LR",
    }
    return aliases.get(name, name.replace("_", " ").title())


def _optimizer_type_hint(annotation: object, current_value: object) -> type:
    """Best-effort type hint for dynamic optimizer widgets."""
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _optimizer_type_hint(args[0], current_value)

    if annotation in (bool, int, float, str):
        return annotation

    if isinstance(current_value, bool):
        return bool
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int
    if isinstance(current_value, float):
        return float
    return str


def _optimizer_fields_to_show(cfg: object) -> list[str]:
    """Pick optimizer fields to expose: defaults + changed + common."""
    field_defs = {f.name: f for f in dataclasses.fields(TrainOptimizerConfig)}
    names: list[str] = []

    selected_defaults = _selected_optimizer_defaults(cfg)
    for key in selected_defaults.keys():
        if key in field_defs and key not in _OPTIMIZER_HIDDEN_FIELDS and key not in names:
            names.append(key)

    baseline = TrainOptimizerConfig(optimizer=_nested(cfg, "optimizer", "optimizer") or Optimizer.ADAMW)
    for field_def in dataclasses.fields(TrainOptimizerConfig):
        key = field_def.name
        if key in _OPTIMIZER_HIDDEN_FIELDS:
            continue
        current = _nested(cfg, "optimizer", key)
        default = getattr(baseline, key, None)
        if current != default and key not in names:
            names.append(key)

    for key in _OPTIMIZER_COMMON_FIELDS:
        if key in field_defs and key not in names:
            names.append(key)

    return names


def _set_optimizer_status(text: str) -> None:
    """Update optimizer status text if the widget exists."""
    for tag in (_OPTIMIZER_STATUS_TAG, _OPTIMIZER_WINDOW_STATUS_TAG):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)


def _rebuild_optimizer_param_list(
    ui: UIState,
    container_tag: str = _OPTIMIZER_WINDOW_LIST_TAG,
) -> None:
    """Render dynamic optimizer parameter controls for the selected optimizer."""
    if not dpg.does_item_exist(container_tag):
        return

    dpg.delete_item(container_tag, children_only=True)
    cfg = ui.config
    field_defs = {f.name: f for f in dataclasses.fields(TrainOptimizerConfig)}
    names = _optimizer_fields_to_show(cfg)
    if not names:
        dpg.add_text("No optimizer parameters available.", parent=container_tag)
        return

    dpg.add_text(
        f"Selected: {_optimizer_name(cfg)}",
        parent=container_tag,
        color=(130, 130, 145),
    )
    dpg.add_spacer(height=4, parent=container_tag)

    for key in names:
        field_def = field_defs.get(key)
        if field_def is None:
            continue

        current_value = _nested(cfg, "optimizer", key)
        type_hint = _optimizer_type_hint(field_def.type, current_value)
        tag = f"optimizer.{key}"
        ui.register(tag, type_hint=type_hint)
        callback = ui.make_callback(tag)

        if type_hint is bool:
            labeled_checkbox(
                _optimizer_label(key),
                tag=tag,
                default_value=bool(current_value),
                callback=callback,
                parent=container_tag,
            )
        elif type_hint is int:
            labeled_int(
                _optimizer_label(key),
                tag=tag,
                default_value=int(current_value) if current_value is not None else 0,
                callback=callback,
                parent=container_tag,
            )
        elif type_hint is float:
            labeled_float(
                _optimizer_label(key),
                tag=tag,
                default_value=float(current_value) if current_value is not None else 0.0,
                callback=callback,
                parent=container_tag,
            )
        else:
            labeled_input(
                _optimizer_label(key),
                tag=tag,
                default_value="" if current_value is None else str(current_value),
                callback=callback,
                parent=container_tag,
            )


def _open_optimizer_window(ui: UIState) -> None:
    """Open (or focus) the advanced optimizer settings popup."""
    if dpg.does_item_exist(_OPTIMIZER_WINDOW_TAG):
        dpg.show_item(_OPTIMIZER_WINDOW_TAG)
        dpg.focus_item(_OPTIMIZER_WINDOW_TAG)
        _rebuild_optimizer_param_list(ui, _OPTIMIZER_WINDOW_LIST_TAG)
        return

    with dpg.window(
        tag=_OPTIMIZER_WINDOW_TAG,
        label="Optimizer Settings",
        width=scaled(760),
        height=scaled(620),
        no_collapse=False,
    ):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Load Defaults",
                callback=lambda _s, _a, _u: _load_optimizer_defaults(ui),
            )
        dpg.add_text(
            "",
            tag=_OPTIMIZER_WINDOW_STATUS_TAG,
            color=(130, 130, 145),
        )
        dpg.add_child_window(
            tag=_OPTIMIZER_WINDOW_LIST_TAG,
            autosize_x=True,
            height=-1,
            border=False,
        )

    _rebuild_optimizer_param_list(ui, _OPTIMIZER_WINDOW_LIST_TAG)


def _on_optimizer_changed(ui: UIState) -> None:
    """Persist optimizer selection and refresh dynamic parameter controls."""
    ui.widget_to_config("optimizer.optimizer")
    _set_optimizer_status("")
    _rebuild_optimizer_param_list(ui, _OPTIMIZER_WINDOW_LIST_TAG)


def _load_optimizer_defaults(ui: UIState) -> None:
    """Load per-optimizer defaults from config; fallback to dataclass defaults."""
    cfg = ui.config
    selected_defaults = _selected_optimizer_defaults(cfg)
    opt_name = _optimizer_name(cfg) or "optimizer"

    if selected_defaults:
        for key, value in selected_defaults.items():
            if key in _OPTIMIZER_HIDDEN_FIELDS:
                continue
            if hasattr(cfg.optimizer, key):
                setattr(cfg.optimizer, key, value)
        _set_optimizer_status(f"Loaded preset defaults for {opt_name}.")
    else:
        baseline = TrainOptimizerConfig(optimizer=_nested(cfg, "optimizer", "optimizer") or Optimizer.ADAMW)
        for field_def in dataclasses.fields(TrainOptimizerConfig):
            key = field_def.name
            if key in _OPTIMIZER_HIDDEN_FIELDS:
                continue
            setattr(cfg.optimizer, key, getattr(baseline, key))
        _set_optimizer_status(f"No preset defaults for {opt_name}; reset to built-in defaults.")

    ui.sync_from_config()
    _rebuild_optimizer_param_list(ui, _OPTIMIZER_WINDOW_LIST_TAG)


# ---------------------------------------------------------------------------
# Column builders
# ---------------------------------------------------------------------------

def _build_left(parent: int | str, ui: UIState) -> None:
    """Left column -- Optimizer & LR, Training, Text Encoder, Embedding."""
    cfg = ui.config

    # ---- Optimizer & LR section ----
    with section("Optimizer & LR", parent=parent):
        with dpg.group(horizontal=True):
            lbl = dpg.add_text("Optimizer", wrap=LABEL_WIDTH)
            tooltip(lbl, "The type of optimizer")
            dpg.add_combo(
                items=enum_values(Optimizer),
                tag="optimizer.optimizer",
                default_value=_enum_default(cfg.optimizer, "optimizer"),
                callback=lambda _s, _a, _u: _on_optimizer_changed(ui),
                width=INPUT_WIDTH - scaled(44),
            )
            opt_btn = dpg.add_button(
                label="...",
                width=scaled(34),
                callback=lambda _s, _a, _u: _open_optimizer_window(ui),
            )
            tooltip(opt_btn, "Advanced optimizer parameters")
        with dpg.group(horizontal=True):
            lbl = dpg.add_text("LR Scheduler", wrap=LABEL_WIDTH)
            tooltip(lbl, "Learning rate scheduler used during training")
            dpg.add_combo(
                items=enum_values(LearningRateScheduler),
                tag="learning_rate_scheduler",
                default_value=_enum_default(cfg, "learning_rate_scheduler"),
                callback=lambda _s, _a, _u: _on_scheduler_changed(ui),
                width=INPUT_WIDTH - scaled(44),
            )
            sched_btn = dpg.add_button(
                tag=_SCHEDULER_ADV_BUTTON_TAG,
                label="...",
                width=scaled(34),
                callback=lambda _s, _a, _u: _open_scheduler_window(ui),
            )
            tooltip(sched_btn, "Scheduler advanced parameters")
            _update_scheduler_adv_button_state(ui)
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
        with dpg.group(horizontal=True):
            lbl = dpg.add_text("Gradient Checkpointing", wrap=LABEL_WIDTH)
            tooltip(lbl, "Gradient checkpointing strategy")
            dpg.add_combo(
                items=enum_values(GradientCheckpointingMethod),
                tag="gradient_checkpointing",
                default_value=_enum_default(cfg, "gradient_checkpointing"),
                callback=ui.make_callback("gradient_checkpointing"),
                width=INPUT_WIDTH - scaled(44),
            )
            gc_btn = dpg.add_button(
                label="...",
                width=scaled(34),
                callback=lambda _s, _a, _u: _open_offloading_window(ui),
            )
            tooltip(gc_btn, "Offloading settings")
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
        with dpg.group(horizontal=True):
            lbl = dpg.add_text("Timestep Distribution", wrap=LABEL_WIDTH)
            tooltip(lbl, "Timestep sampling function")
            dpg.add_combo(
                items=enum_values(TimestepDistribution),
                tag="timestep_distribution",
                default_value=_enum_default(cfg, "timestep_distribution"),
                callback=ui.make_callback("timestep_distribution"),
                width=INPUT_WIDTH - scaled(44),
            )
            ts_btn = dpg.add_button(
                label="...",
                width=scaled(34),
                callback=lambda _s, _a, _u: _open_timestep_window(ui),
            )
            tooltip(ts_btn, "Open timestep distribution advanced settings")
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
