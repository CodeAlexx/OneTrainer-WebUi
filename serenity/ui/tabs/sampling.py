"""Sampling settings tab for the Serenity training UI.

Mirrors OneTrainer's sampling tab: global schedule settings at the top,
followed by a dynamic list of sample prompt definitions.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.core.sample_config import SampleConfig
from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_float,
    labeled_int,
    section,
    time_entry,
    tooltip,
)

__all__ = ["build_sampling_tab"]

# Stable tag for the scrollable sample list container.
SAMPLE_LIST_TAG = "sample_list"

# Supported output formats for sample images.
SAMPLE_IMAGE_FORMATS = ["PNG", "JPG"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_sampling_tab(ui_state: UIState) -> None:
    """Build the Sampling tab with global settings and a sample definition list."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Global sampling schedule -------------------------------------------
    with section("Sampling Schedule"):
        time_entry(
            "Sample After",
            value_tag="sample_after",
            unit_tag="sample_after_unit",
            default_value=cfg.sample_after,
            default_unit=cfg.sample_after_unit.value
            if hasattr(cfg.sample_after_unit, "value")
            else str(cfg.sample_after_unit),
            callback=cb("sample_after"),
            tip="Generate samples after this interval",
        )
        labeled_int(
            "Skip First",
            tag="sample_skip_first",
            default_value=cfg.sample_skip_first,
            callback=cb("sample_skip_first"),
            tip="Skip sample generation for the first N intervals",
            min_value=0,
            max_value=10000,
        )
        labeled_combo(
            "Image Format",
            SAMPLE_IMAGE_FORMATS,
            tag="sample_image_format",
            default_value=cfg.sample_image_format.value
            if hasattr(cfg.sample_image_format, "value")
            else str(cfg.sample_image_format),
            callback=cb("sample_image_format"),
            tip="Output format for generated sample images",
        )
        labeled_checkbox(
            "Non-EMA Sampling",
            tag="non_ema_sampling",
            default_value=cfg.non_ema_sampling,
            callback=cb("non_ema_sampling"),
            tip="Also generate samples from the non-EMA model weights",
        )
        labeled_checkbox(
            "Samples to Tensorboard",
            tag="samples_to_tensorboard",
            default_value=cfg.samples_to_tensorboard,
            callback=cb("samples_to_tensorboard"),
            tip="Write generated samples to Tensorboard image log",
        )

    # -- Sample definitions -------------------------------------------------
    with section("Sample Definitions"):
        with dpg.group(horizontal=True):
            btn = dpg.add_button(
                label="Add Sample",
                callback=lambda: _add_sample(ui_state),
            )
            tooltip(btn, "Append a new sample prompt definition")
            dpg.add_text(
                tag="sample_count",
                default_value=f"Samples: {len(ui_state.config.samples or [])}",
            )

        dpg.add_spacer(height=4)

        with dpg.child_window(tag=SAMPLE_LIST_TAG, autosize_x=True, height=-1):
            _rebuild_sample_list(ui_state)

    # Register global bindings
    for tag in [
        "sample_after",
        "sample_after_unit",
        "sample_skip_first",
        "sample_image_format",
        "non_ema_sampling",
        "samples_to_tensorboard",
    ]:
        ui_state.register(tag)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_count(ui_state: UIState) -> None:
    """Refresh the sample count label."""
    try:
        count = len(ui_state.config.samples or [])
        dpg.set_value("sample_count", f"Samples: {count}")
    except Exception:
        pass


def _add_sample(ui_state: UIState) -> None:
    """Append a blank sample definition and rebuild the list."""
    if ui_state.config.samples is None:
        ui_state.config.samples = []
    sample = SampleConfig()
    ui_state.config.samples.append(sample)
    _rebuild_sample_list(ui_state)
    _update_count(ui_state)


def _remove_sample(ui_state: UIState, index: int) -> None:
    """Remove a sample by index and rebuild the list."""
    samples = ui_state.config.samples
    if samples is None:
        return
    try:
        samples.pop(index)
    except IndexError:
        return
    _rebuild_sample_list(ui_state)
    _update_count(ui_state)


def _rebuild_sample_list(ui_state: UIState) -> None:
    """Clear and redraw every sample item inside the scroll region."""
    dpg.delete_item(SAMPLE_LIST_TAG, children_only=True)
    samples = ui_state.config.samples or []
    for i, sample in enumerate(samples):
        _build_sample_item(ui_state, i, sample)


def _build_sample_item(ui_state: UIState, index: int, sample: SampleConfig) -> None:
    """Render a single collapsible sample entry."""
    # Build a short header from the first line of the prompt.
    prompt_preview = (sample.prompt or "").split("\n", 1)[0][:60]
    header_label = prompt_preview if prompt_preview else f"Sample {index}"
    with dpg.collapsing_header(
        label=header_label,
        parent=SAMPLE_LIST_TAG,
        default_open=True,
    ):
        # -- Enabled --
        labeled_checkbox(
            "Enabled",
            default_value=sample.enabled,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "enabled", val,
            ),
            tip="Include this sample definition during generation",
        )

        # -- Prompt (multiline) --
        dpg.add_text("Prompt")
        dpg.add_input_text(
            default_value=sample.prompt,
            multiline=True,
            height=80,
            width=-1,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "prompt", val,
            ),
            on_enter=False,
        )

        # -- Negative prompt (multiline) --
        dpg.add_text("Negative Prompt")
        dpg.add_input_text(
            default_value=sample.negative_prompt,
            multiline=True,
            height=60,
            width=-1,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "negative_prompt", val,
            ),
            on_enter=False,
        )

        dpg.add_spacer(height=2)

        # -- Dimensions --
        with dpg.group(horizontal=True):
            labeled_int(
                "Width",
                default_value=sample.width,
                callback=lambda _s, val: _set_sample_field(
                    ui_state, index, "width", int(val),
                ),
                min_value=64,
                max_value=4096,
                tip="Output image width in pixels",
            )
        with dpg.group(horizontal=True):
            labeled_int(
                "Height",
                default_value=sample.height,
                callback=lambda _s, val: _set_sample_field(
                    ui_state, index, "height", int(val),
                ),
                min_value=64,
                max_value=4096,
                tip="Output image height in pixels",
            )

        # -- Seed --
        labeled_int(
            "Seed",
            default_value=sample.seed,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "seed", int(val),
            ),
            tip="Deterministic seed for reproducible samples",
        )
        labeled_checkbox(
            "Random Seed",
            default_value=sample.random_seed,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "random_seed", val,
            ),
            tip="Use a random seed each time instead of the fixed seed above",
        )

        # -- Inference parameters --
        labeled_int(
            "Inference Steps",
            default_value=sample.num_inference_steps,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "num_inference_steps", int(val),
            ),
            min_value=1,
            max_value=200,
            tip="Number of diffusion sampling steps",
        )
        labeled_float(
            "Guidance Scale",
            default_value=sample.guidance_scale,
            callback=lambda _s, val: _set_sample_field(
                ui_state, index, "guidance_scale", float(val),
            ),
            min_value=0.0,
            max_value=100.0,
            format_str="%.1f",
            tip="Classifier-free guidance strength",
        )

        dpg.add_spacer(height=4)

        # -- Remove button --
        dpg.add_button(
            label="Remove",
            callback=lambda _s, _a, idx=index: _remove_sample(ui_state, idx),
        )


def _set_sample_field(
    ui_state: UIState, index: int, field: str, value: object,
) -> None:
    """Write *value* into ``ui_state.config.samples[index].<field>``."""
    samples = ui_state.config.samples
    if samples is None:
        return
    try:
        sample = samples[index]
    except IndexError:
        return
    setattr(sample, field, value)
    # Rebuild when prompt changes so the header preview stays current.
    if field == "prompt":
        _rebuild_sample_list(ui_state)
