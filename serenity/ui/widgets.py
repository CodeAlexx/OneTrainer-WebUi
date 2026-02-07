"""Reusable DearPyGui widget helpers for Serenity UI.

These helpers mirror OneTrainer's `components.py` pattern: each creates a
labeled widget bound to a config field, returning the widget tag so callers
can manipulate it further.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import dearpygui.dearpygui as dpg

__all__ = [
    "labeled_input",
    "labeled_float",
    "labeled_int",
    "labeled_checkbox",
    "labeled_combo",
    "labeled_combo_kv",
    "labeled_file",
    "labeled_dir",
    "labeled_separator",
    "section_header",
    "tooltip",
    "time_entry",
]

# Default label column width (sized for 4K readability with 22px font)
LABEL_WIDTH = 280

# Max width for input fields to prevent full-screen stretch on 4K
INPUT_WIDTH = 420


def tooltip(parent: int | str, text: str) -> None:
    """Attach a hover tooltip to *parent*."""
    with dpg.tooltip(parent):
        dpg.add_text(text, wrap=500)


def section_header(label: str, parent: int | str = 0) -> int:
    """Add a collapsing header / section divider."""
    return dpg.add_collapsing_header(label=label, parent=parent, default_open=True)


def labeled_separator(label: str, parent: int | str = 0) -> None:
    """Visual separator with a colored label and breathing room."""
    dpg.add_spacer(height=10, parent=parent)
    dpg.add_separator(parent=parent)
    dpg.add_spacer(height=4, parent=parent)
    dpg.add_text(label, parent=parent, color=(86, 156, 240))
    dpg.add_spacer(height=6, parent=parent)


def labeled_input(
    label: str,
    *,
    tag: str = "",
    default_value: str = "",
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
    hint: str = "",
) -> int:
    """Label + text input on same row."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "default_value": default_value,
            "width": width if width != 0 else INPUT_WIDTH,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        if hint:
            kwargs["hint"] = hint
        item = dpg.add_input_text(**kwargs)
    return item


def labeled_float(
    label: str,
    *,
    tag: str = "",
    default_value: float = 0.0,
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
    min_value: float = 0.0,
    max_value: float = 0.0,
    format_str: str = "%.6f",
    step: float = 0.0,
) -> int:
    """Label + float input."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "default_value": default_value,
            "width": width if width != 0 else INPUT_WIDTH,
            "format": format_str,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        if step:
            kwargs["step"] = step
        if min_value != max_value:
            kwargs["min_value"] = min_value
            kwargs["max_value"] = max_value
            kwargs["min_clamped"] = True
            kwargs["max_clamped"] = True
        item = dpg.add_input_float(**kwargs)
    return item


def labeled_int(
    label: str,
    *,
    tag: str = "",
    default_value: int = 0,
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
    min_value: int = 0,
    max_value: int = 0,
) -> int:
    """Label + integer input."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "default_value": default_value,
            "width": width if width != 0 else INPUT_WIDTH,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        if min_value != max_value:
            kwargs["min_value"] = min_value
            kwargs["max_value"] = max_value
            kwargs["min_clamped"] = True
            kwargs["max_clamped"] = True
        item = dpg.add_input_int(**kwargs)
    return item


def labeled_checkbox(
    label: str,
    *,
    tag: str = "",
    default_value: bool = False,
    callback: Callable | None = None,
    parent: int | str = 0,
    tip: str = "",
) -> int:
    """Label + checkbox."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {"default_value": default_value}
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        item = dpg.add_checkbox(**kwargs)
    return item


def labeled_combo(
    label: str,
    items: list[str],
    *,
    tag: str = "",
    default_value: str = "",
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
) -> int:
    """Label + dropdown combo."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "items": items,
            "default_value": default_value,
            "width": width if width != 0 else INPUT_WIDTH,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        item = dpg.add_combo(**kwargs)
    return item


def labeled_combo_kv(
    label: str,
    options: list[tuple[str, Any]],
    *,
    tag: str = "",
    default_value: str = "",
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = -1,
    tip: str = "",
) -> int:
    """Label + dropdown from (display_name, enum_value) pairs.

    The combo shows display names; the callback receives the display string.
    Use the returned options list to map back to values.
    """
    display_names = [name for name, _val in options]
    return labeled_combo(
        label,
        display_names,
        tag=tag,
        default_value=default_value,
        callback=callback,
        parent=parent,
        width=width,
        tip=tip,
    )


def labeled_file(
    label: str,
    *,
    tag: str = "",
    default_value: str = "",
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
) -> int:
    """Label + text input + browse button for files."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "default_value": default_value,
            "width": width if width > 0 else INPUT_WIDTH - 60,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        item = dpg.add_input_text(**kwargs)

        def _browse() -> None:
            with dpg.file_dialog(
                callback=lambda _s, data: dpg.set_value(item, data["file_path_name"]),
                width=700,
                height=400,
            ):
                dpg.add_file_extension(".*")
                dpg.add_file_extension(".safetensors", color=(0, 255, 0))
                dpg.add_file_extension(".ckpt", color=(0, 255, 0))
                dpg.add_file_extension(".gguf", color=(0, 200, 200))

        dpg.add_button(label="...", callback=_browse, width=50)
    return item


def labeled_dir(
    label: str,
    *,
    tag: str = "",
    default_value: str = "",
    callback: Callable | None = None,
    parent: int | str = 0,
    width: int = 0,
    tip: str = "",
) -> int:
    """Label + text input + browse button for directories."""
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        kwargs: dict[str, Any] = {
            "default_value": default_value,
            "width": width if width > 0 else INPUT_WIDTH - 60,
        }
        if tag:
            kwargs["tag"] = tag
        if callback:
            kwargs["callback"] = callback
        item = dpg.add_input_text(**kwargs)

        def _browse() -> None:
            with dpg.file_dialog(
                directory_selector=True,
                callback=lambda _s, data: dpg.set_value(item, data["file_path_name"]),
                width=700,
                height=400,
            ):
                pass

        dpg.add_button(label="...", callback=_browse, width=50)
    return item


def time_entry(
    label: str,
    *,
    value_tag: str = "",
    unit_tag: str = "",
    default_value: float = 0.0,
    default_unit: str = "NEVER",
    unit_items: list[str] | None = None,
    callback: Callable | None = None,
    parent: int | str = 0,
    tip: str = "",
) -> tuple[int, int]:
    """A value input + time-unit combo (e.g., "10 MINUTE")."""
    if unit_items is None:
        unit_items = ["NEVER", "EPOCH", "STEP", "SECOND", "MINUTE", "HOUR", "ALWAYS"]
    with dpg.group(horizontal=True, parent=parent):
        t = dpg.add_text(label, wrap=LABEL_WIDTH)
        if tip:
            tooltip(t, tip)
        v_kw: dict[str, Any] = {"default_value": default_value, "width": 160, "format": "%.1f"}
        if value_tag:
            v_kw["tag"] = value_tag
        if callback:
            v_kw["callback"] = callback
        val_item = dpg.add_input_float(**v_kw)

        u_kw: dict[str, Any] = {
            "items": unit_items,
            "default_value": default_unit,
            "width": 160,
        }
        if unit_tag:
            u_kw["tag"] = unit_tag
        if callback:
            u_kw["callback"] = callback
        unit_item = dpg.add_combo(**u_kw)
    return val_item, unit_item


def enum_values(enum_cls: type[Enum]) -> list[str]:
    """Get display-friendly values from a str Enum."""
    return [e.value for e in enum_cls]
