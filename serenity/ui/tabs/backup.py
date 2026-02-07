"""Backup and save settings tab for the Serenity training UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_input,
    labeled_int,
    labeled_separator,
    time_entry,
)

__all__ = ["build_backup_tab"]


def build_backup_tab(ui_state: UIState) -> None:
    """Build the Backup & Save settings tab."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Backup Settings ----------------------------------------------------
    labeled_separator("Backup Settings")

    time_entry(
        "Backup After",
        value_tag="backup_after",
        unit_tag="backup_after_unit",
        default_value=cfg.backup_after,
        default_unit=cfg.backup_after_unit.value
        if hasattr(cfg.backup_after_unit, "value")
        else str(cfg.backup_after_unit),
        callback=cb("backup_after"),
        tip="Create a backup after this interval",
    )
    labeled_checkbox(
        "Rolling Backup",
        tag="rolling_backup",
        default_value=cfg.rolling_backup,
        callback=cb("rolling_backup"),
        tip="Keep only the N most recent backups",
    )
    labeled_int(
        "Rolling Backup Count",
        tag="rolling_backup_count",
        default_value=cfg.rolling_backup_count,
        callback=cb("rolling_backup_count"),
        tip="Number of rolling backups to keep",
        min_value=1,
        max_value=100,
    )
    labeled_checkbox(
        "Backup Before Save",
        tag="backup_before_save",
        default_value=cfg.backup_before_save,
        callback=cb("backup_before_save"),
        tip="Create a backup before each permanent save",
    )

    # -- Save Settings ------------------------------------------------------
    labeled_separator("Save Settings")

    time_entry(
        "Save Every",
        value_tag="save_every",
        unit_tag="save_every_unit",
        default_value=float(cfg.save_every),
        default_unit=cfg.save_every_unit.value
        if hasattr(cfg.save_every_unit, "value")
        else str(cfg.save_every_unit),
        callback=cb("save_every"),
        tip="Save a permanent checkpoint at this interval (0 = disabled)",
    )
    labeled_int(
        "Save Skip First",
        tag="save_skip_first",
        default_value=cfg.save_skip_first,
        callback=cb("save_skip_first"),
        tip="Skip saving for the first N intervals",
        min_value=0,
        max_value=10000,
    )
    labeled_input(
        "Save Filename Prefix",
        tag="save_filename_prefix",
        default_value=cfg.save_filename_prefix,
        callback=cb("save_filename_prefix"),
        tip="Prefix for saved checkpoint filenames",
    )

    # -- Register bindings --------------------------------------------------
    for tag in [
        "backup_after",
        "backup_after_unit",
        "rolling_backup",
        "rolling_backup_count",
        "backup_before_save",
        "save_every",
        "save_every_unit",
        "save_skip_first",
        "save_filename_prefix",
    ]:
        ui_state.register(tag)
