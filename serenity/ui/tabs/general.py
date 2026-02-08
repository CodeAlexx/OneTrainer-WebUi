"""General settings tab for the Serenity training UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_dir,
    labeled_input,
    labeled_int,
    section,
    time_entry,
)

__all__ = ["build_general_tab"]


def build_general_tab(ui_state: UIState) -> None:
    """Build the General settings tab, mirroring OneTrainer's general tab."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Workspace ----------------------------------------------------------
    with section("Workspace"):
        labeled_dir(
            "Workspace Dir",
            tag="workspace_dir",
            default_value=cfg.workspace_dir,
            callback=cb("workspace_dir"),
            tip="Working directory for training run artifacts",
        )
        labeled_dir(
            "Cache Dir",
            tag="cache_dir",
            default_value=cfg.cache_dir,
            callback=cb("cache_dir"),
            tip="Directory for latent and other caches",
        )
        labeled_checkbox(
            "Continue Last Backup",
            tag="continue_last_backup",
            default_value=cfg.continue_last_backup,
            callback=cb("continue_last_backup"),
            tip="Resume training from the most recent backup",
        )
        labeled_checkbox(
            "Only Cache",
            tag="only_cache",
            default_value=cfg.only_cache,
            callback=cb("only_cache"),
            tip="Only build caches, do not train",
        )

    # -- Tensorboard --------------------------------------------------------
    with section("Tensorboard"):
        labeled_checkbox(
            "Tensorboard",
            tag="tensorboard",
            default_value=cfg.tensorboard,
            callback=cb("tensorboard"),
            tip="Enable Tensorboard logging",
        )
        labeled_checkbox(
            "Always On",
            tag="tensorboard_always_on",
            default_value=cfg.tensorboard_always_on,
            callback=cb("tensorboard_always_on"),
            tip="Keep Tensorboard running even when not training",
        )
        labeled_checkbox(
            "Expose",
            tag="tensorboard_expose",
            default_value=cfg.tensorboard_expose,
            callback=cb("tensorboard_expose"),
            tip="Expose Tensorboard to external connections (0.0.0.0)",
        )
        labeled_int(
            "Port",
            tag="tensorboard_port",
            default_value=cfg.tensorboard_port,
            callback=cb("tensorboard_port"),
            tip="Tensorboard server port",
            min_value=1024,
            max_value=65535,
        )

    # -- Device Settings ----------------------------------------------------
    with section("Device"):
        labeled_int(
            "Dataloader Threads",
            tag="dataloader_threads",
            default_value=cfg.dataloader_threads,
            callback=cb("dataloader_threads"),
            tip="Number of dataloader worker threads",
            min_value=0,
            max_value=32,
        )
        labeled_input(
            "Train Device",
            tag="train_device",
            default_value=cfg.train_device,
            callback=cb("train_device"),
            tip="Primary training device (e.g. cuda, cuda:0)",
        )
        labeled_input(
            "Temp Device",
            tag="temp_device",
            default_value=cfg.temp_device,
            callback=cb("temp_device"),
            tip="Device for temporary tensors (e.g. cpu)",
        )

    # -- Validation ---------------------------------------------------------
    with section("Validation"):
        labeled_checkbox(
            "Validation",
            tag="validation",
            default_value=cfg.validation,
            callback=cb("validation"),
            tip="Enable validation during training",
        )
        time_entry(
            "Validate After",
            value_tag="validate_after",
            unit_tag="validate_after_unit",
            default_value=cfg.validate_after,
            default_unit=cfg.validate_after_unit.value
            if hasattr(cfg.validate_after_unit, "value")
            else str(cfg.validate_after_unit),
            callback=cb("validate_after"),
            tip="Run validation after this interval",
        )

    # -- Debug --------------------------------------------------------------
    with section("Debug", default_open=False):
        labeled_checkbox(
            "Debug Mode",
            tag="debug_mode",
            default_value=cfg.debug_mode,
            callback=cb("debug_mode"),
            tip="Enable debug output and extra logging",
        )
        labeled_dir(
            "Debug Dir",
            tag="debug_dir",
            default_value=cfg.debug_dir,
            callback=cb("debug_dir"),
            tip="Directory for debug artifacts",
        )

    # -- Multi-GPU ----------------------------------------------------------
    with section("Multi-GPU", default_open=False):
        labeled_checkbox(
            "Multi-GPU",
            tag="multi_gpu",
            default_value=cfg.multi_gpu,
            callback=cb("multi_gpu"),
            tip="Enable multi-GPU distributed training",
        )
        labeled_input(
            "Device Indexes",
            tag="device_indexes",
            default_value=cfg.device_indexes,
            callback=cb("device_indexes"),
            tip="Comma-separated GPU device indexes (e.g. 0,1,2)",
        )
        labeled_checkbox(
            "Fused Gradient Reduce",
            tag="fused_gradient_reduce",
            default_value=cfg.fused_gradient_reduce,
            callback=cb("fused_gradient_reduce"),
            tip="Use fused gradient all-reduce for multi-GPU",
        )
        labeled_checkbox(
            "Async Gradient Reduce",
            tag="async_gradient_reduce",
            default_value=cfg.async_gradient_reduce,
            callback=cb("async_gradient_reduce"),
            tip="Use asynchronous gradient reduction",
        )
        labeled_int(
            "Async Reduce Buffer",
            tag="async_gradient_reduce_buffer",
            default_value=cfg.async_gradient_reduce_buffer,
            callback=cb("async_gradient_reduce_buffer"),
            tip="Buffer size for async gradient reduction (MB)",
            min_value=1,
            max_value=10000,
        )

    # -- Register all bindings with UIState ---------------------------------
    for tag in [
        "workspace_dir",
        "cache_dir",
        "continue_last_backup",
        "only_cache",
        "debug_mode",
        "debug_dir",
        "tensorboard",
        "tensorboard_always_on",
        "tensorboard_expose",
        "tensorboard_port",
        "validation",
        "validate_after",
        "validate_after_unit",
        "dataloader_threads",
        "train_device",
        "temp_device",
        "multi_gpu",
        "device_indexes",
        "fused_gradient_reduce",
        "async_gradient_reduce",
        "async_gradient_reduce_buffer",
    ]:
        ui_state.register(tag)
