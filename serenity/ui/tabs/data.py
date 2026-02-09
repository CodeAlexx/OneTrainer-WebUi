"""Data settings tab for the Serenity training UI."""

from __future__ import annotations


from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    section,
)

__all__ = ["build_data_tab"]


def build_data_tab(ui_state: UIState) -> None:
    """Build the Data settings tab with caching and bucketing options."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Caching & Bucketing ------------------------------------------------
    with section("Caching & Bucketing"):
        labeled_checkbox(
            "Aspect Ratio Bucketing",
            tag="aspect_ratio_bucketing",
            default_value=cfg.aspect_ratio_bucketing,
            callback=cb("aspect_ratio_bucketing"),
            tip="Group images by aspect ratio to reduce padding waste",
        )
        labeled_checkbox(
            "Latent Caching",
            tag="latent_caching",
            default_value=cfg.latent_caching,
            callback=cb("latent_caching"),
            tip="Cache VAE-encoded latents to disk for faster epochs",
        )
        labeled_checkbox(
            "Clear Cache Before Training",
            tag="clear_cache_before_training",
            default_value=cfg.clear_cache_before_training,
            callback=cb("clear_cache_before_training"),
            tip="Delete existing latent cache before starting training",
        )

    # -- Register bindings --------------------------------------------------
    for tag in [
        "aspect_ratio_bucketing",
        "latent_caching",
        "clear_cache_before_training",
    ]:
        ui_state.register(tag)
