"""Model settings tab for the Serenity training UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_file,
    labeled_separator,
)

__all__ = ["build_model_tab"]

# Common dtype options for weight/output combos
_DTYPE_ITEMS = ["FLOAT_32", "BFLOAT_16", "FLOAT_16", "FLOAT_8", "NFLOAT_4"]


def build_model_tab(ui_state: UIState) -> None:
    """Build the Model settings tab, mirroring OneTrainer's model tab."""
    cfg = ui_state.config
    cb = ui_state.make_callback

    # -- Base Model ---------------------------------------------------------
    labeled_separator("Base Model")

    labeled_file(
        "Base Model",
        tag="base_model_name",
        default_value=cfg.base_model_name,
        callback=cb("base_model_name"),
        tip="Path to the base model checkpoint or diffusers directory",
    )
    labeled_checkbox(
        "Compile Transformer Blocks",
        tag="compile",
        default_value=cfg.compile,
        callback=cb("compile"),
        tip="Use torch.compile on transformer blocks for faster training",
    )

    # -- Component Data Types -----------------------------------------------
    labeled_separator("Component Data Types")

    labeled_combo(
        "Transformer Data Type",
        items=_DTYPE_ITEMS,
        tag="transformer.weight_dtype",
        default_value=cfg.transformer.weight_dtype.value
        if hasattr(cfg.transformer.weight_dtype, "value")
        else str(cfg.transformer.weight_dtype),
        callback=cb("transformer.weight_dtype"),
        tip="Weight precision for the transformer / DiT",
    )
    labeled_combo(
        "UNet Data Type",
        items=_DTYPE_ITEMS,
        tag="unet.weight_dtype",
        default_value=cfg.unet.weight_dtype.value
        if hasattr(cfg.unet.weight_dtype, "value")
        else str(cfg.unet.weight_dtype),
        callback=cb("unet.weight_dtype"),
        tip="Weight precision for the UNet",
    )
    labeled_combo(
        "Text Encoder Data Type",
        items=_DTYPE_ITEMS,
        tag="text_encoder.weight_dtype",
        default_value=cfg.text_encoder.weight_dtype.value
        if hasattr(cfg.text_encoder.weight_dtype, "value")
        else str(cfg.text_encoder.weight_dtype),
        callback=cb("text_encoder.weight_dtype"),
        tip="Weight precision for text encoder 1",
    )
    labeled_combo(
        "Text Encoder 2 Data Type",
        items=_DTYPE_ITEMS,
        tag="text_encoder_2.weight_dtype",
        default_value=cfg.text_encoder_2.weight_dtype.value
        if hasattr(cfg.text_encoder_2.weight_dtype, "value")
        else str(cfg.text_encoder_2.weight_dtype),
        callback=cb("text_encoder_2.weight_dtype"),
        tip="Weight precision for text encoder 2",
    )
    labeled_combo(
        "VAE Data Type",
        items=_DTYPE_ITEMS,
        tag="vae.weight_dtype",
        default_value=cfg.vae.weight_dtype.value
        if hasattr(cfg.vae.weight_dtype, "value")
        else str(cfg.vae.weight_dtype),
        callback=cb("vae.weight_dtype"),
        tip="Weight precision for the VAE",
    )

    # -- Model Overrides ----------------------------------------------------
    labeled_separator("Model Overrides")

    labeled_file(
        "Override Transformer / GGUF",
        tag="transformer.model_name",
        default_value=cfg.transformer.model_name,
        callback=cb("transformer.model_name"),
        tip="Path to an alternative transformer checkpoint or GGUF file",
    )
    labeled_file(
        "VAE Override",
        tag="vae.model_name",
        default_value=cfg.vae.model_name,
        callback=cb("vae.model_name"),
        tip="Path to an alternative VAE checkpoint",
    )

    # -- Output -------------------------------------------------------------
    labeled_separator("Output")

    labeled_file(
        "Output Destination",
        tag="output_model_destination",
        default_value=cfg.output_model_destination,
        callback=cb("output_model_destination"),
        tip="Path for the saved output model",
    )
    labeled_combo(
        "Output Data Type",
        items=_DTYPE_ITEMS,
        tag="output_dtype",
        default_value=cfg.output_dtype.value
        if hasattr(cfg.output_dtype, "value")
        else str(cfg.output_dtype),
        callback=cb("output_dtype"),
        tip="Data type for the saved model weights",
    )
    labeled_combo(
        "Output Format",
        items=["SAFETENSORS", "DIFFUSERS"],
        tag="output_model_format",
        default_value=cfg.output_model_format.value
        if hasattr(cfg.output_model_format, "value")
        else str(cfg.output_model_format),
        callback=cb("output_model_format"),
        tip="File format for the saved model",
    )

    # -- Register all bindings with UIState ---------------------------------
    for tag in [
        "base_model_name",
        "compile",
        "transformer.weight_dtype",
        "unet.weight_dtype",
        "text_encoder.weight_dtype",
        "text_encoder_2.weight_dtype",
        "vae.weight_dtype",
        "transformer.model_name",
        "vae.model_name",
        "output_model_destination",
        "output_dtype",
        "output_model_format",
    ]:
        ui_state.register(tag)
