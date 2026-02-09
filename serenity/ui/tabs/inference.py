"""Inference tab -- generate images directly from the TUI.

Two-column layout matching the Training tab.  Left column has model
selection, prompts, and core sampling parameters.  Right column has
resolution, LoRA, advanced settings, and generation output/history.

Generation runs in a background thread using the inference engine
library directly -- no web server or HTTP required.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

from serenity.ui.theme import scaled
from serenity.ui.web.scanner import ModelScanner
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_dir,
    labeled_float,
    labeled_input,
    labeled_int,
    section,
    tooltip,
)

__all__ = ["build_inference_tab"]

logger = logging.getLogger(__name__)

# Column width for two-column layout
_COL_W = scaled(560)

# -- Stable widget tags -------------------------------------------------------

TAG_INF_MODEL_PATH = "inf_model_path"
TAG_INF_PROMPT = "inf_prompt"
TAG_INF_NEG_PROMPT = "inf_neg_prompt"
TAG_INF_SEED = "inf_seed"
TAG_INF_STEPS = "inf_steps"
TAG_INF_CFG = "inf_cfg_scale"
TAG_INF_WIDTH = "inf_width"
TAG_INF_HEIGHT = "inf_height"
TAG_INF_SAMPLER = "inf_sampler"
TAG_INF_SCHEDULER = "inf_scheduler"
TAG_INF_GENERATE_BTN = "inf_generate_btn"
TAG_INF_PROGRESS = "inf_progress_bar"
TAG_INF_STATUS = "inf_status"
TAG_INF_OUTPUT_DIR = "inf_output_dir"
TAG_INF_VRAM_LABEL = "inf_vram_label"
TAG_INF_INFO_PANEL = "inf_info_panel"
TAG_INF_HISTORY_LIST = "inf_history_list"

# Advanced
TAG_INF_VRAM_MODE = "inf_vram_mode"
TAG_INF_QUANTIZATION = "inf_quantization"
TAG_INF_ATTENTION = "inf_attention"
TAG_INF_RESCALE_CFG = "inf_rescale_cfg"
TAG_INF_MAHIRO = "inf_mahiro"
TAG_INF_BATCH_COUNT = "inf_batch_count"

# LoRA
TAG_INF_LORA_PATH = "inf_lora_path"
TAG_INF_LORA_WEIGHT = "inf_lora_weight"

# -- Enum value lists (avoid importing torch at module level) ------------------

_SAMPLERS = [
    "euler", "euler_a",
    "dpm_pp_2m", "dpm_pp_2m_sde",
    "dpm_2m", "dpm_2m_sde",
    "lcm", "heun", "deis", "unipc",
]

_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform",
    "simple", "ddim_uniform", "beta",
    "linear_quadratic", "ays",
]

_VRAM_MODES = ["auto", "high", "normal", "low", "no_vram"]
_QUANTIZATIONS = ["none", "int8", "fp8", "bnb_nf4", "bnb_fp4"]
_ATTENTION_BACKENDS = ["auto", "sage", "flash", "xformers", "sdp", "einsum"]

_ASPECT_PRESETS = {
    "1:1 (1024x1024)": (1024, 1024),
    "3:2 (1216x832)": (1216, 832),
    "2:3 (832x1216)": (832, 1216),
    "16:9 (1344x768)": (1344, 768),
    "9:16 (768x1344)": (768, 1344),
    "4:3 (1152x896)": (1152, 896),
    "3:4 (896x1152)": (896, 1152),
}

# -- Module-level state --------------------------------------------------------

_scanner = ModelScanner(root_dirs=["/home/alex/EriDiffusion/Models"])
_model_path_map: dict[str, str] = {}
_lora_path_map: dict[str, str] = {}

_engine: Any = None
_generating = False
_cancel_flag = False
_gen_thread: threading.Thread | None = None
_history: list[dict[str, Any]] = []


# -- Scanner helpers -----------------------------------------------------------


def _fmt_size(mb: float) -> str:
    """Format a size in MB to a human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _populate_models() -> list[str]:
    """Scan and return display labels for checkpoint models."""
    global _model_path_map
    models = _scanner.scan_checkpoints()
    _model_path_map = {}
    items: list[str] = []
    for m in models:
        label = f"{m.name} ({_fmt_size(m.size_mb)})"
        _model_path_map[label] = m.path
        items.append(label)
    return items


def _populate_loras() -> list[str]:
    """Scan and return display labels for LoRA models."""
    global _lora_path_map
    loras = _scanner.scan_loras()
    _lora_path_map = {"(None)": ""}
    items: list[str] = ["(None)"]
    for m in loras:
        label = f"{m.name} ({_fmt_size(m.size_mb)})"
        _lora_path_map[label] = m.path
        items.append(label)
    return items


def _refresh_models(
    _s: Any = None, _a: Any = None, _u: Any = None,
) -> None:
    """Refresh model and LoRA lists from disk."""
    _scanner.refresh()
    model_items = _populate_models()
    lora_items = _populate_loras()
    try:
        dpg.configure_item(TAG_INF_MODEL_PATH, items=model_items)
        if model_items:
            dpg.set_value(TAG_INF_MODEL_PATH, model_items[0])
        dpg.configure_item(TAG_INF_LORA_PATH, items=lora_items)
        dpg.set_value(TAG_INF_LORA_PATH, "(None)")
    except SystemError:
        pass


# -- Public entry point --------------------------------------------------------


def build_inference_tab() -> None:
    """Build the Inference tab inside the current DPG parent."""
    with dpg.group(horizontal=True):
        col_left = dpg.add_child_window(width=_COL_W, border=False)
        col_right = dpg.add_child_window(width=_COL_W, border=False)
    _build_left(col_left)
    _build_right(col_right)


# -- Left column --------------------------------------------------------------


def _build_left(parent: int | str) -> None:
    """Model, prompts, and core sampling parameters."""

    # -- Model section ---------------------------------------------------------
    with section("Model", parent=parent):
        model_items = _populate_models()
        labeled_combo(
            "Model",
            model_items,
            tag=TAG_INF_MODEL_PATH,
            default_value=model_items[0] if model_items else "",
            tip="Select a model from scanned directories",
        )
        dpg.add_button(
            label="Refresh",
            callback=_refresh_models,
            width=scaled(70),
        )

    # -- Prompt section --------------------------------------------------------
    with section("Prompt", parent=parent):
        dpg.add_text("Prompt", parent=parent)
        dpg.add_input_text(
            tag=TAG_INF_PROMPT,
            multiline=True,
            height=scaled(80),
            width=-1,
            parent=parent,
            hint="Describe the image you want to generate...",
        )
        dpg.add_spacer(height=4, parent=parent)
        dpg.add_text("Negative Prompt", parent=parent)
        dpg.add_input_text(
            tag=TAG_INF_NEG_PROMPT,
            multiline=True,
            height=scaled(50),
            width=-1,
            parent=parent,
            hint="What to avoid...",
        )

    # -- Sampling section ------------------------------------------------------
    with section("Sampling", parent=parent):
        labeled_int(
            "Seed",
            tag=TAG_INF_SEED,
            default_value=-1,
            tip="-1 for random seed",
        )
        labeled_int(
            "Steps",
            tag=TAG_INF_STEPS,
            default_value=20,
            min_value=1,
            max_value=200,
        )
        labeled_float(
            "CFG Scale",
            tag=TAG_INF_CFG,
            default_value=7.0,
            min_value=0.0,
            max_value=100.0,
            format_str="%.1f",
        )
        labeled_combo(
            "Sampler",
            _SAMPLERS,
            tag=TAG_INF_SAMPLER,
            default_value="euler",
        )
        labeled_combo(
            "Scheduler",
            _SCHEDULERS,
            tag=TAG_INF_SCHEDULER,
            default_value="normal",
        )

    # -- Generate button & progress -------------------------------------------
    dpg.add_spacer(height=8, parent=parent)
    with dpg.group(horizontal=True, parent=parent):
        btn = dpg.add_button(
            tag=TAG_INF_GENERATE_BTN,
            label="Generate",
            callback=_on_generate_click,
            width=scaled(130),
            height=scaled(28),
        )
        # Apply green theme
        with dpg.theme() as gen_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 160, 95))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (25, 135, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 110, 65))
        dpg.bind_item_theme(btn, gen_theme)

        dpg.add_spacer(width=scaled(12))
        dpg.add_text("Ready", tag=TAG_INF_STATUS, color=(100, 200, 130))

    dpg.add_progress_bar(
        tag=TAG_INF_PROGRESS,
        default_value=0.0,
        width=-1,
        overlay="0 / 0",
        parent=parent,
    )

    dpg.add_spacer(height=4, parent=parent)
    dpg.add_text("", tag=TAG_INF_VRAM_LABEL, color=(130, 130, 145), parent=parent)
    _update_vram_label()


# -- Right column --------------------------------------------------------------


def _build_right(parent: int | str) -> None:
    """Resolution, LoRA, advanced, output, and history."""

    # -- Resolution section ----------------------------------------------------
    with section("Resolution", parent=parent):
        labeled_int(
            "Width",
            tag=TAG_INF_WIDTH,
            default_value=1024,
            min_value=64,
            max_value=4096,
        )
        labeled_int(
            "Height",
            tag=TAG_INF_HEIGHT,
            default_value=1024,
            min_value=64,
            max_value=4096,
        )
        dpg.add_spacer(height=4)
        dpg.add_text("Aspect Presets", color=(130, 130, 145))
        with dpg.group(horizontal=True):
            for label, (w, h) in _ASPECT_PRESETS.items():
                short = label.split(" ")[0]  # "1:1", "3:2", etc.
                dpg.add_button(
                    label=short,
                    callback=lambda _s, _a, dims=(w, h): _set_resolution(*dims),
                    width=scaled(48),
                )

    # -- LoRA section ----------------------------------------------------------
    with section("LoRA", parent=parent, default_open=False):
        lora_items = _populate_loras()
        labeled_combo(
            "LoRA",
            lora_items,
            tag=TAG_INF_LORA_PATH,
            default_value="(None)",
            tip="Select a LoRA from scanned directories",
        )
        labeled_float(
            "LoRA Weight",
            tag=TAG_INF_LORA_WEIGHT,
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
            format_str="%.2f",
        )

    # -- Advanced section ------------------------------------------------------
    with section("Advanced", parent=parent, default_open=False):
        labeled_combo(
            "VRAM Mode",
            _VRAM_MODES,
            tag=TAG_INF_VRAM_MODE,
            default_value="auto",
            tip="HIGH=full VRAM, AUTO=available, NORMAL=70%, LOW=30%, NO_VRAM=CPU offload",
        )
        labeled_combo(
            "Quantization",
            _QUANTIZATIONS,
            tag=TAG_INF_QUANTIZATION,
            default_value="none",
        )
        labeled_combo(
            "Attention",
            _ATTENTION_BACKENDS,
            tag=TAG_INF_ATTENTION,
            default_value="auto",
        )
        labeled_float(
            "RescaleCFG",
            tag=TAG_INF_RESCALE_CFG,
            default_value=0.0,
            min_value=0.0,
            max_value=1.0,
            format_str="%.2f",
            tip="RescaleCFG strength (0 = disabled)",
        )
        labeled_checkbox(
            "MaHiRo",
            tag=TAG_INF_MAHIRO,
            default_value=False,
            tip="Enable MaHiRo post-CFG correction",
        )
        labeled_int(
            "Batch Count",
            tag=TAG_INF_BATCH_COUNT,
            default_value=1,
            min_value=1,
            max_value=16,
            tip="Number of images to generate",
        )

    # -- Output section --------------------------------------------------------
    with section("Output", parent=parent):
        labeled_dir(
            "Output Directory",
            tag=TAG_INF_OUTPUT_DIR,
            default_value=str(Path.home() / "serenity" / "output"),
            tip="Where generated images are saved",
        )

    # -- Generation Info -------------------------------------------------------
    with section("Generation Info", parent=parent, default_open=True):
        dpg.add_input_text(
            tag=TAG_INF_INFO_PANEL,
            multiline=True,
            readonly=True,
            height=scaled(90),
            width=-1,
            default_value="No generation yet.",
        )

    # -- History ---------------------------------------------------------------
    with section("History", parent=parent, default_open=False):
        dpg.add_child_window(
            tag=TAG_INF_HISTORY_LIST,
            autosize_x=True,
            height=scaled(200),
        )


# -- Resolution helpers --------------------------------------------------------


def _set_resolution(w: int, h: int) -> None:
    """Set width and height from an aspect preset button."""
    dpg.set_value(TAG_INF_WIDTH, w)
    dpg.set_value(TAG_INF_HEIGHT, h)


# -- VRAM display --------------------------------------------------------------


def _update_vram_label() -> None:
    """Update the VRAM status label."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            text = f"VRAM: {free_gb:.1f} / {total_gb:.1f} GB free"
        else:
            text = "VRAM: No CUDA device"
    except ImportError:
        text = "VRAM: torch not available"
    except RuntimeError:
        text = "VRAM: unknown"

    try:
        dpg.set_value(TAG_INF_VRAM_LABEL, text)
    except SystemError:
        pass


# -- Generation logic ----------------------------------------------------------


def _set_status(text: str, color: tuple[int, int, int] = (100, 200, 130)) -> None:
    """Update the status label text and color."""
    try:
        dpg.set_value(TAG_INF_STATUS, text)
        dpg.configure_item(TAG_INF_STATUS, color=color)
    except SystemError:
        pass


def _on_generate_click(
    sender: Any = None, app_data: Any = None, user_data: Any = None,
) -> None:
    """Handle the Generate / Cancel button click."""
    global _generating, _cancel_flag

    if _generating:
        # Cancel current generation
        _cancel_flag = True
        _set_status("Cancelling...", (210, 55, 65))
        return

    # Start generation
    _cancel_flag = False
    _generating = True
    dpg.set_item_label(TAG_INF_GENERATE_BTN, "Cancel")

    # Apply red theme while generating
    with dpg.theme() as cancel_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (210, 55, 65))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (180, 45, 55))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (155, 38, 48))
    dpg.bind_item_theme(TAG_INF_GENERATE_BTN, cancel_theme)

    # Gather parameters
    params = {
        "model_path": _model_path_map.get(dpg.get_value(TAG_INF_MODEL_PATH), ""),
        "prompt": dpg.get_value(TAG_INF_PROMPT),
        "negative_prompt": dpg.get_value(TAG_INF_NEG_PROMPT),
        "seed": dpg.get_value(TAG_INF_SEED),
        "steps": dpg.get_value(TAG_INF_STEPS),
        "cfg_scale": dpg.get_value(TAG_INF_CFG),
        "width": dpg.get_value(TAG_INF_WIDTH),
        "height": dpg.get_value(TAG_INF_HEIGHT),
        "sampler": dpg.get_value(TAG_INF_SAMPLER),
        "scheduler": dpg.get_value(TAG_INF_SCHEDULER),
        "output_dir": dpg.get_value(TAG_INF_OUTPUT_DIR),
        "vram_mode": dpg.get_value(TAG_INF_VRAM_MODE),
        "quantization": dpg.get_value(TAG_INF_QUANTIZATION),
        "attention": dpg.get_value(TAG_INF_ATTENTION),
        "rescale_cfg": dpg.get_value(TAG_INF_RESCALE_CFG),
        "mahiro": dpg.get_value(TAG_INF_MAHIRO),
        "batch_count": dpg.get_value(TAG_INF_BATCH_COUNT),
        "lora_path": _lora_path_map.get(dpg.get_value(TAG_INF_LORA_PATH), ""),
        "lora_weight": dpg.get_value(TAG_INF_LORA_WEIGHT),
    }

    if not params["model_path"]:
        _set_status("Error: No model selected", (210, 55, 65))
        _finish_generation()
        return

    global _gen_thread
    _gen_thread = threading.Thread(
        target=_run_generation, args=(params,), daemon=True,
    )
    _gen_thread.start()


def _run_generation(params: dict[str, Any]) -> None:
    """Run inference in a background thread."""
    global _engine, _generating, _cancel_flag

    t0 = time.monotonic()

    try:
        _set_status("Loading engine...", (86, 156, 240))

        from serenity.inference.config import (
            AttentionBackend,
            InferenceConfig,
            QuantizationMode,
            VRAMMode,
        )
        from serenity.inference.engine import InferenceEngine

        # Build config
        config = InferenceConfig(
            model_path=params["model_path"],
            model_dtype="bfloat16",
            vram_mode=VRAMMode(params["vram_mode"]),
            quantization=QuantizationMode(params["quantization"]),
            attention_backend=AttentionBackend(params["attention"]),
            sampler=params["sampler"],
            scheduler=params["scheduler"],
            steps=params["steps"],
            cfg_scale=params["cfg_scale"],
            width=params["width"],
            height=params["height"],
            rescale_cfg=params["rescale_cfg"],
            mahiro=params["mahiro"],
        )

        # LoRA
        lora_path = params.get("lora_path", "")
        if lora_path and os.path.isfile(lora_path):
            config.lora_paths = [lora_path]
            config.lora_weights = [params.get("lora_weight", 1.0)]

        # Create or reuse engine
        if _engine is None or _engine._config.model_path != params["model_path"]:
            if _engine is not None:
                _engine.unload_all()
            _engine = InferenceEngine(config)
            _set_status("Loading model...", (86, 156, 240))
            _engine.load_model()
        else:
            # Update config params on existing engine
            _engine._config.sampler = config.sampler
            _engine._config.scheduler = config.scheduler
            _engine._config.steps = config.steps
            _engine._config.cfg_scale = config.cfg_scale
            _engine._config.width = config.width
            _engine._config.height = config.height
            _engine._config.lora_paths = config.lora_paths
            _engine._config.lora_weights = config.lora_weights

        if _cancel_flag:
            _set_status("Cancelled", (210, 55, 65))
            _finish_generation()
            return

        _set_status("Generating...", (86, 156, 240))

        # Step callback for progress bar
        total_steps = params["steps"]

        def _on_step(step: int, total: int, sigma: Any, denoised: Any) -> None:
            if _cancel_flag:
                raise KeyboardInterrupt("Cancelled by user")
            frac = (step + 1) / total if total > 0 else 0
            try:
                dpg.set_value(TAG_INF_PROGRESS, frac)
                dpg.configure_item(
                    TAG_INF_PROGRESS,
                    overlay=f"{step + 1} / {total}",
                )
            except SystemError:
                pass

        # Run generation
        batch_count = max(1, params.get("batch_count", 1))
        all_images: list[str] = []
        all_seeds: list[int] = []

        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx in range(batch_count):
            if _cancel_flag:
                break

            result = _engine.generate(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                seed=params["seed"],
                steps=params["steps"],
                cfg_scale=params["cfg_scale"],
                width=params["width"],
                height=params["height"],
                sampler=params["sampler"],
                scheduler=params["scheduler"],
                batch_size=1,
                callback=_on_step,
            )

            # Save images
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            for i, img_tensor in enumerate(result.images):
                seed_val = result.seeds[i] if i < len(result.seeds) else 0
                filename = f"{ts}_{seed_val}_{batch_idx}_{i}.png"
                filepath = output_dir / filename

                _save_image(img_tensor, filepath)
                all_images.append(str(filepath))
                all_seeds.append(seed_val)

                # Save JSON sidecar
                meta = {
                    "prompt": params["prompt"],
                    "negative_prompt": params["negative_prompt"],
                    "seed": seed_val,
                    "steps": params["steps"],
                    "cfg_scale": params["cfg_scale"],
                    "width": params["width"],
                    "height": params["height"],
                    "sampler": params["sampler"],
                    "scheduler": params["scheduler"],
                    "model": params["model_path"],
                    "lora": params.get("lora_path", ""),
                    "lora_weight": params.get("lora_weight", 1.0),
                }
                json_path = filepath.with_suffix(".json")
                json_path.write_text(json.dumps(meta, indent=2))

        elapsed = round(time.monotonic() - t0, 2)

        if _cancel_flag:
            _set_status("Cancelled", (210, 55, 65))
        else:
            _set_status(f"Done ({elapsed}s)", (100, 200, 130))

        # Update info panel
        info_lines = [
            f"Images: {', '.join(os.path.basename(p) for p in all_images)}",
            f"Seeds: {all_seeds}",
            f"Time: {elapsed}s",
            f"Steps: {params['steps']}, CFG: {params['cfg_scale']}, "
            f"Sampler: {params['sampler']}, Scheduler: {params['scheduler']}",
            f"Resolution: {params['width']}x{params['height']}",
            f"Output: {params['output_dir']}",
        ]
        try:
            dpg.set_value(TAG_INF_INFO_PANEL, "\n".join(info_lines))
        except SystemError:
            pass

        # Add to history
        _add_history_entry({
            "time": datetime.now().strftime("%H:%M:%S"),
            "prompt": params["prompt"][:60],
            "seed": all_seeds[0] if all_seeds else -1,
            "steps": params["steps"],
            "elapsed": elapsed,
            "images": all_images,
        })

    except KeyboardInterrupt:
        _set_status("Cancelled", (210, 55, 65))
    except Exception as exc:
        logger.exception("Generation failed")
        _set_status(f"Error: {exc}", (210, 55, 65))
    finally:
        _finish_generation()
        _update_vram_label()


def _finish_generation() -> None:
    """Reset UI state after generation completes."""
    global _generating
    _generating = False

    try:
        dpg.set_item_label(TAG_INF_GENERATE_BTN, "Generate")
        with dpg.theme() as gen_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 160, 95))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (25, 135, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 110, 65))
        dpg.bind_item_theme(TAG_INF_GENERATE_BTN, gen_theme)
    except SystemError:
        pass


# -- Image saving --------------------------------------------------------------


def _save_image(tensor: Any, path: Path) -> None:
    """Save a [C, H, W] tensor in [0, 1] range as PNG."""
    import torch

    t = tensor.detach().cpu().float().clamp(0, 1)
    if t.ndim == 3:
        t = t.permute(1, 2, 0)  # CHW -> HWC
    img_np = (t * 255).to(torch.uint8).numpy()

    try:
        from PIL import Image
        Image.fromarray(img_np).save(str(path))
    except ImportError:
        # Fallback: PPM
        h, w = img_np.shape[:2]
        header = f"P6\n{w} {h}\n255\n".encode()
        path.with_suffix(".ppm").write_bytes(header + img_np.tobytes())
        logger.warning("PIL not available, saved PPM to %s", path.with_suffix(".ppm"))


# -- History -------------------------------------------------------------------


def _add_history_entry(entry: dict[str, Any]) -> None:
    """Add an entry to the history list and rebuild the display."""
    _history.insert(0, entry)
    # Keep last 50
    while len(_history) > 50:
        _history.pop()
    _rebuild_history()


def _rebuild_history() -> None:
    """Redraw the history list."""
    try:
        dpg.delete_item(TAG_INF_HISTORY_LIST, children_only=True)
    except SystemError:
        return

    for entry in _history:
        prompt_preview = entry.get("prompt", "")[:50]
        seed = entry.get("seed", "?")
        elapsed = entry.get("elapsed", 0)
        t = entry.get("time", "")
        label = f"[{t}] seed={seed} {elapsed}s - {prompt_preview}"

        with dpg.group(parent=TAG_INF_HISTORY_LIST, horizontal=True):
            dpg.add_text(label, wrap=scaled(500))

        images = entry.get("images", [])
        if images:
            dpg.add_text(
                f"  -> {', '.join(os.path.basename(p) for p in images)}",
                parent=TAG_INF_HISTORY_LIST,
                color=(130, 130, 145),
            )
        dpg.add_separator(parent=TAG_INF_HISTORY_LIST)
