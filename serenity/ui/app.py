"""Main Serenity UI application window."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

from serenity.core.config import TrainingMethod
from serenity.core.interfaces import ModelType
from serenity.ui.state import UIState
from serenity.ui.theme import apply_dark_theme, create_start_button_theme, create_stop_button_theme, scaled, setup_fonts

__all__ = ["SerenityApp"]

# Tags for key widgets
TAG_PRIMARY_WINDOW = "primary_window"
TAG_TAB_BAR = "tab_bar"
TAG_STATUS_LABEL = "status_label"
TAG_ETA_LABEL = "eta_label"
TAG_STEP_PROGRESS = "step_progress"
TAG_EPOCH_PROGRESS = "epoch_progress"
TAG_TRAIN_BUTTON = "train_button"
TAG_MODEL_TYPE = "top_model_type"
TAG_TRAINING_METHOD = "top_training_method"

# Conditional tabs
TAG_LORA_TAB = "lora_tab"
TAG_EMBEDDING_TAB = "embedding_tab"


class SerenityApp:
    """Manages the entire Serenity training UI."""

    def __init__(self) -> None:
        self.ui_state = UIState()
        self.training_thread: threading.Thread | None = None
        self._is_training = False

        # Lazy-imported tab builders
        self._tab_builders: dict[str, Any] = {}

    def run(self) -> None:
        """Launch the UI."""
        dpg.create_context()
        dpg.create_viewport(
            title="Serenity - Training UI",
            width=scaled(1140),
            height=scaled(720),
            min_width=scaled(720),
            min_height=scaled(500),
        )

        setup_fonts()
        apply_dark_theme()
        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(TAG_PRIMARY_WINDOW, True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        with dpg.window(tag=TAG_PRIMARY_WINDOW):
            self._build_top_bar()
            dpg.add_spacer(height=6)
            self._build_tab_area()
            self._build_bottom_bar()

    def _build_top_bar(self) -> None:
        """Header with branding, model/method selectors, and train button."""
        # Row 1: Brand + primary controls + train button
        with dpg.group(horizontal=True):
            # Brand
            dpg.add_text("Serenity", color=(86, 156, 240))
            dpg.add_spacer(width=scaled(20))

            # Model type dropdown
            model_types = [
                ("SD 1.5", "sd15"), ("SDXL", "sdxl_10_base"),
                ("SD3", "sd3"), ("SD3.5", "sd35"),
                ("Flux.1 Dev", "flux_dev"), ("Flux Fill Dev", "flux_fill_dev"),
                ("Flux 2", "flux_2"), ("Flux 2 Klein 4B", "flux_2_klein_4b"),
                ("Chroma", "chroma_1"), ("Qwen", "qwen"),
                ("Z-Image", "zimage"), ("LTX2", "ltx2"),
                ("Hunyuan Video", "hunyuan_video"),
                ("Sana", "sana"), ("HiDream", "hi_dream_full"),
            ]
            dpg.add_text("Model:", color=(160, 160, 175))
            model_names = [n for n, _ in model_types]
            self._model_type_map = {n: v for n, v in model_types}
            self._model_type_rev = {v: n for n, v in model_types}
            current_mt = self._model_type_rev.get(
                self.ui_state.config.model_type.value, "Flux.1 Dev"
            )
            dpg.add_combo(
                tag=TAG_MODEL_TYPE,
                items=model_names,
                default_value=current_mt,
                width=scaled(130),
                callback=self._on_model_type_changed,
            )

            dpg.add_spacer(width=scaled(10))

            # Training method dropdown
            dpg.add_text("Method:", color=(160, 160, 175))
            method_names = ["Fine Tune", "LoRA", "Embedding"]
            self._method_map = {
                "Fine Tune": TrainingMethod.FINE_TUNE,
                "LoRA": TrainingMethod.LORA,
                "Embedding": TrainingMethod.EMBEDDING,
                "Fine Tune VAE": TrainingMethod.FINE_TUNE_VAE,
            }
            self._method_rev = {v: k for k, v in self._method_map.items()}
            current_tm = self._method_rev.get(
                self.ui_state.config.training_method, "LoRA"
            )
            dpg.add_combo(
                tag=TAG_TRAINING_METHOD,
                items=method_names,
                default_value=current_tm,
                width=scaled(100),
                callback=self._on_training_method_changed,
            )

            dpg.add_spacer(width=scaled(20))

            # Config preset
            dpg.add_text("Config:", color=(160, 160, 175))
            dpg.add_combo(
                tag="config_preset",
                items=self._list_presets(),
                default_value="(new)",
                width=scaled(145),
                callback=self._on_preset_selected,
            )
            dpg.add_button(label="Save", callback=self._save_config, width=scaled(50))
            dpg.add_button(label="Save As", callback=self._save_config_as, width=scaled(58))
            dpg.add_button(label="Open", callback=self._open_config, width=scaled(50))

            dpg.add_spacer(width=scaled(28))

            # Train button -- prominent, right side
            dpg.add_button(
                tag=TAG_TRAIN_BUTTON,
                label="Start Training",
                callback=self._toggle_training,
                width=scaled(130),
                height=scaled(26),
            )
            start_theme = create_start_button_theme()
            dpg.bind_item_theme(TAG_TRAIN_BUTTON, start_theme)

        dpg.add_separator()

    def _build_tab_area(self) -> None:
        """Main tabbed content area -- workflow order."""
        from serenity.ui.tabs.model import build_model_tab
        from serenity.ui.tabs.data import build_data_tab
        from serenity.ui.tabs.concepts import build_concepts_tab
        from serenity.ui.tabs.training import build_training_tab
        from serenity.ui.tabs.sampling import build_sampling_tab
        from serenity.ui.tabs.backup import build_backup_tab
        from serenity.ui.tabs.general import build_general_tab
        from serenity.ui.tabs.lora import build_lora_tab
        from serenity.ui.tabs.embedding import build_embedding_tab
        from serenity.ui.tabs.inference import build_inference_tab

        with dpg.tab_bar(tag=TAG_TAB_BAR):
            with dpg.tab(label="  Model  "):
                build_model_tab(self.ui_state)
            with dpg.tab(label="  Data  "):
                build_data_tab(self.ui_state)
            with dpg.tab(label="  Concepts  "):
                build_concepts_tab(self.ui_state)
            with dpg.tab(label="  Training  "):
                build_training_tab(self.ui_state)

            # Conditional tabs based on training method
            if self.ui_state.config.training_method == TrainingMethod.LORA:
                with dpg.tab(label="  LoRA  ", tag=TAG_LORA_TAB):
                    build_lora_tab(self.ui_state)
            if self.ui_state.config.training_method == TrainingMethod.EMBEDDING:
                with dpg.tab(label="  Embedding  ", tag=TAG_EMBEDDING_TAB):
                    build_embedding_tab(self.ui_state)

            with dpg.tab(label="  Sampling  "):
                build_sampling_tab(self.ui_state)
            with dpg.tab(label="  Inference  "):
                build_inference_tab()
            with dpg.tab(label="  Backup  "):
                build_backup_tab(self.ui_state)
            with dpg.tab(label="  Settings  "):
                build_general_tab(self.ui_state)

    def _build_bottom_bar(self) -> None:
        """Compact status bar with progress and status."""
        dpg.add_spacer(height=4)
        dpg.add_separator()
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            # Progress
            dpg.add_text("Step", color=(130, 130, 145))
            dpg.add_progress_bar(
                tag=TAG_STEP_PROGRESS,
                default_value=0.0,
                width=scaled(215),
                overlay="0 / 0",
            )
            dpg.add_spacer(width=scaled(12))
            dpg.add_text("Epoch", color=(130, 130, 145))
            dpg.add_progress_bar(
                tag=TAG_EPOCH_PROGRESS,
                default_value=0.0,
                width=scaled(215),
                overlay="0 / 0",
            )
            dpg.add_spacer(width=scaled(16))

            # Status text
            dpg.add_text("Ready", tag=TAG_STATUS_LABEL, color=(100, 200, 130))
            dpg.add_text("", tag=TAG_ETA_LABEL, color=(130, 130, 145))

            dpg.add_spacer(width=scaled(16))

            # Utility buttons
            dpg.add_button(
                label="Tensorboard",
                callback=self._open_tensorboard,
                width=scaled(86),
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_model_type_changed(self, sender: Any, app_data: str, user_data: Any = None) -> None:
        mt_value = self._model_type_map.get(app_data)
        if mt_value:
            self.ui_state.config.model_type = ModelType(mt_value)

    def _on_training_method_changed(self, sender: Any, app_data: str, user_data: Any = None) -> None:
        method = self._method_map.get(app_data)
        if method:
            self.ui_state.config.training_method = method

    def _on_preset_selected(self, sender: Any, app_data: str, user_data: Any = None) -> None:
        if app_data == "(new)":
            self.ui_state.new_config()
            return
        preset_path = Path("training_presets") / f"{app_data}.json"
        if preset_path.exists():
            self.ui_state.load_config_file(preset_path)
            mt = self.ui_state.config.model_type.value
            if mt in self._model_type_rev:
                dpg.set_value(TAG_MODEL_TYPE, self._model_type_rev[mt])
            tm = self.ui_state.config.training_method
            if tm in self._method_rev:
                dpg.set_value(TAG_TRAINING_METHOD, self._method_rev[tm])

    def _save_config(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        self.ui_state.sync_to_config()
        preset_name = dpg.get_value("config_preset")
        if preset_name == "(new)":
            self._save_config_as()
            return
        path = Path("training_presets") / f"{preset_name}.json"
        self.ui_state.save_config_file(path)
        self._set_status(f"Saved: {path}")

    def _save_config_as(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        def _do_save(_s: Any, data: dict) -> None:
            path = data.get("file_path_name", "")
            if path:
                self.ui_state.save_config_file(path)
                self._set_status(f"Saved: {path}")

        with dpg.file_dialog(callback=_do_save, width=700, height=400):
            dpg.add_file_extension(".json")
            dpg.add_file_extension(".yaml")

    def _open_config(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        def _do_open(_s: Any, data: dict) -> None:
            path = data.get("file_path_name", "")
            if path:
                try:
                    self.ui_state.load_config_file(path)
                    self._set_status(f"Loaded: {path}")
                    mt = self.ui_state.config.model_type.value
                    if mt in self._model_type_rev:
                        dpg.set_value(TAG_MODEL_TYPE, self._model_type_rev[mt])
                    tm = self.ui_state.config.training_method
                    if tm in self._method_rev:
                        dpg.set_value(TAG_TRAINING_METHOD, self._method_rev[tm])
                except Exception as exc:
                    self._set_status(f"Error: {exc}")

        with dpg.file_dialog(callback=_do_open, width=700, height=400):
            dpg.add_file_extension(".json")
            dpg.add_file_extension(".yaml")
            dpg.add_file_extension(".yml")

    def _export_config(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        self._save_config_as()

    def _open_tensorboard(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        import webbrowser
        port = self.ui_state.config.tensorboard_port
        webbrowser.open(f"http://localhost:{port}", new=0, autoraise=False)

    def _toggle_training(self, sender: Any = None, app_data: Any = None, user_data: Any = None) -> None:
        if self._is_training:
            self._stop_training()
        else:
            self._start_training()

    def _start_training(self) -> None:
        self.ui_state.sync_to_config()
        self._is_training = True
        dpg.set_item_label(TAG_TRAIN_BUTTON, "Stop Training")
        stop_theme = create_stop_button_theme()
        dpg.bind_item_theme(TAG_TRAIN_BUTTON, stop_theme)
        self._set_status("Starting training...")

        def _train_thread() -> None:
            try:
                from serenity.core.trainer import Trainer
                trainer = Trainer(self.ui_state.config)
                trainer.train(
                    progress_callback=self._on_training_progress,
                    status_callback=self._on_training_status,
                )
            except Exception as exc:
                self._set_status(f"Error: {exc}")
            finally:
                self._is_training = False
                dpg.set_item_label(TAG_TRAIN_BUTTON, "Start Training")
                start_theme = create_start_button_theme()
                dpg.bind_item_theme(TAG_TRAIN_BUTTON, start_theme)

        self.training_thread = threading.Thread(target=_train_thread, daemon=True)
        self.training_thread.start()

    def _stop_training(self) -> None:
        self._set_status("Stopping...")
        dpg.set_item_label(TAG_TRAIN_BUTTON, "Stopping...")
        dpg.configure_item(TAG_TRAIN_BUTTON, enabled=False)
        self._is_training = False

    def _on_training_progress(self, step: int, total_steps: int, epoch: int, total_epochs: int) -> None:
        if total_steps > 0:
            dpg.set_value(TAG_STEP_PROGRESS, step / total_steps)
            dpg.configure_item(TAG_STEP_PROGRESS, overlay=f"{step} / {total_steps}")
        if total_epochs > 0:
            dpg.set_value(TAG_EPOCH_PROGRESS, epoch / total_epochs)
            dpg.configure_item(TAG_EPOCH_PROGRESS, overlay=f"{epoch} / {total_epochs}")

    def _on_training_status(self, status: str) -> None:
        self._set_status(status)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        try:
            dpg.set_value(TAG_STATUS_LABEL, text)
        except SystemError:
            pass

    def _list_presets(self) -> list[str]:
        presets = ["(new)"]
        preset_dir = Path("training_presets")
        if preset_dir.exists():
            for p in sorted(preset_dir.glob("*.json")):
                if p.stem != "#":
                    presets.append(p.stem)
        serenity_dir = Path(__file__).parent.parent / "presets"
        if serenity_dir.exists():
            for p in sorted(serenity_dir.glob("*.json")):
                name = f"[built-in] {p.stem}"
                presets.append(name)
        return presets
