"""UI state management: bidirectional binding between TrainConfig and DPG widgets."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

from serenity.core.config import (
    TrainConfig,
    load_config,
)

__all__ = ["UIState"]


class UIState:
    """Manages bidirectional binding between a TrainConfig and DPG widgets.

    Widget tags follow a dotted-path convention:
        "learning_rate"  -> config.learning_rate
        "optimizer.optimizer" -> config.optimizer.optimizer
        "text_encoder.train" -> config.text_encoder.train

    Call ``register(tag, field_path)`` to bind a widget.
    Call ``sync_to_config()`` to push all widget values into the config.
    Call ``sync_from_config()`` to push all config values into widgets.
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or _default_config()
        # Map of widget_tag -> (field_path, type_hint)
        self._bindings: dict[str, tuple[str, type]] = {}
        # Reverse map for enum display-name -> value
        self._enum_maps: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tag: str,
        field_path: str | None = None,
        type_hint: type | None = None,
    ) -> None:
        """Register a widget tag <-> config field binding.

        If *field_path* is None, uses *tag* as the path.
        """
        path = field_path or tag
        hint = type_hint or _resolve_type(self.config, path)
        self._bindings[tag] = (path, hint)

    def register_enum_map(self, tag: str, mapping: dict[str, Any]) -> None:
        """Register a display-name -> enum value mapping for a combo widget."""
        self._enum_maps[tag] = mapping

    # ------------------------------------------------------------------
    # Sync: widgets -> config
    # ------------------------------------------------------------------

    def sync_to_config(self) -> None:
        """Push all widget values into the config object."""
        for tag, (path, hint) in self._bindings.items():
            try:
                raw = dpg.get_value(tag)
            except SystemError:
                continue
            value = self._convert_value(tag, raw, hint)
            _set_nested(self.config, path, value)

    def widget_to_config(self, tag: str) -> None:
        """Push a single widget value into config."""
        if tag not in self._bindings:
            return
        path, hint = self._bindings[tag]
        try:
            raw = dpg.get_value(tag)
        except SystemError:
            return
        value = self._convert_value(tag, raw, hint)
        _set_nested(self.config, path, value)

    # ------------------------------------------------------------------
    # Sync: config -> widgets
    # ------------------------------------------------------------------

    def sync_from_config(self) -> None:
        """Push all config values into widgets."""
        for tag, (path, hint) in self._bindings.items():
            value = _get_nested(self.config, path)
            display = self._display_value(tag, value, hint)
            try:
                dpg.set_value(tag, display)
            except SystemError:
                pass

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def load_config_file(self, path: str | Path) -> None:
        """Load a config file and update all widgets."""
        self.config = load_config(path)
        self.sync_from_config()

    def save_config_file(self, path: str | Path) -> None:
        """Save current config to a JSON file."""
        self.sync_to_config()
        path = Path(path)
        data = dataclasses.asdict(self.config)
        # Convert enums to their string values
        _stringify_enums(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def new_config(self) -> None:
        """Reset to a fresh default config."""
        self.config = _default_config()
        self.sync_from_config()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def make_callback(self, tag: str):
        """Create a DPG callback that syncs a single widget to config."""
        def _cb(sender, app_data, user_data):
            self.widget_to_config(tag)
        return _cb

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _convert_value(self, tag: str, raw: Any, hint: type) -> Any:
        """Convert raw widget value to config-compatible type."""
        if tag in self._enum_maps:
            mapping = self._enum_maps[tag]
            return mapping.get(str(raw), raw)
        if hint is bool:
            return bool(raw)
        if hint is int:
            try:
                return int(raw) if raw != "" else 0
            except (ValueError, TypeError):
                return 0
        if hint is float:
            try:
                return float(raw) if raw != "" else 0.0
            except (ValueError, TypeError):
                return 0.0
        if isinstance(hint, type) and issubclass(hint, Enum):
            # Try to coerce string to enum
            try:
                return hint(raw)
            except (ValueError, KeyError):
                for member in hint:
                    if str(member) == raw or member.value == raw or member.name == raw:
                        return member
                return raw
        return raw

    def _display_value(self, tag: str, value: Any, hint: type) -> Any:
        """Convert config value to widget-displayable form."""
        if tag in self._enum_maps:
            # Reverse lookup: value -> display name
            rev = {v: k for k, v in self._enum_maps[tag].items()}
            if value in rev:
                return rev[value]
            return str(value) if value is not None else ""
        if isinstance(value, Enum):
            return value.value
        if value is None:
            if hint in (int, float):
                return 0
            return ""
        return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> TrainConfig:
    """Create a default TrainConfig for new sessions."""
    return TrainConfig(
        model_type="flux_dev",
        training_method="lora",
        transformer_path="",
        output_dir="output",
        concepts=[],
    )


def _get_nested(obj: Any, path: str) -> Any:
    """Get a dotted-path attribute, e.g., 'optimizer.optimizer'."""
    parts = path.split(".")
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


def _set_nested(obj: Any, path: str, value: Any) -> None:
    """Set a dotted-path attribute."""
    parts = path.split(".")
    for part in parts[:-1]:
        if isinstance(obj, dict):
            obj = obj.get(part, {})
        else:
            obj = getattr(obj, part, None)
    if obj is not None:
        leaf = parts[-1]
        if isinstance(obj, dict):
            obj[leaf] = value
        else:
            setattr(obj, leaf, value)


def _resolve_type(config: TrainConfig, path: str) -> type:
    """Try to determine the type of a config field from annotations."""
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
    if obj is None:
        return str
    leaf = parts[-1]
    # Try dataclass fields
    if dataclasses.is_dataclass(obj):
        for f in fields(type(obj)):
            if f.name == leaf:
                t = f.type
                if isinstance(t, str):
                    return str  # Can't resolve forward refs here
                origin = getattr(t, "__origin__", None)
                if origin is not None:
                    # Handle Optional[X] etc
                    args = getattr(t, "__args__", ())
                    non_none = [a for a in args if a is not type(None)]
                    if non_none:
                        return non_none[0]
                return t
    return str


def _stringify_enums(d: dict) -> None:
    """Recursively convert enum values to strings in a dict."""
    for k, v in d.items():
        if isinstance(v, Enum):
            d[k] = v.value
        elif isinstance(v, dict):
            _stringify_enums(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, Enum):
                    d[k][i] = item.value
                elif isinstance(item, dict):
                    _stringify_enums(item)
