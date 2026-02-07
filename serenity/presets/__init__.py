"""Training preset loading and management.

Provides structured preset definitions organized by model type and VRAM
tier, with a loader that merges preset defaults with user overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Location of built-in preset files
PRESETS_DIR = Path(__file__).parent


def list_presets() -> list[str]:
    """Return names of all available built-in presets (without extension)."""
    results: list[str] = []
    for ext in (".json", ".yaml", ".yml"):
        for p in PRESETS_DIR.glob(f"*{ext}"):
            if p.name.startswith("_") or p.name == "__init__.py":
                continue
            results.append(p.stem)
    return sorted(set(results))


def load_preset(name: str) -> dict[str, Any]:
    """Load a built-in preset by name.

    Searches for ``{name}.json``, ``{name}.yaml``, or ``{name}.yml``
    in the presets directory.

    Returns:
        Dict of preset configuration values.

    Raises:
        FileNotFoundError: If no preset with the given name exists.
    """
    for ext in (".json", ".yaml", ".yml"):
        path = PRESETS_DIR / f"{name}{ext}"
        if path.exists():
            return _load_file(path)

    raise FileNotFoundError(
        f"Preset '{name}' not found. Available: {list_presets()}"
    )


def load_preset_file(path: str | Path) -> dict[str, Any]:
    """Load a preset from an arbitrary file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    return _load_file(path)


def merge_preset(preset: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge user overrides on top of preset defaults.

    User values take precedence.  Nested dicts are merged recursively.
    """
    result = {}
    for key in set(preset) | set(overrides):
        preset_val = preset.get(key)
        override_val = overrides.get(key)

        if key in overrides and key in preset:
            if isinstance(preset_val, dict) and isinstance(override_val, dict):
                result[key] = merge_preset(preset_val, override_val)
            else:
                result[key] = override_val
        elif key in overrides:
            result[key] = override_val
        else:
            result[key] = preset_val

    return result


def apply_preset(
    name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a preset and apply user overrides.

    Convenience wrapper combining ``load_preset`` and ``merge_preset``.
    """
    preset = load_preset(name)
    if overrides:
        return merge_preset(preset, overrides)
    return preset


def _load_file(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML file into a dict."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML presets. "
                "Install with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported preset format: {suffix}")


__all__ = [
    "PRESETS_DIR",
    "apply_preset",
    "list_presets",
    "load_preset",
    "load_preset_file",
    "merge_preset",
]
