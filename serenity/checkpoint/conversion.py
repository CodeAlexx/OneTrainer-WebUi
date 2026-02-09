"""Model format detection and conversion utilities.

Converts between diffusers, safetensors, and legacy checkpoint formats.
Also handles LoRA key mapping between formats (OMI, diffusers, legacy).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format Detection
# ---------------------------------------------------------------------------


class ModelFormat(str, Enum):
    """Recognized model/checkpoint formats."""

    DIFFUSERS = "diffusers"
    SAFETENSORS = "safetensors"
    TORCH_CKPT = "torch_ckpt"
    LORA_SAFETENSORS = "lora_safetensors"
    LORA_TORCH = "lora_torch"
    UNKNOWN = "unknown"


def detect_model_format(path: str | Path) -> ModelFormat:
    """Automatically detect the format of a model at *path*.

    Heuristics
    ----------
    * ``.safetensors`` file -> SAFETENSORS (or LORA_SAFETENSORS if small/LoRA keys)
    * ``.pt``/``.bin``/``.ckpt`` file -> TORCH_CKPT (or LORA_TORCH)
    * Directory with ``model_index.json`` -> DIFFUSERS
    * Directory with diffusers sub-folders -> DIFFUSERS
    """
    p = Path(path).expanduser()

    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".safetensors":
            if _is_lora_safetensors(p):
                return ModelFormat.LORA_SAFETENSORS
            return ModelFormat.SAFETENSORS
        if suffix in {".pt", ".pth", ".bin", ".ckpt"}:
            if _is_lora_torch(p):
                return ModelFormat.LORA_TORCH
            return ModelFormat.TORCH_CKPT
        return ModelFormat.UNKNOWN

    if p.is_dir():
        if (p / "model_index.json").exists():
            return ModelFormat.DIFFUSERS
        # Check for diffusers-style sub-folders
        diffusers_markers = {"transformer", "unet", "vae", "text_encoder", "scheduler"}
        found = {d.name for d in p.iterdir() if d.is_dir()} & diffusers_markers
        if len(found) >= 2:
            return ModelFormat.DIFFUSERS
        if any(p.glob("*.safetensors")):
            return ModelFormat.SAFETENSORS
        return ModelFormat.UNKNOWN

    return ModelFormat.UNKNOWN


def _is_lora_safetensors(path: Path) -> bool:
    """Check if a safetensors file contains LoRA weights."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            keys = f.keys()
            # LoRA files have keys like "lora_unet_..." or contain "lora" in keys
            lora_keys = [k for k in keys if "lora" in k.lower() or "alpha" in k.lower()]
            return len(lora_keys) > len(keys) * 0.3
    except (OSError, RuntimeError, KeyError):
        # Fall back to size heuristic: LoRA files are typically < 500MB
        return path.stat().st_size < 500 * 1024 * 1024


def _is_lora_torch(path: Path) -> bool:
    """Check if a torch checkpoint contains LoRA weights."""
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            keys = list(state.keys())
            lora_keys = [k for k in keys if "lora" in k.lower()]
            return len(lora_keys) > len(keys) * 0.3
    except (OSError, RuntimeError, KeyError):
        logger.debug("Could not read checkpoint file %s for LoRA detection, using size heuristic", path)
    return path.stat().st_size < 500 * 1024 * 1024


# ---------------------------------------------------------------------------
# Diffusers <-> Safetensors Conversion
# ---------------------------------------------------------------------------


def convert_diffusers_to_safetensors(
    diffusers_path: str | Path,
    output_path: str | Path,
    component: str = "transformer",
    dtype: torch.dtype | None = None,
) -> Path:
    """Extract a component's state dict from a diffusers directory to safetensors.

    Parameters
    ----------
    diffusers_path : str | Path
        Path to diffusers model directory.
    output_path : str | Path
        Output ``.safetensors`` file path.
    component : str
        Component to extract (e.g. ``'transformer'``, ``'unet'``, ``'vae'``).
    dtype : torch.dtype | None
        Optional dtype cast for the output.

    Returns
    -------
    Path
        The output file path.
    """
    from safetensors.torch import save_file

    diffusers_path = Path(diffusers_path).expanduser()
    output_path = Path(output_path).expanduser()

    component_dir = diffusers_path / component
    if not component_dir.exists():
        raise FileNotFoundError(f"Component '{component}' not found at {diffusers_path}")

    state_dict = _load_state_dict_from_directory(component_dir)

    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path))
    logger.info("Converted %s/%s -> %s (%d tensors)", diffusers_path, component, output_path, len(state_dict))
    return output_path


def convert_safetensors_to_diffusers(
    safetensors_path: str | Path,
    diffusers_path: str | Path,
    component: str = "transformer",
    config_path: str | Path | None = None,
) -> Path:
    """Insert a safetensors state dict into a diffusers directory layout.

    Parameters
    ----------
    safetensors_path : str | Path
        Input ``.safetensors`` file.
    diffusers_path : str | Path
        Target diffusers directory.
    component : str
        Component name to save under.
    config_path : str | Path | None
        Optional config.json to copy alongside the weights.

    Returns
    -------
    Path
        The diffusers component directory.
    """
    from safetensors.torch import load_file, save_file

    safetensors_path = Path(safetensors_path).expanduser()
    diffusers_path = Path(diffusers_path).expanduser()

    state_dict = load_file(str(safetensors_path))

    component_dir = diffusers_path / component
    component_dir.mkdir(parents=True, exist_ok=True)

    save_file(state_dict, str(component_dir / "model.safetensors"))

    if config_path is not None:
        config_path = Path(config_path).expanduser()
        if config_path.exists():
            (component_dir / "config.json").write_text(config_path.read_text())

    logger.info("Converted %s -> %s/%s (%d tensors)", safetensors_path, diffusers_path, component, len(state_dict))
    return component_dir


def _load_state_dict_from_directory(directory: Path) -> dict[str, Tensor]:
    """Load a state dict from a directory of safetensors/bin files."""
    state_dict: dict[str, Tensor] = {}

    # Try safetensors first
    sf_files = sorted(directory.glob("*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file
        for sf_file in sf_files:
            state_dict.update(load_file(str(sf_file)))
        return state_dict

    # Try torch files
    bin_files = sorted(directory.glob("*.bin")) + sorted(directory.glob("*.pt"))
    for bin_file in bin_files:
        data = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            state_dict.update(data)

    return state_dict


# ---------------------------------------------------------------------------
# LoRA Format Conversion
# ---------------------------------------------------------------------------


@dataclass
class LoRAKeyMapping:
    """Mapping between LoRA key formats."""

    omi_prefix: str = ""
    diffusers_prefix: str = ""
    legacy_prefix: str = ""


# Common LoRA key prefix mappings
_SD15_LORA_KEY_MAP: list[tuple[str, str]] = [
    ("lora_unet_", "unet."),
    ("lora_te_", "text_encoder."),
]

_SDXL_LORA_KEY_MAP: list[tuple[str, str]] = [
    ("lora_unet_", "unet."),
    ("lora_te1_", "text_encoder."),
    ("lora_te2_", "text_encoder_2."),
]

_FLUX_LORA_KEY_MAP: list[tuple[str, str]] = [
    ("lora_transformer_", "transformer."),
    ("lora_te1_", "text_encoder."),
    ("lora_te2_", "text_encoder_2."),
]


def convert_lora_format(
    state_dict: dict[str, Tensor],
    source_format: str = "auto",
    target_format: str = "diffusers",
    model_family: str = "sd15",
) -> dict[str, Tensor]:
    """Convert LoRA state dict keys between formats.

    Parameters
    ----------
    state_dict : dict[str, Tensor]
        Input LoRA state dict.
    source_format : str
        Source key format: ``'legacy'``, ``'diffusers'``, ``'auto'``.
    target_format : str
        Target key format: ``'legacy'``, ``'diffusers'``.
    model_family : str
        Model family for key mapping: ``'sd15'``, ``'sdxl'``, ``'flux'``.

    Returns
    -------
    dict[str, Tensor]
        Converted state dict.
    """
    if source_format == "auto":
        source_format = _detect_lora_format(state_dict)

    if source_format == target_format:
        return state_dict

    key_map = _get_lora_key_map(model_family)

    result: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = _convert_lora_key(key, source_format, target_format, key_map)
        result[new_key] = value

    logger.info(
        "Converted LoRA keys: %s -> %s (%s, %d keys)",
        source_format, target_format, model_family, len(result),
    )
    return result


def _detect_lora_format(state_dict: dict[str, Tensor]) -> str:
    """Detect whether a LoRA state dict uses legacy or diffusers key format."""
    keys = list(state_dict.keys())
    if not keys:
        return "diffusers"

    # Legacy format uses underscores: lora_unet_down_blocks_0_...
    # Diffusers format uses dots: unet.down_blocks.0....
    legacy_count = sum(1 for k in keys if k.startswith("lora_"))
    dots_count = sum(1 for k in keys if "." in k.split("lora_", 1)[-1] if "lora_" in k)

    if legacy_count > len(keys) * 0.5:
        return "legacy"
    return "diffusers"


def _get_lora_key_map(model_family: str) -> list[tuple[str, str]]:
    """Get the legacy<->diffusers prefix mapping for a model family."""
    _maps = {
        "sd15": _SD15_LORA_KEY_MAP,
        "sd20": _SD15_LORA_KEY_MAP,
        "sd21": _SD15_LORA_KEY_MAP,
        "sdxl": _SDXL_LORA_KEY_MAP,
        "flux": _FLUX_LORA_KEY_MAP,
        "flux2": _FLUX_LORA_KEY_MAP,
        "chroma": _FLUX_LORA_KEY_MAP,
    }
    return _maps.get(model_family, _SD15_LORA_KEY_MAP)


def _convert_lora_key(
    key: str,
    source: str,
    target: str,
    key_map: list[tuple[str, str]],
) -> str:
    """Convert a single LoRA key between formats."""
    if source == "legacy" and target == "diffusers":
        for legacy_prefix, diffusers_prefix in key_map:
            if key.startswith(legacy_prefix):
                suffix = key[len(legacy_prefix):]
                # Convert underscores to dots, but keep suffixes like .alpha, .weight
                parts = suffix.rsplit(".", 1)
                if len(parts) == 2:
                    module_path, param = parts
                    module_path = _underscore_to_dots(module_path)
                    return f"{diffusers_prefix}{module_path}.{param}"
                return f"{diffusers_prefix}{_underscore_to_dots(suffix)}"
        return key

    if source == "diffusers" and target == "legacy":
        for legacy_prefix, diffusers_prefix in key_map:
            if key.startswith(diffusers_prefix):
                suffix = key[len(diffusers_prefix):]
                return f"{legacy_prefix}{suffix.replace('.', '_')}"
        return key

    return key


def _underscore_to_dots(s: str) -> str:
    """Convert module path underscores to dots, preserving numeric indices."""
    # Replace patterns like _0_ with .0.
    result = re.sub(r"_(\d+)_", r".\1.", s)
    result = re.sub(r"_(\d+)$", r".\1", result)
    # Replace remaining leading underscores with dots
    result = result.replace("_", ".")
    return result


# ---------------------------------------------------------------------------
# Batch Conversion Helpers
# ---------------------------------------------------------------------------


def convert_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    output_format: ModelFormat | str = ModelFormat.SAFETENSORS,
    component: str = "transformer",
    dtype: torch.dtype | None = None,
) -> Path:
    """High-level checkpoint conversion dispatcher.

    Detects input format automatically and converts to the target format.
    """
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if isinstance(output_format, str):
        output_format = ModelFormat(output_format)

    input_format = detect_model_format(input_path)
    logger.info("Converting %s (%s) -> %s", input_path, input_format.value, output_format.value)

    if input_format == ModelFormat.DIFFUSERS and output_format == ModelFormat.SAFETENSORS:
        return convert_diffusers_to_safetensors(input_path, output_path, component, dtype)

    if input_format == ModelFormat.SAFETENSORS and output_format == ModelFormat.DIFFUSERS:
        return convert_safetensors_to_diffusers(input_path, output_path, component)

    if input_format in (ModelFormat.LORA_SAFETENSORS, ModelFormat.LORA_TORCH):
        # Load, convert keys, save
        state_dict = _load_lora_state_dict(input_path)
        if output_format == ModelFormat.LORA_SAFETENSORS:
            from safetensors.torch import save_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(state_dict, str(output_path))
            return output_path

    raise ValueError(
        f"Unsupported conversion: {input_format.value} -> {output_format.value}"
    )


def _load_lora_state_dict(path: Path) -> dict[str, Tensor]:
    """Load a LoRA state dict from any format."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


__all__ = [
    "ModelFormat",
    "detect_model_format",
    "convert_diffusers_to_safetensors",
    "convert_safetensors_to_diffusers",
    "convert_lora_format",
    "convert_checkpoint",
]
