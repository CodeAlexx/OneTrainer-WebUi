"""LoRA file loading and key normalisation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

__all__ = [
    "load_lora",
    "detect_lora_type",
    "normalize_lora_keys",
]

logger = logging.getLogger(__name__)

# Try safetensors; fall back to torch.load for .pt/.bin files.
try:
    from safetensors.torch import load_file as _load_safetensors

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Key-pattern constants
# ---------------------------------------------------------------------------

# Standard LoRA uses lora_up / lora_down
_STANDARD_UP = "lora_up.weight"
_STANDARD_DOWN = "lora_down.weight"

# Kohya format uses lora_A / lora_B (note: A = down, B = up)
_KOHYA_A = "lora_A.weight"
_KOHYA_B = "lora_B.weight"

# Diffusers format typically uses specific prefix patterns
_DIFFUSERS_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_lora(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a LoRA state dict from a safetensors or PyTorch file.

    Returns the raw state dict.  Use :func:`detect_lora_type` and
    :func:`normalize_lora_keys` to convert to a standard format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".safetensors",) and _SAFETENSORS_AVAILABLE:
        state_dict = _load_safetensors(str(path))
    elif suffix in (".safetensors",) and not _SAFETENSORS_AVAILABLE:
        raise RuntimeError(
            "safetensors is required to load .safetensors files but is not installed"
        )
    else:
        # .pt, .bin, .ckpt
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Expected a dict from {path}, got {type(state_dict)}")
    return state_dict


def detect_lora_type(state_dict: dict[str, torch.Tensor]) -> str:
    """Detect the LoRA format from key patterns.

    Returns one of ``"standard"``, ``"kohya"``, or ``"diffusers"``.
    """
    keys = list(state_dict.keys())
    if not keys:
        return "standard"

    has_standard = any(_STANDARD_UP in k or _STANDARD_DOWN in k for k in keys)
    has_kohya = any(_KOHYA_A in k or _KOHYA_B in k for k in keys)
    has_diffusers = any(k.startswith(_DIFFUSERS_PREFIX) for k in keys)

    if has_diffusers:
        return "diffusers"
    if has_kohya and not has_standard:
        return "kohya"
    return "standard"


def normalize_lora_keys(
    state_dict: dict[str, torch.Tensor],
    lora_type: str | None = None,
) -> dict[str, torch.Tensor]:
    """Convert LoRA state dict keys to the standard format.

    Standard format uses ``lora_up.weight`` / ``lora_down.weight`` keys.
    Alpha tensors are preserved under their original ``alpha`` key names.
    """
    if lora_type is None:
        lora_type = detect_lora_type(state_dict)

    if lora_type == "standard":
        return dict(state_dict)

    normalised: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        new_key = key

        if lora_type == "kohya":
            # lora_A.weight -> lora_down.weight
            # lora_B.weight -> lora_up.weight
            new_key = new_key.replace("lora_A.weight", "lora_down.weight")
            new_key = new_key.replace("lora_B.weight", "lora_up.weight")

        elif lora_type == "diffusers":
            # Strip the diffusers prefix
            if new_key.startswith(_DIFFUSERS_PREFIX):
                new_key = new_key[len(_DIFFUSERS_PREFIX) :]
            # Also normalise A/B -> down/up
            new_key = new_key.replace("lora_A.weight", "lora_down.weight")
            new_key = new_key.replace("lora_B.weight", "lora_up.weight")

        normalised[new_key] = tensor

    return normalised
