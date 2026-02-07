"""Checkpoint and adapter weight loading.

Ported from OneTrainer's modelLoader hierarchy:
- BaseModelLoader._load_internal_state  -> optimizer / EMA / meta.json
- safetensors.torch.load_file           -> LoRA state dicts

Serenity provides simple free-functions instead of a class-per-model tree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from safetensors.torch import load_file as _sf_load
except ImportError:  # pragma: no cover - optional dependency
    _sf_load = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class CheckpointFormat:
    """Known checkpoint formats."""
    SAFETENSORS = "safetensors"
    DIFFUSERS = "diffusers"
    INTERNAL = "internal"
    TORCH_PT = "torch_pt"
    UNKNOWN = "unknown"


def detect_format(path: str | Path) -> str:
    """Detect the checkpoint format at *path*.

    Returns one of the ``CheckpointFormat`` constants.

    Heuristics
    ----------
    * Single ``.safetensors`` file -> ``SAFETENSORS``
    * Single ``.pt`` / ``.bin`` file -> ``TORCH_PT``
    * Directory with ``meta.json`` -> ``INTERNAL`` (Serenity/OneTrainer)
    * Directory with ``model_index.json`` -> ``DIFFUSERS``
    * Directory with any ``.safetensors`` -> ``DIFFUSERS`` (likely)
    """
    p = Path(path).expanduser()

    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".safetensors":
            return CheckpointFormat.SAFETENSORS
        if suffix in {".pt", ".pth", ".bin", ".ckpt"}:
            return CheckpointFormat.TORCH_PT
        return CheckpointFormat.UNKNOWN

    if p.is_dir():
        if (p / "meta.json").exists():
            return CheckpointFormat.INTERNAL
        if (p / "model_index.json").exists():
            return CheckpointFormat.DIFFUSERS
        # Diffusers sub-directories commonly contain safetensors shards
        if any(p.glob("**/*.safetensors")):
            return CheckpointFormat.DIFFUSERS
        if any(p.glob("*.pt")):
            return CheckpointFormat.TORCH_PT
        return CheckpointFormat.UNKNOWN

    return CheckpointFormat.UNKNOWN


# ---------------------------------------------------------------------------
# LoRA / adapter loading
# ---------------------------------------------------------------------------

def load_lora(path: str | Path) -> dict[str, Tensor]:
    """Load LoRA / adapter weights from a ``.safetensors`` (preferred) or ``.pt`` file.

    Returns the raw state dict with original key names.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Adapter weights not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        if _sf_load is None:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install it with: pip install safetensors"
            )
        state_dict = _sf_load(str(p), device="cpu")
        logger.info("Loaded LoRA weights <- %s (%d tensors)", p, len(state_dict))
        return state_dict

    if suffix in {".pt", ".pth", ".bin", ".ckpt"}:
        state_dict = torch.load(str(p), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict from {p}, got {type(state_dict).__name__}")
        logger.info("Loaded LoRA weights <- %s (%d tensors)", p, len(state_dict))
        return state_dict

    raise ValueError(f"Unsupported adapter file format: {suffix}")


# ---------------------------------------------------------------------------
# Full training checkpoint loading
# ---------------------------------------------------------------------------

@dataclass
class CheckpointData:
    """Deserialized training checkpoint.

    Mirrors the directory layout written by ``ModelSaver.save_checkpoint``.
    """
    model_state: dict[str, Tensor] | None = None
    optimizer_state: dict[str, Any] | None = None
    ema_state: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


def load_checkpoint(path: str | Path) -> CheckpointData:
    """Load a Serenity / OneTrainer internal checkpoint directory.

    Expected layout::

        <path>/
            model.safetensors | lora/lora.safetensors
            optimizer/optimizer.pt
            ema/ema.pt
            meta.json

    All components are optional; missing ones are set to ``None``.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {root}")

    data = CheckpointData()

    # -- Model / adapter weights -----------------------------------------------
    model_sf = root / "model.safetensors"
    lora_sf = root / "lora" / "lora.safetensors"

    if model_sf.exists():
        if _sf_load is None:
            raise ImportError("safetensors is required to load model.safetensors")
        data.model_state = _sf_load(str(model_sf), device="cpu")
        logger.info("Loaded model state <- %s (%d tensors)", model_sf, len(data.model_state))
    elif lora_sf.exists():
        if _sf_load is None:
            raise ImportError("safetensors is required to load lora.safetensors")
        data.model_state = _sf_load(str(lora_sf), device="cpu")
        logger.info("Loaded LoRA state <- %s (%d tensors)", lora_sf, len(data.model_state))

    # -- Optimizer state -------------------------------------------------------
    opt_pt = root / "optimizer" / "optimizer.pt"
    if opt_pt.exists():
        data.optimizer_state = torch.load(str(opt_pt), map_location="cpu", weights_only=True)
        logger.info("Loaded optimizer state <- %s", opt_pt)

    # -- EMA state -------------------------------------------------------------
    ema_pt = root / "ema" / "ema.pt"
    if ema_pt.exists():
        data.ema_state = torch.load(str(ema_pt), map_location="cpu", weights_only=True)
        logger.info("Loaded EMA state <- %s", ema_pt)

    # -- Meta / progress -------------------------------------------------------
    meta_json = root / "meta.json"
    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as f:
            data.meta = json.load(f)
        data.progress = data.meta.get("train_progress")
        logger.info("Loaded meta <- %s", meta_json)

    return data


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embedding(path: str | Path) -> dict[str, Tensor]:
    """Load a textual-inversion embedding from safetensors or pt."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Embedding not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        if _sf_load is None:
            raise ImportError("safetensors is required to load .safetensors embeddings")
        return _sf_load(str(p), device="cpu")

    if suffix in {".pt", ".pth", ".bin"}:
        loaded = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(loaded, Tensor):
            return {"emb_params": loaded}
        if isinstance(loaded, dict):
            return loaded
        raise TypeError(f"Unexpected type in embedding file: {type(loaded).__name__}")

    raise ValueError(f"Unsupported embedding format: {suffix}")


__all__ = [
    "CheckpointFormat",
    "CheckpointData",
    "detect_format",
    "load_lora",
    "load_checkpoint",
    "load_embedding",
]
