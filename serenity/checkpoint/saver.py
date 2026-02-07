"""Model saving in safetensors, diffusers, and internal checkpoint formats.

Ported from OneTrainer's modelSaver hierarchy:
- DtypeModelSaverMixin  -> dtype conversion + safetensors header
- LoRASaverMixin        -> LoRA/adapter safetensors
- InternalModelSaverMixin -> optimizer + EMA + meta.json
- StableDiffusionModelSaver -> diffusers save_pretrained

Serenity collapses the deep mixin/class-per-model tree into a single
``ModelSaver`` that dispatches on format and training method.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from safetensors.torch import save_file as _sf_save
except ImportError:  # pragma: no cover - optional dependency
    _sf_save = None

from serenity.core.interfaces import ModelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_safetensors() -> None:
    """Raise early if safetensors is missing."""
    if _sf_save is None:
        raise ImportError(
            "safetensors is required for saving models. "
            "Install it with: pip install safetensors"
        )


def _convert_state_dict_dtype(
    state_dict: dict[str, Tensor],
    dtype: torch.dtype | None,
) -> dict[str, Tensor]:
    """Move every tensor to CPU and optionally cast to *dtype*.

    Mirrors OneTrainer ``DtypeModelSaverMixin._convert_state_dict_dtype``.
    """
    if dtype is None:
        return {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    return {
        k: v.detach().to(device="cpu", dtype=dtype).contiguous()
        for k, v in state_dict.items()
    }


def _calculate_safetensors_hash(state_dict: dict[str, Tensor]) -> str:
    """SHA-256 over ordered tensor bytes for safetensors header."""
    sha = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # Convert tensor to bytes in a stable way
        sha.update(key.encode("utf-8"))
        sha.update(tensor.cpu().contiguous().numpy().tobytes())
    return f"0x{sha.hexdigest()}"


def _build_safetensors_header(
    model_type: ModelType | str | None = None,
    training_method: str | None = None,
    metadata: dict[str, str] | None = None,
    state_dict: dict[str, Tensor] | None = None,
) -> dict[str, str]:
    """Build a safetensors metadata header.

    Compatible with the header format produced by OneTrainer's
    ``DtypeModelSaverMixin._create_safetensors_header`` but uses Serenity
    branding.
    """
    header: dict[str, str] = {}

    header["serenity_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if model_type is not None:
        header["serenity_model_type"] = str(model_type)

    if training_method is not None:
        header["serenity_training_method"] = str(training_method)

    if state_dict is not None:
        header["hash_sha256"] = _calculate_safetensors_hash(state_dict)

    # Kohya-compat keys so A1111/Forge/ComfyUI can detect model version
    if model_type is not None:
        mt = str(model_type).lower()
        if "sdxl" in mt:
            header["ss_base_model_version"] = "sdxl_"
        elif mt in {"sd20", "sd20_base", "sd20_inpainting", "sd20_depth", "sd21", "sd21_base"}:
            header["ss_v2"] = "True"

    if metadata:
        header.update(metadata)

    return header


# ---------------------------------------------------------------------------
# ModelSaver
# ---------------------------------------------------------------------------

@dataclass
class ModelSaver:
    """Unified model/adapter saver for Serenity.

    Saves LoRA/adapter weights, full models (diffusers), training
    checkpoints (optimizer + EMA + progress), and textual-inversion
    embeddings -- all in safetensors when possible.
    """

    model_type: ModelType | str | None = None
    training_method: str | None = None
    default_dtype: torch.dtype | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    # -- LoRA / adapter weights ------------------------------------------------

    def save_lora(
        self,
        state_dict: dict[str, Tensor],
        path: str | Path,
        *,
        dtype: torch.dtype | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        """Save adapter weights as a single ``.safetensors`` file.

        Parameters
        ----------
        state_dict:
            Adapter state dict (e.g. from ``LyCORISManager.state_dict()``).
        path:
            Destination file path.  Parent directories are created as needed.
        dtype:
            Optional output dtype cast.  Falls back to ``self.default_dtype``.
        metadata:
            Extra key/value pairs written into the safetensors header.
        """
        _ensure_safetensors()
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        cast_dtype = dtype or self.default_dtype
        save_dict = _convert_state_dict_dtype(state_dict, cast_dtype)

        merged_meta = {**self.extra_metadata}
        if metadata:
            merged_meta.update(metadata)
        header = _build_safetensors_header(
            model_type=self.model_type,
            training_method=self.training_method or "lora",
            metadata=merged_meta,
            state_dict=save_dict,
        )

        _sf_save(save_dict, str(out), header)
        logger.info("Saved LoRA weights -> %s (%d tensors)", out, len(save_dict))
        return out

    # -- Full model (diffusers) ------------------------------------------------

    def save_full_model(
        self,
        pipeline_or_model: Any,
        path: str | Path,
        *,
        format: str = "diffusers",
        dtype: torch.dtype | None = None,
    ) -> Path:
        """Save a full model in diffusers ``save_pretrained`` format.

        Parameters
        ----------
        pipeline_or_model:
            A diffusers pipeline, a model with ``.save_pretrained``, or an
            ``nn.Module`` (falls back to safetensors state-dict save).
        path:
            Destination directory (for diffusers) or file (for safetensors).
        format:
            ``"diffusers"`` (default) or ``"safetensors"``.
        dtype:
            Optional output dtype cast.
        """
        out = Path(path).expanduser()
        cast_dtype = dtype or self.default_dtype

        if format == "diffusers":
            return self._save_diffusers(pipeline_or_model, out, cast_dtype)
        elif format == "safetensors":
            return self._save_model_safetensors(pipeline_or_model, out, cast_dtype)
        else:
            raise ValueError(f"Unknown save format: {format!r}  (expected 'diffusers' or 'safetensors')")

    def _save_diffusers(
        self,
        pipeline_or_model: Any,
        out: Path,
        dtype: torch.dtype | None,
    ) -> Path:
        """Save via diffusers ``save_pretrained``."""
        import copy

        out.mkdir(parents=True, exist_ok=True)

        if dtype is not None:
            save_obj = copy.deepcopy(pipeline_or_model)
            save_obj.to(device="cpu", dtype=dtype)
        else:
            save_obj = pipeline_or_model

        if hasattr(save_obj, "save_pretrained"):
            save_obj.save_pretrained(str(out))
        else:
            raise TypeError(
                f"Object of type {type(save_obj).__name__} does not support "
                "save_pretrained.  Use format='safetensors' for raw state-dict saves."
            )

        if dtype is not None:
            del save_obj

        logger.info("Saved full model (diffusers) -> %s", out)
        return out

    def _save_model_safetensors(
        self,
        model: Any,
        out: Path,
        dtype: torch.dtype | None,
    ) -> Path:
        """Save a model's state_dict as safetensors."""
        _ensure_safetensors()
        out.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "state_dict"):
            raw_dict = model.state_dict()
        elif isinstance(model, dict):
            raw_dict = model
        else:
            raise TypeError(
                f"Cannot extract state_dict from {type(model).__name__}"
            )

        save_dict = _convert_state_dict_dtype(raw_dict, dtype)

        header = _build_safetensors_header(
            model_type=self.model_type,
            training_method=self.training_method or "fine_tune",
            metadata=self.extra_metadata,
            state_dict=save_dict,
        )

        _sf_save(save_dict, str(out), header)
        logger.info("Saved full model (safetensors) -> %s (%d tensors)", out, len(save_dict))
        return out

    # -- Training checkpoint ---------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        model_state: dict[str, Tensor] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        ema_state: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a full training checkpoint (model + optimizer + EMA + meta).

        Mirrors OneTrainer's ``InternalModelSaverMixin._save_internal_data``
        but keeps everything under one directory.

        Layout::

            <path>/
                model.safetensors   -- model or adapter weights
                optimizer/
                    optimizer.pt    -- optimizer state dict
                ema/
                    ema.pt          -- EMA state dict
                meta.json           -- progress + extra metadata
        """
        out = Path(path).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        # Model weights
        if model_state is not None:
            _ensure_safetensors()
            model_path = out / "model.safetensors"
            save_dict = _convert_state_dict_dtype(model_state, None)
            header = _build_safetensors_header(
                model_type=self.model_type,
                training_method=self.training_method,
                state_dict=save_dict,
            )
            _sf_save(save_dict, str(model_path), header)

        # Optimizer
        if optimizer_state is not None:
            opt_dir = out / "optimizer"
            opt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(optimizer_state, str(opt_dir / "optimizer.pt"))

        # EMA
        if ema_state is not None:
            ema_dir = out / "ema"
            ema_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ema_state, str(ema_dir / "ema.pt"))

        # Meta
        meta: dict[str, Any] = {}
        if progress is not None:
            meta["train_progress"] = progress
        if extra is not None:
            meta["extra"] = extra
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        if self.model_type is not None:
            meta["model_type"] = str(self.model_type)
        if self.training_method is not None:
            meta["training_method"] = str(self.training_method)

        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved training checkpoint -> %s", out)
        return out

    # -- Embedding (textual inversion) -----------------------------------------

    def save_embedding(
        self,
        embedding: dict[str, Tensor] | Tensor,
        path: str | Path,
        *,
        dtype: torch.dtype | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        """Save a textual-inversion embedding as safetensors.

        Parameters
        ----------
        embedding:
            Either a state dict ``{name: tensor}`` or a single tensor.
        path:
            Destination ``.safetensors`` file.
        dtype:
            Optional cast dtype.
        metadata:
            Extra safetensors header entries.
        """
        _ensure_safetensors()
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(embedding, Tensor):
            state_dict: dict[str, Tensor] = {"emb_params": embedding}
        else:
            state_dict = dict(embedding)

        cast_dtype = dtype or self.default_dtype
        save_dict = _convert_state_dict_dtype(state_dict, cast_dtype)

        merged_meta = {**self.extra_metadata}
        if metadata:
            merged_meta.update(metadata)
        header = _build_safetensors_header(
            model_type=self.model_type,
            training_method="embedding",
            metadata=merged_meta,
            state_dict=save_dict,
        )

        _sf_save(save_dict, str(out), header)
        logger.info("Saved embedding -> %s (%d tensors)", out, len(save_dict))
        return out


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_saver(
    model_type: ModelType | str | None = None,
    training_method: str | None = None,
    dtype: torch.dtype | None = None,
    **metadata: str,
) -> ModelSaver:
    """Create a ``ModelSaver`` with common defaults."""
    return ModelSaver(
        model_type=model_type,
        training_method=training_method,
        default_dtype=dtype,
        extra_metadata=dict(metadata),
    )


__all__ = [
    "ModelSaver",
    "create_saver",
]
