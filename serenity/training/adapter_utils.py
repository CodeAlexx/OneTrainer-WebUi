"""Shared utilities for LoRA and LyCORIS adapter managers.

Model type aliases, dtype coercion, module resolution, deduplication.
Extracted from lora_manager.py and lycoris_manager.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model type aliases (superset of both managers)
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "z_image": "zimage",
    "sd_15": "sd15",
    "sd_15_inpainting": "sd15_inpainting",
    "sd_20": "sd20",
    "sd_20_base": "sd20_base",
    "sd_20_inpainting": "sd20_inpainting",
    "sd_20_depth": "sd20_depth",
    "sd_21": "sd21",
    "sd_21_base": "sd21_base",
    "sd3.5": "sd35",
    "stable_diffusion_3": "sd3",
    "stable_diffusion_35": "sd35",
    "stable_diffusion_3.5": "sd35",
    "flux2": "flux_2",
    "flux_2_dev": "flux_2",
    "flux_klein": "flux_2_klein",
    "flux2_klein": "flux_2_klein",
    "flux_2_klein_4b_base": "flux_2_klein_4b",
    "flux_2_klein_9b_base": "flux_2_klein_9b",
    "flux2_klein_4b": "flux_2_klein_4b",
    "flux2_klein_9b": "flux_2_klein_9b",
    "flux_fill": "flux_fill_dev",
    "hidream": "hi_dream_full",
    "chroma": "chroma_1",
}


def normalize_model_type(value: str) -> str:
    """Normalize model type string via alias lookup."""
    normalized = str(value).strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(normalized, normalized)


def coerce_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    """Convert string/dtype to torch.dtype, or None if unrecognized."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    return None


def resolve_target_module(model_or_pipeline: Any) -> nn.Module:
    """Walk model/pipeline to find the trainable nn.Module for adapter attachment."""
    candidates: list[Any] = []
    if isinstance(model_or_pipeline, dict):
        candidates.extend(
            model_or_pipeline[key]
            for key in ("module", "pipeline", "model", "unet", "transformer")
            if key in model_or_pipeline
        )
    else:
        candidates.append(model_or_pipeline)
        candidates.extend(
            getattr(model_or_pipeline, attr)
            for attr in ("pipeline", "model", "unet", "transformer")
            if hasattr(model_or_pipeline, attr)
        )

    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("unet", "transformer", "prior_prior"):
            component = getattr(candidate, attr, None)
            if isinstance(component, nn.Module):
                return component
        if isinstance(candidate, nn.Module):
            return candidate

    raise TypeError("Could not resolve a trainable torch.nn.Module for adapter attachment.")


def dedupe(values: list[str], *, strip: bool = True) -> list[str]:
    """Deduplicate a list of strings preserving order.

    Args:
        values: Input string list.
        strip: If True, strip whitespace and skip empty strings (LoRA behavior).
               If False, use raw values (LyCORIS behavior).
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip() if strip else value
        if strip and not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


__all__ = [
    "MODEL_ALIASES",
    "normalize_model_type",
    "coerce_dtype",
    "resolve_target_module",
    "dedupe",
]
