"""Offline LoRA merge — apply / unapply LoRA weights to a model."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "merge_lora_to_weight",
    "weight_decompose",
    "merge_lora_into_model",
    "unmerge_lora_from_model",
    "dequantize_if_needed",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantized weight helpers (P4.13)
# ---------------------------------------------------------------------------


def dequantize_if_needed(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Dequantize a weight tensor if it is a quantized type.

    Returns ``(dequantized_weight, was_quantized)``.
    """
    # Check for GGUF quantized tensor (has gguf_cls attribute)
    if hasattr(weight, "gguf_cls") and weight.gguf_cls is not None:
        from serenity.inference.quantization.gguf import dequantize_tensor

        return dequantize_tensor(weight, torch.float32), True

    # Check for PyTorch native quantized tensors (qint8, quint8, etc.)
    if weight.is_quantized:
        return weight.dequantize().float(), True

    # Check for __tensor_flatten__ protocol (generic quantized tensor, e.g. AffineQuantizedTensor)
    if hasattr(weight, "__tensor_flatten__"):
        try:
            inner, _meta = weight.__tensor_flatten__()
            main_key = next(iter(inner.keys()))
            return inner[main_key].float(), True
        except Exception:
            pass

    return weight, False


def _is_quantized_layer(mod: nn.Module) -> bool:
    """Check if a module is a quantized linear layer."""
    cls_name = type(mod).__name__
    return cls_name in ("BnbLinear4bit", "GGUFLinear", "Int8Linear", "NunchakuLinear")

# Attribute used to store backup weights for unmerge
_BACKUP_ATTR = "_serenity_lora_backup"


# ---------------------------------------------------------------------------
# Low-level weight operations (from Forge patcher/lora.py)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def weight_decompose(
    dora_scale: torch.Tensor,
    weight: torch.Tensor,
    lora_diff: torch.Tensor,
    alpha: float = 1.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """DoRA (Weight-Decomposed Low-Rank Adaptation) decomposition.

    Adjusts the weight direction using *dora_scale* while keeping the
    magnitude separate.  Derived from Forge's ``lora.py:26``.
    """
    lora_diff = lora_diff * alpha
    weight_calc = weight + lora_diff.to(weight.dtype)

    # Compute column-norm depending on which axis dora_scale matches
    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
        # Reshape dora_scale to be broadcastable: (out_features, 1, 1, ...)
        dora_scale = dora_scale.reshape(weight.shape[0], *[1] * (weight.dim() - 1))
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(
                weight_calc.shape[1], *[1] * (weight_calc.dim() - 1)
            )
            .transpose(0, 1)
        )
        # Reshape dora_scale for the non-output axis
        dora_scale = dora_scale.reshape(
            1, weight_calc.shape[1], *[1] * (weight_calc.dim() - 2)
        ) if weight_calc.dim() > 1 else dora_scale

    weight_norm = weight_norm + torch.finfo(weight.dtype).eps
    weight_calc = weight_calc * (dora_scale / weight_norm).to(weight.dtype)

    if strength != 1.0:
        weight_calc = weight_calc - weight
        weight = weight + strength * weight_calc
    else:
        weight = weight_calc

    return weight


@torch.inference_mode()
def merge_lora_to_weight(
    weight: torch.Tensor,
    lora_patches: list[dict[str, Any]],
    strength: float = 1.0,
    computation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Merge a list of LoRA patches into *weight*.

    Each element in *lora_patches* is a dict with:
    - ``"type"``: ``"diff"`` (additive) or ``"set"`` (replacement)
    - ``"diff"``: the diff tensor for ``"diff"`` type
    - ``"up"``, ``"down"``: the LoRA up/down matrices
    - ``"alpha"`` (optional): LoRA alpha scaling factor
    - ``"dora_scale"`` (optional): DoRA scale for weight decomposition

    For the simple low-rank case the math is::

        weight += strength * (alpha / rank) * (up @ down)
    """
    # Dequantize if the weight is a quantized type (BnB, GGUF, etc.)
    weight, was_quantized = dequantize_if_needed(weight)

    weight_backup_dtype = weight.dtype
    if computation_dtype != weight.dtype:
        weight = weight.to(computation_dtype)
    else:
        weight = weight.clone()

    for patch in lora_patches:
        patch_type = patch.get("type", "diff")

        if patch_type == "set":
            # Full weight replacement
            replacement = patch["weight"]
            weight.copy_(replacement.to(weight.dtype))
            continue

        if patch_type == "diff":
            diff = patch.get("diff")
            if diff is not None:
                # Simple additive diff
                if diff.shape != weight.shape:
                    logger.warning(
                        "Shape mismatch for diff patch: %s vs %s — skipping",
                        diff.shape,
                        weight.shape,
                    )
                    continue
                weight += strength * diff.to(device=weight.device, dtype=weight.dtype)
                continue

        # Standard LoRA: up @ down
        up = patch.get("up")
        down = patch.get("down")
        if up is None or down is None:
            logger.warning("LoRA patch missing up/down matrices — skipping")
            continue

        up = up.to(device=weight.device, dtype=computation_dtype)
        down = down.to(device=weight.device, dtype=computation_dtype)

        alpha = patch.get("alpha")
        rank = down.shape[0]
        if alpha is not None:
            scale = (alpha / rank) * strength
        else:
            scale = strength

        # Compute low-rank diff
        if up.ndim == 2 and down.ndim == 2:
            lora_diff = up @ down
        else:
            # Conv layers: reshape for matmul
            lora_diff = torch.einsum("i...,ij->j...", up.flatten(1), down.flatten(1))
            lora_diff = lora_diff.reshape(weight.shape)

        # DoRA path
        dora_scale = patch.get("dora_scale")
        if dora_scale is not None:
            weight = weight_decompose(
                dora_scale.to(device=weight.device, dtype=computation_dtype),
                weight,
                lora_diff,
                alpha=scale,
                strength=1.0,
            )
        else:
            weight += scale * lora_diff.to(weight.dtype)

    if weight_backup_dtype != weight.dtype:
        weight = weight.to(weight_backup_dtype)

    return weight


# ---------------------------------------------------------------------------
# Model-level merge / unmerge
# ---------------------------------------------------------------------------


def _find_lora_pairs(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Group LoRA state dict entries by module prefix.

    Returns a dict mapping module-key -> { "up": Tensor, "down": Tensor,
    "alpha": Tensor | None }.
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if "lora_up.weight" in key:
            prefix = key.replace("lora_up.weight", "").rstrip(".")
            pairs.setdefault(prefix, {})["up"] = tensor
        elif "lora_down.weight" in key:
            prefix = key.replace("lora_down.weight", "").rstrip(".")
            pairs.setdefault(prefix, {})["down"] = tensor
        elif ".alpha" in key:
            prefix = key.replace(".alpha", "").rstrip(".")
            pairs.setdefault(prefix, {})["alpha"] = tensor
    return pairs


def _resolve_module(
    model: nn.Module, prefix: str
) -> nn.Module | None:
    """Walk *model* to find the module at *prefix* (dot-separated)."""
    parts = prefix.split(".")
    current = model
    for part in parts:
        if not part:
            continue
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


@torch.inference_mode()
def merge_lora_into_model(
    model: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    strength: float = 1.0,
) -> None:
    """Merge LoRA weights into a model's parameters in-place.

    Backs up original weights so that :func:`unmerge_lora_from_model` can
    restore them later.
    """
    pairs = _find_lora_pairs(lora_state_dict)

    if not pairs:
        logger.warning("No LoRA pairs found in state dict")
        return

    if not hasattr(model, _BACKUP_ATTR):
        setattr(model, _BACKUP_ATTR, {})
    backup: dict[str, torch.Tensor] = getattr(model, _BACKUP_ATTR)

    merged_count = 0
    for prefix, tensors in pairs.items():
        up = tensors.get("up")
        down = tensors.get("down")
        if up is None or down is None:
            continue

        # Try to find the target weight
        mod = _resolve_module(model, prefix)
        if mod is None or not hasattr(mod, "weight"):
            logger.debug("No module found at %s — skipping", prefix)
            continue

        weight_param = mod.weight
        if not isinstance(weight_param, nn.Parameter):
            continue

        # Backup original weight (only on first merge)
        weight_key = prefix + ".weight"
        if weight_key not in backup:
            backup[weight_key] = weight_param.data.clone()

        alpha_tensor = tensors.get("alpha")
        alpha_val = alpha_tensor.item() if alpha_tensor is not None else None

        # For quantized layers, use the layer's dequantize method for best accuracy
        is_quant = _is_quantized_layer(mod)
        if is_quant and hasattr(mod, "dequantize_weight"):
            source_weight = mod.dequantize_weight()
        else:
            source_weight = weight_param.data

        patch = {
            "type": "diff",
            "up": up,
            "down": down,
            "alpha": alpha_val,
        }
        new_weight = merge_lora_to_weight(
            source_weight, [patch], strength=strength
        )

        if is_quant:
            # For quantized layers, store the merged (dequantized) float result.
            # NOTE: the weight is now float, not quantized.  Re-quantization does
            # NOT happen automatically.  This means the layer will use the
            # dequantize-then-matmul fallback path, increasing memory usage.
            # Use online LoRA (online.py) if preserving quantization is required.
            logger.debug(
                "Merging LoRA into quantized layer %s (%s) — weight is now float",
                prefix,
                type(mod).__name__,
            )
        weight_param.data.copy_(new_weight)
        merged_count += 1

    logger.info("Merged LoRA into %d layers (strength=%.2f)", merged_count, strength)


@torch.inference_mode()
def unmerge_lora_from_model(model: nn.Module) -> None:
    """Restore original weights, undoing a previous :func:`merge_lora_into_model`."""
    backup: dict[str, torch.Tensor] | None = getattr(model, _BACKUP_ATTR, None)
    if not backup:
        logger.warning("No LoRA backup found — nothing to unmerge")
        return

    restored = 0
    for weight_key, original_data in backup.items():
        # weight_key is like "some.module.weight"
        prefix = weight_key.rsplit(".weight", 1)[0]
        mod = _resolve_module(model, prefix)
        if mod is not None and hasattr(mod, "weight"):
            mod.weight.data.copy_(original_data)
            restored += 1

    backup.clear()
    logger.info("Restored %d layers to pre-LoRA state", restored)
