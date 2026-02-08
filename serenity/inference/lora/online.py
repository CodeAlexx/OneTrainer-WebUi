"""Online LoRA — apply LoRA at inference time without modifying weights.

For quantized models where weight merging is impossible (e.g. INT8, BnB, GGUF),
LoRA patches are stored on each module and applied dynamically during the
forward pass via ``get_weight_and_bias()`` in ``quantization.ops``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

__all__ = [
    "apply_online_lora",
    "remove_online_lora",
]

logger = logging.getLogger(__name__)

# Attribute name used to store online LoRA patches on modules
_ONLINE_ATTR = "_online_lora_patches"


def _compute_lora_delta(
    up: torch.Tensor,
    down: torch.Tensor,
    alpha: float | None,
    strength: float,
) -> torch.Tensor:
    """Compute the additive LoRA delta: ``strength * (alpha/rank) * (up @ down)``."""
    rank = down.shape[0]
    if alpha is not None:
        scale = (alpha / rank) * strength
    else:
        scale = strength

    if up.ndim == 2 and down.ndim == 2:
        delta = scale * (up @ down)
    else:
        delta = scale * torch.einsum("i...,ij->j...", up.flatten(1), down.flatten(1))

    return delta


def apply_online_lora(
    model: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    strength: float = 1.0,
) -> None:
    """Attach online LoRA patches to model modules for forward-time application.

    Patches are stored in ``module._online_lora_patches["weight"]`` as a list
    of additive delta tensors.  The ``get_weight_and_bias()`` function in
    ``quantization.ops`` applies them during the forward pass.
    """
    from serenity.inference.lora.merge import _find_lora_pairs, _resolve_module

    pairs = _find_lora_pairs(lora_state_dict)
    applied = 0

    for prefix, tensors in pairs.items():
        up = tensors.get("up")
        down = tensors.get("down")
        if up is None or down is None:
            continue

        mod = _resolve_module(model, prefix)
        if mod is None or not hasattr(mod, "weight"):
            logger.debug("No module at %s for online LoRA — skipping", prefix)
            continue

        alpha_tensor = tensors.get("alpha")
        alpha_val = alpha_tensor.item() if alpha_tensor is not None else None

        delta = _compute_lora_delta(up, down, alpha_val, strength)

        # Reshape delta to match weight shape if needed
        weight = mod.weight
        if delta.shape != weight.shape:
            try:
                delta = delta.reshape(weight.shape)
            except RuntimeError:
                logger.warning(
                    "Cannot reshape LoRA delta %s to weight %s at %s — skipping",
                    delta.shape,
                    weight.shape,
                    prefix,
                )
                continue

        if not hasattr(mod, _ONLINE_ATTR):
            setattr(mod, _ONLINE_ATTR, {})

        loras: dict[str, list[torch.Tensor]] = getattr(mod, _ONLINE_ATTR)
        loras.setdefault("weight", []).append(delta)
        applied += 1

    logger.info(
        "Applied online LoRA to %d modules (strength=%.2f)", applied, strength
    )


def remove_online_lora(model: nn.Module) -> None:
    """Remove all online LoRA patches from *model*."""
    removed = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, _ONLINE_ATTR):
            delattr(mod, _ONLINE_ATTR)
            removed += 1

    logger.info("Removed online LoRA from %d modules", removed)
