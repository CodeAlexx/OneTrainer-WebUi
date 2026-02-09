"""Native LoRA (Low-Rank Adaptation) implementation.

Replaces PEFT with a lightweight, Serenity-native LoRA that wraps
``nn.Linear`` layers with trainable low-rank decomposition matrices.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "apply_lora",
    "get_lora_params",
    "extract_lora_state_dict",
    "load_lora_state_dict",
    "merge_lora",
    "unmerge_lora",
]


# ---------------------------------------------------------------------------
# Core LoRA module
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with low-rank adaptation.

    The original linear is kept frozen; only ``lora_down`` and ``lora_up``
    are trainable.
    """

    def __init__(
        self,
        orig: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.orig = orig
        for p in self.orig.parameters():
            p.requires_grad_(False)

        in_features = orig.in_features
        out_features = orig.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        # Initialisation: kaiming uniform for down, zero for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.orig(x)
        if not self.merged:
            result = result + self.dropout(self.lora_up(self.lora_down(x))) * self.scaling
        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original linear."""
        if self.merged:
            return
        with torch.no_grad():
            delta = (self.lora_up.weight @ self.lora_down.weight) * self.scaling
            self.orig.weight.data.add_(delta)
        self.merged = True

    def unmerge_weights(self) -> None:
        """Subtract LoRA weights from the original linear."""
        if not self.merged:
            return
        with torch.no_grad():
            delta = (self.lora_up.weight @ self.lora_down.weight) * self.scaling
            self.orig.weight.data.sub_(delta)
        self.merged = False


# ---------------------------------------------------------------------------
# Module-tree helpers
# ---------------------------------------------------------------------------


def _set_submodule(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named child on *parent* (supports dotted paths)."""
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _walk_linears(
    module: nn.Module,
    target_modules: list[str],
) -> list[tuple[str, nn.Linear]]:
    """Find ``nn.Linear`` layers whose fully-qualified name contains a target string."""
    matches: list[tuple[str, nn.Linear]] = []
    for fqn, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if any(target in fqn for target in target_modules):
            matches.append((fqn, child))
    return matches


def apply_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_modules: list[str],
    dropout: float = 0.0,
    *,
    _wrapper_cls: type[LoRALinear] | None = None,
) -> list[LoRALinear]:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear`` wrappers.

    Parameters
    ----------
    module:
        Root module to modify in-place.
    rank:
        LoRA rank (inner dimension of the low-rank matrices).
    alpha:
        LoRA alpha scaling factor.
    target_modules:
        Substrings to match against fully-qualified module names.
    dropout:
        Dropout probability applied to the LoRA path.

    Returns
    -------
    list[LoRALinear]
        All newly created adapter layers.
    """
    wrapper_cls = _wrapper_cls or LoRALinear
    matches = _walk_linears(module, target_modules)
    created: list[LoRALinear] = []
    for fqn, linear in matches:
        wrapper = wrapper_cls(linear, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(module, fqn, wrapper)
        created.append(wrapper)
    return created


# ---------------------------------------------------------------------------
# Parameter and state-dict utilities
# ---------------------------------------------------------------------------


def get_lora_params(module: nn.Module) -> list[nn.Parameter]:
    """Collect all trainable LoRA parameters from a module tree."""
    params: list[nn.Parameter] = []
    for child in module.modules():
        if isinstance(child, LoRALinear):
            params.extend(p for p in child.lora_down.parameters())
            params.extend(p for p in child.lora_up.parameters())
            # DoRALinear subclass exposes magnitude_vector as a direct attribute
            mag = getattr(child, "magnitude_vector", None)
            if mag is not None and isinstance(mag, nn.Parameter):
                params.append(mag)
    return params


def extract_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA weights keyed by their fully-qualified names."""
    state: dict[str, torch.Tensor] = {}
    for fqn, child in module.named_modules():
        if isinstance(child, LoRALinear):
            prefix = f"{fqn}." if fqn else ""
            state[f"{prefix}lora_down.weight"] = child.lora_down.weight.data.clone()
            state[f"{prefix}lora_up.weight"] = child.lora_up.weight.data.clone()
            mag = getattr(child, "magnitude_vector", None)
            if mag is not None and isinstance(mag, nn.Parameter):
                state[f"{prefix}magnitude_vector"] = mag.data.clone()
    return state


def load_lora_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load LoRA weights back into the module tree.

    Parameters
    ----------
    module:
        Root module containing ``LoRALinear`` layers.
    state_dict:
        Mapping produced by ``extract_lora_state_dict``.
    strict:
        If True, raise on missing or unexpected keys.
    """
    lora_modules: dict[str, LoRALinear] = {
        fqn: child for fqn, child in module.named_modules() if isinstance(child, LoRALinear)
    }

    consumed: set[str] = set()
    for fqn, lora_mod in lora_modules.items():
        prefix = f"{fqn}." if fqn else ""
        down_key = f"{prefix}lora_down.weight"
        up_key = f"{prefix}lora_up.weight"
        mag_key = f"{prefix}magnitude_vector"

        if down_key in state_dict:
            lora_mod.lora_down.weight.data.copy_(state_dict[down_key])
            consumed.add(down_key)
        elif strict:
            raise KeyError(f"Missing key in state_dict: {down_key}")

        if up_key in state_dict:
            lora_mod.lora_up.weight.data.copy_(state_dict[up_key])
            consumed.add(up_key)
        elif strict:
            raise KeyError(f"Missing key in state_dict: {up_key}")

        mag = getattr(lora_mod, "magnitude_vector", None)
        if mag is not None and isinstance(mag, nn.Parameter) and mag_key in state_dict:
            mag.data.copy_(state_dict[mag_key])
            consumed.add(mag_key)

    if strict:
        unexpected = set(state_dict.keys()) - consumed
        if unexpected:
            raise KeyError(f"Unexpected keys in state_dict: {unexpected}")


# ---------------------------------------------------------------------------
# Merge / unmerge helpers
# ---------------------------------------------------------------------------


def merge_lora(module: nn.Module) -> None:
    """Merge all LoRA weights into their originals across the module tree."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.merge_weights()


def unmerge_lora(module: nn.Module) -> None:
    """Unmerge all LoRA weights from their originals across the module tree."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.unmerge_weights()
