"""Native DoRA (Weight-Decomposed Low-Rank Adaptation) implementation.

Extends LoRA by decomposing the weight update into magnitude and direction
components, yielding better training stability on many tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from serenity.training.adapters.lora import LoRALinear, apply_lora

__all__ = [
    "DoRALinear",
    "apply_dora",
]


class DoRALinear(LoRALinear):
    """LoRA variant with a learnable per-output magnitude vector.

    The forward pass normalises the adapted weight columns and rescales
    them by ``magnitude_vector``, decoupling direction from magnitude.
    """

    def __init__(
        self,
        orig: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(orig, rank=rank, alpha=alpha, dropout=dropout)
        # Magnitude initialised from column norms of the original weight
        self.magnitude_vector = nn.Parameter(
            torch.linalg.norm(orig.weight.detach(), dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adapted weight (original + low-rank delta)
        adapted_weight = self.orig.weight + (self.lora_up.weight @ self.lora_down.weight) * self.scaling
        # Normalise each output neuron's weight vector
        column_norms = torch.linalg.norm(adapted_weight, dim=1, keepdim=True)
        normed_weight = adapted_weight / (column_norms + 1e-8)
        # Rescale by the learned magnitude
        final_weight = normed_weight * self.magnitude_vector.unsqueeze(1)
        return F.linear(x, final_weight, self.orig.bias)

    def merge_weights(self) -> None:
        raise NotImplementedError(
            "DoRA merge is not supported; use LoRA if merge/unmerge is needed."
        )

    def unmerge_weights(self) -> None:
        raise NotImplementedError(
            "DoRA unmerge is not supported; use LoRA if merge/unmerge is needed."
        )


def apply_dora(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_modules: list[str],
    dropout: float = 0.0,
) -> list[DoRALinear]:
    """Replace matching ``nn.Linear`` layers with ``DoRALinear`` wrappers.

    Parameters are identical to ``apply_lora``; see its docstring for details.
    """
    return apply_lora(
        module,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        dropout=dropout,
        _wrapper_cls=DoRALinear,
    )
