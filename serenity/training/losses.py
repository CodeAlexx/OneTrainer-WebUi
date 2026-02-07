"""Loss utilities for feature parity checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def snr_weighted_loss(pred: torch.Tensor, target: torch.Tensor, snr: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.mse_loss(pred, target, reduction="none")
    if snr is None:
        return loss.mean()
    weight = 1.0 / (snr + 1.0)
    while weight.dim() < loss.dim():
        weight = weight.unsqueeze(-1)
    return (loss * weight).mean()


def velocity_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


__all__ = [
    "flow_matching_loss",
    "snr_weighted_loss",
    "velocity_loss",
]
