"""Synchronization and device helpers for memory management."""

from __future__ import annotations

import gc
from typing import Optional, Sequence

import torch


def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_equals(a: Optional[torch.device], b: Optional[torch.device]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.type == b.type and a.index == b.index


def tensors_match_device(
    tensors: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
    indices: Optional[Sequence[int]] = None,
) -> bool:
    if isinstance(tensors, torch.Tensor):
        return device_equals(tensors.device, device)

    if indices is None:
        iterable = tensors
    else:
        iterable = [tensors[i] for i in indices]

    for tensor in iterable:
        if not device_equals(tensor.device, device):
            return False
    return True
