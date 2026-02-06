"""Torch utility helpers used by training code."""

from __future__ import annotations

import gc
from typing import Optional

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
