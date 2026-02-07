"""Static memory allocator placeholders."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class StaticLayerAllocator:
    device: torch.device


@dataclass
class StaticActivationAllocator:
    device: torch.device
