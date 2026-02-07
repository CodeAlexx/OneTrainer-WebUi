"""Block swap offloading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


def weights_to_device(module: nn.Module, device: torch.device) -> None:
    """Move module parameters and buffers to a device."""

    module.to(device=device)


@dataclass
class BlockSwapOffloader:
    block_type: str
    blocks: List[nn.Module]
    num_blocks: int
    blocks_to_swap: int
    device: torch.device

    def __post_init__(self) -> None:
        self._requested_blocks_to_swap = int(self.blocks_to_swap)
        self.blocks_to_swap = self._clamp_blocks(self.blocks_to_swap)

    def _clamp_blocks(self, value: int) -> int:
        max_swappable = max(self.num_blocks - 2, 0)
        return max(0, min(int(value), max_swappable))

    def disable_block_swap(self) -> None:
        self.blocks_to_swap = 0

    def enable_block_swap(self) -> None:
        self.blocks_to_swap = self._clamp_blocks(self._requested_blocks_to_swap)

    def prepare_block_devices_before_forward(self) -> None:
        if self.num_blocks <= 0:
            return

        if self.blocks_to_swap <= 0:
            for block in self.blocks:
                weights_to_device(block, self.device)
            return

        keep_on_device = max(self.num_blocks - self.blocks_to_swap, 0)
        cpu_device = torch.device("cpu")

        for idx, block in enumerate(self.blocks):
            target = self.device if idx < keep_on_device else cpu_device
            weights_to_device(block, target)
