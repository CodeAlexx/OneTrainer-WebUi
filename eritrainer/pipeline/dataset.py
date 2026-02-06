"""Dataset abstractions."""

from __future__ import annotations

from typing import Any, List

import torch
from torch.utils.data import Dataset


class EriDataset(Dataset):
    """Simple dataset wrapper for pipeline tests."""

    def __init__(self, items: List[Any] | None = None) -> None:
        self.items = items or []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, idx: int) -> Any:  # pragma: no cover - trivial
        return self.items[idx]
