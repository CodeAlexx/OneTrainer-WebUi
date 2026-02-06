"""Base model implementation used by concrete models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from eritrainer.core.interfaces import ModelType


@dataclass
class BaseModelImpl:
    model_type: ModelType

    def to(self, device):  # pragma: no cover - trivial
        return self

    def train(self):  # pragma: no cover - trivial
        return self

    def eval(self):  # pragma: no cover - trivial
        return self

    def parameters(self) -> Iterable:  # pragma: no cover - trivial
        return []
