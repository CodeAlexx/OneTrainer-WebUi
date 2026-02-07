"""Cache manager stub for staged loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheManager:
    root: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
