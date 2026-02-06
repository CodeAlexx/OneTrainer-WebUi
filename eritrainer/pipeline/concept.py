"""Concept definitions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Concept:
    name: str
    path: Path
    caption_file: Optional[Path] = None
