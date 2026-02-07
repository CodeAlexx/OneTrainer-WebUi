"""Concept scanning and caption utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

from serenity.pipeline.concept import Concept


class ConceptScanner:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def scan(self) -> List[Concept]:
        if not self.root.exists():
            return []
        concepts: List[Concept] = []
        for child in self.root.iterdir():
            if child.is_dir():
                caption_file = child / "captions.txt"
                concepts.append(Concept(name=child.name, path=child, caption_file=caption_file))
        return concepts


class CaptionLoader:
    def __init__(self, caption_file: Path) -> None:
        self.caption_file = Path(caption_file)

    def load(self) -> List[str]:
        if not self.caption_file.exists():
            return []
        return [line.strip() for line in self.caption_file.read_text().splitlines() if line.strip()]
