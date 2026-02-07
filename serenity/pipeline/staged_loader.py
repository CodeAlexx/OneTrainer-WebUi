"""Staged loading placeholder."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List


@dataclass
class StagedLoader:
    stages: List[Iterable]

    def __iter__(self) -> Iterator:
        for stage in self.stages:
            for item in stage:
                yield item
