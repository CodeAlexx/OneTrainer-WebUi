"""DataLoader helpers."""

from __future__ import annotations

from math import ceil
from typing import Iterator, List

from torch.utils.data import DataLoader, Sampler


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self._indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        for idx in self._indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return ceil(len(self._indices) / float(self.batch_size))


def create_dataloader(dataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
