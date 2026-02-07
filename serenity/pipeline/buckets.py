"""Bucket manager for grouping samples by resolution."""

from __future__ import annotations

import math
import random
from typing import Iterator

from serenity.pipeline.bucket import Bucket, generate_buckets


class BucketManager:
    """Manage resolution buckets and assign samples to the nearest bucket."""

    def __init__(self, buckets: list[Bucket] | None = None) -> None:
        self.buckets: list[Bucket] = list(buckets) if buckets else []

    @classmethod
    def from_config(
        cls,
        base_resolution: int = 512,
        min_dim: int = 256,
        max_dim: int = 1024,
        quantize: int = 64,
    ) -> BucketManager:
        """Create a BucketManager with auto-generated resolution buckets."""
        buckets = generate_buckets(
            base_resolution=base_resolution,
            min_dim=min_dim,
            max_dim=max_dim,
            quantize=quantize,
        )
        return cls(buckets)

    def add_bucket(self, bucket: Bucket) -> None:
        self.buckets.append(bucket)

    def select_bucket(self, width: int, height: int) -> Bucket | None:
        """Find the nearest bucket for the given dimensions.

        Uses minimum aspect ratio distance to find the best match.
        """
        if not self.buckets:
            return None

        target_ar = width / height if height > 0 else 1.0
        best: Bucket | None = None
        best_dist = float("inf")

        for bucket in self.buckets:
            dist = abs(bucket.aspect_ratio - target_ar)
            if dist < best_dist:
                best_dist = dist
                best = bucket

        return best

    def assign_sample(self, sample_idx: int, width: int, height: int) -> Bucket | None:
        """Assign a sample index to the nearest bucket.

        Returns the bucket it was assigned to, or None if no buckets exist.
        """
        bucket = self.select_bucket(width, height)
        if bucket is not None:
            bucket.samples.append(sample_idx)
        return bucket

    def assign_samples(
        self,
        sizes: list[tuple[int, int]],
    ) -> None:
        """Batch-assign samples given a list of (width, height) tuples."""
        for idx, (w, h) in enumerate(sizes):
            self.assign_sample(idx, w, h)

    def get_nonempty_buckets(self) -> list[Bucket]:
        """Return only buckets that have at least one sample."""
        return [b for b in self.buckets if b.samples]

    def total_samples(self) -> int:
        return sum(len(b.samples) for b in self.buckets)

    def clear(self) -> None:
        """Remove all sample assignments from all buckets."""
        for bucket in self.buckets:
            bucket.samples.clear()


class BucketBatchSampler:
    """Batch sampler that yields batches of same-resolution samples.

    Samples within each bucket are shuffled, then batches are formed per bucket.
    The order of batches across buckets is also shuffled.
    """

    def __init__(
        self,
        bucket_manager: BucketManager,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.bucket_manager = bucket_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        all_batches: list[list[int]] = []

        for bucket in self.bucket_manager.get_nonempty_buckets():
            indices = list(bucket.samples)
            if self.shuffle:
                random.shuffle(indices)

            bs = bucket.batch_size if bucket.batch_size > 0 else self.batch_size

            for i in range(0, len(indices), bs):
                batch = indices[i : i + bs]
                if self.drop_last and len(batch) < bs:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for bucket in self.bucket_manager.get_nonempty_buckets():
            bs = bucket.batch_size if bucket.batch_size > 0 else self.batch_size
            n = len(bucket.samples)
            if self.drop_last:
                total += n // bs
            else:
                total += math.ceil(n / bs)
        return total


__all__ = [
    "BucketManager",
    "BucketBatchSampler",
]
