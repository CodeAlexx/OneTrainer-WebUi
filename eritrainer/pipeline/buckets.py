"""Bucket manager for grouping samples by resolution."""

from __future__ import annotations

from typing import List, Optional

from eritrainer.pipeline.bucket import Bucket


class BucketManager:
    def __init__(self, buckets: Optional[List[Bucket]] = None) -> None:
        self.buckets: List[Bucket] = list(buckets) if buckets else []

    def add_bucket(self, bucket: Bucket) -> None:
        self.buckets.append(bucket)

    def select_bucket(self, width: int, height: int) -> Optional[Bucket]:
        for bucket in self.buckets:
            if bucket.width == width and bucket.height == height:
                return bucket
        return None
