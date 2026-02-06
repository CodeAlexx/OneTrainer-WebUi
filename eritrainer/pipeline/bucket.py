"""Bucket definitions for the data pipeline."""

from dataclasses import dataclass


@dataclass
class Bucket:
    width: int
    height: int
    batch_size: int
