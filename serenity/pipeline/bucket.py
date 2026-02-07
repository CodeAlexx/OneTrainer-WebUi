"""Bucket definitions for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bucket:
    """A resolution bucket for aspect-ratio-aware batching."""

    width: int
    height: int
    batch_size: int = 1
    samples: list[int] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return f"Bucket({self.width}x{self.height}, n={len(self.samples)})"


def generate_buckets(
    base_resolution: int = 512,
    min_dim: int = 256,
    max_dim: int = 1024,
    quantize: int = 64,
    target_pixels: int | None = None,
) -> list[Bucket]:
    """Generate a set of resolution buckets around a target pixel count.

    Args:
        base_resolution: The base square resolution (e.g., 512).
        min_dim: Minimum dimension (width or height).
        max_dim: Maximum dimension (width or height).
        quantize: Quantization step for dimensions (e.g., 64).
        target_pixels: Target pixel count. Defaults to base_resolution^2.

    Returns:
        List of Bucket objects covering various aspect ratios.
    """
    if target_pixels is None:
        target_pixels = base_resolution * base_resolution

    buckets: list[Bucket] = []
    seen: set[tuple[int, int]] = set()

    # Generate widths from min to max in quantize steps
    w = min_dim
    while w <= max_dim:
        # Calculate height to match target pixel count
        h_raw = target_pixels / w
        h = round(h_raw / quantize) * quantize
        h = max(min_dim, min(max_dim, h))

        key = (w, h)
        if key not in seen and h >= min_dim:
            seen.add(key)
            buckets.append(Bucket(width=w, height=h))
        w += quantize

    return sorted(buckets, key=lambda b: b.aspect_ratio)


__all__ = [
    "Bucket",
    "generate_buckets",
]
