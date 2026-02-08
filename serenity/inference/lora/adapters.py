"""Weight adapter types for advanced LoRA variants."""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

__all__ = [
    "WeightAdapterBase",
    "LoRAAdapter",
    "OFTAdapter",
    "LoHAAdapter",
    "LoKRAdapter",
    "detect_adapter_type",
    "create_adapter",
]

logger = logging.getLogger(__name__)


class WeightAdapterBase:
    """Base class for weight adaptation methods."""

    def apply(self, weight: Tensor, strength: float = 1.0) -> Tensor:
        """Apply adaptation to weight tensor."""
        raise NotImplementedError

    @staticmethod
    def detect(lora_data: dict[str, Tensor]) -> bool:
        """Check if lora_data matches this adapter type."""
        return False


class LoRAAdapter(WeightAdapterBase):
    """Standard LoRA: weight + strength * (alpha/rank) * (up @ down)."""

    def __init__(self, up: Tensor, down: Tensor, alpha: float | None = None) -> None:
        self.up = up
        self.down = down
        self.alpha = alpha

    def apply(self, weight: Tensor, strength: float = 1.0) -> Tensor:
        rank = self.down.shape[0]
        scale = strength * (self.alpha / rank if self.alpha else 1.0)
        delta = self.up.float() @ self.down.float()
        if delta.shape != weight.shape:
            delta = delta.reshape(weight.shape)
        return weight + scale * delta.to(weight.dtype)

    @staticmethod
    def detect(lora_data: dict[str, Tensor]) -> bool:
        keys = set(lora_data.keys())
        return any("lora_up" in k for k in keys) and any("lora_down" in k for k in keys)


class OFTAdapter(WeightAdapterBase):
    """Orthogonal Finetuning: applies orthogonal rotation to weight matrix.

    Preserves singular values (magnitude) while adjusting directions.
    weight_out = weight @ (I + R)^(-1) @ (I + R) where R is block-diagonal
    skew-symmetric.
    """

    def __init__(self, blocks: Tensor, alpha: float | None = None) -> None:
        self.blocks = blocks  # (num_blocks, block_size, block_size)
        self.alpha = alpha

    def apply(self, weight: Tensor, strength: float = 1.0) -> Tensor:
        blocks = self.blocks.float()
        num_blocks, block_size, _ = blocks.shape

        # Construct block-diagonal orthogonal matrix via Cayley transform
        # R = (I - B)(I + B)^(-1) where B is skew-symmetric
        eye = torch.eye(block_size, device=blocks.device, dtype=blocks.dtype)
        # Make blocks skew-symmetric
        skew = blocks - blocks.transpose(-1, -2)

        # Cayley transform per block
        orth_blocks = torch.linalg.solve(eye + skew, eye - skew)  # (num_blocks, bs, bs)

        # Assemble into full rotation matrix
        total_size = num_blocks * block_size
        rotation = torch.block_diag(*[orth_blocks[i] for i in range(num_blocks)])

        # Apply with strength interpolation
        eye_full = torch.eye(total_size, device=weight.device, dtype=torch.float32)
        rotation = eye_full + strength * (rotation - eye_full)

        # Reshape weight and apply rotation
        orig_shape = weight.shape
        w = weight.float().reshape(weight.shape[0], -1)
        if w.shape[1] == total_size:
            w = w @ rotation.to(w.device)
        elif w.shape[0] == total_size:
            w = rotation.to(w.device).T @ w

        return w.reshape(orig_shape).to(weight.dtype)

    @staticmethod
    def detect(lora_data: dict[str, Tensor]) -> bool:
        return any("oft_diag" in k or "oft_blocks" in k for k in lora_data)


class LoHAAdapter(WeightAdapterBase):
    """Low-rank Hadamard adaptation: delta = (up1 @ down1) * (up2 @ down2).

    Uses Hadamard (element-wise) product of two low-rank decompositions.
    """

    def __init__(
        self,
        up1: Tensor,
        down1: Tensor,
        up2: Tensor,
        down2: Tensor,
        alpha: float | None = None,
    ) -> None:
        self.up1, self.down1 = up1, down1
        self.up2, self.down2 = up2, down2
        self.alpha = alpha

    def apply(self, weight: Tensor, strength: float = 1.0) -> Tensor:
        rank = self.down1.shape[0]
        scale = strength * (self.alpha / rank if self.alpha else 1.0)

        part1 = self.up1.float() @ self.down1.float()
        part2 = self.up2.float() @ self.down2.float()
        delta = part1 * part2  # Hadamard product

        if delta.shape != weight.shape:
            delta = delta.reshape(weight.shape)
        return weight + scale * delta.to(weight.dtype)

    @staticmethod
    def detect(lora_data: dict[str, Tensor]) -> bool:
        keys = set(lora_data.keys())
        return any("hada_w1_a" in k for k in keys) and any("hada_w1_b" in k for k in keys)


class LoKRAdapter(WeightAdapterBase):
    """Low-Kronecker adaptation: delta = kron(A, B).

    Uses Kronecker product structure for parameter-efficient adaptation.
    """

    def __init__(
        self,
        w1: Tensor | None = None,
        w2: Tensor | None = None,
        w1_a: Tensor | None = None,
        w1_b: Tensor | None = None,
        w2_a: Tensor | None = None,
        w2_b: Tensor | None = None,
        alpha: float | None = None,
    ) -> None:
        self.w1, self.w2 = w1, w2
        self.w1_a, self.w1_b = w1_a, w1_b
        self.w2_a, self.w2_b = w2_a, w2_b
        self.alpha = alpha

    def apply(self, weight: Tensor, strength: float = 1.0) -> Tensor:
        # Reconstruct w1 and w2 from low-rank if provided
        if self.w1 is not None:
            w1 = self.w1.float()
        elif self.w1_a is not None and self.w1_b is not None:
            w1 = self.w1_a.float() @ self.w1_b.float()
        else:
            raise ValueError("LoKR: need either w1 or w1_a+w1_b")

        if self.w2 is not None:
            w2 = self.w2.float()
        elif self.w2_a is not None and self.w2_b is not None:
            w2 = self.w2_a.float() @ self.w2_b.float()
        else:
            raise ValueError("LoKR: need either w2 or w2_a+w2_b")

        delta = torch.kron(w1, w2)

        rank = max(w1.shape[0], w2.shape[0], 1)
        scale = strength * (self.alpha / rank if self.alpha else 1.0)

        if delta.shape != weight.shape:
            delta = delta.reshape(weight.shape)
        return weight + scale * delta.to(weight.dtype)

    @staticmethod
    def detect(lora_data: dict[str, Tensor]) -> bool:
        keys = set(lora_data.keys())
        return any("lokr_w1" in k for k in keys) or any("lokr_w2" in k for k in keys)


def detect_adapter_type(lora_data: dict[str, Tensor]) -> str:
    """Detect the adapter type from LoRA file keys.

    Returns one of: "lora", "oft", "loha", "lokr", "unknown".
    """
    if OFTAdapter.detect(lora_data):
        return "oft"
    if LoHAAdapter.detect(lora_data):
        return "loha"
    if LoKRAdapter.detect(lora_data):
        return "lokr"
    if LoRAAdapter.detect(lora_data):
        return "lora"
    return "unknown"


def create_adapter(
    adapter_type: str,
    tensors: dict[str, Tensor],
    alpha: float | None = None,
) -> WeightAdapterBase:
    """Create a weight adapter from loaded tensors.

    Args:
        adapter_type: One of "lora", "oft", "loha", "lokr".
        tensors: Dictionary of tensor names to tensor values for this layer.
        alpha: Optional alpha scaling value.
    """
    if adapter_type == "lora":
        up = tensors.get("lora_up.weight")
        if up is None:
            up = tensors.get("lora_A.weight")
        down = tensors.get("lora_down.weight")
        if down is None:
            down = tensors.get("lora_B.weight")
        if up is None or down is None:
            raise ValueError(
                f"LoRA adapter missing up/down weights. Keys: {list(tensors.keys())}"
            )
        return LoRAAdapter(up, down, alpha)

    if adapter_type == "oft":
        blocks = tensors.get("oft_diag")
        if blocks is None:
            blocks = tensors.get("oft_blocks")
        if blocks is None:
            raise ValueError(
                f"OFT adapter missing blocks. Keys: {list(tensors.keys())}"
            )
        if blocks.ndim == 2:
            # Reshape (num_blocks, block_size*block_size) -> (num_blocks, block_size, block_size)
            num_blocks = blocks.shape[0]
            block_size = int(math.isqrt(blocks.shape[1]))
            blocks = blocks.reshape(num_blocks, block_size, block_size)
        return OFTAdapter(blocks, alpha)

    if adapter_type == "loha":
        up1 = tensors.get("hada_w1_a")
        down1 = tensors.get("hada_w1_b")
        up2 = tensors.get("hada_w2_a")
        down2 = tensors.get("hada_w2_b")
        if up1 is None or down1 is None or up2 is None or down2 is None:
            raise ValueError(
                f"LoHA adapter missing weights. Keys: {list(tensors.keys())}"
            )
        return LoHAAdapter(up1, down1, up2, down2, alpha)

    if adapter_type == "lokr":
        return LoKRAdapter(
            w1=tensors.get("lokr_w1"),
            w2=tensors.get("lokr_w2"),
            w1_a=tensors.get("lokr_w1_a"),
            w1_b=tensors.get("lokr_w1_b"),
            w2_a=tensors.get("lokr_w2_a"),
            w2_b=tensors.get("lokr_w2_b"),
            alpha=alpha,
        )

    raise ValueError(f"Unknown adapter type: {adapter_type}")
