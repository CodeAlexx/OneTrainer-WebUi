"""bf16 stochastic rounding for more precise optimizer updates.

Parity with OneTrainer's bf16_stochastic_rounding.py.
Stochastic rounding reduces the bias introduced by truncation when
converting float32 optimizer states to bfloat16.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Module-level generator for reproducible stochastic rounding
_generator: torch.Generator | None = None


def set_seed(seed: int, device: torch.device) -> None:
    """Set the RNG seed for stochastic rounding operations."""
    global _generator
    if _generator is None or _generator.device != device:
        _generator = torch.Generator(device=device)
    _generator.manual_seed(seed)


def copy_stochastic_(target: Tensor, source: Tensor) -> None:
    """Copy source (float32) into target (bfloat16) using stochastic rounding.

    Instead of deterministic truncation, adds a random value to the lower
    16 bits of the mantissa before truncating. This gives an unbiased
    estimate of the full-precision value in expectation.
    """
    global _generator

    # Create a random 16-bit integer
    result = torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=_generator,
    )

    # Add the random number to the lower 16 bits of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # Mask off the lower 16 bits of the mantissa
    # -65536 = 0xFFFF0000 as a signed int32
    result.bitwise_and_(-65536)

    # Copy the higher 16 bits into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0) -> None:
    """Add ``other`` to ``input`` using stochastic rounding.

    ``input`` should be bfloat16; ``other`` can be any floating type.
    The computation is done in float32 then rounded stochastically.
    """
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)
    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)


def addcdiv_stochastic_(
    input: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    value: float = 1.0,
) -> None:
    """Compute ``input + value * (tensor1 / tensor2)`` with stochastic rounding.

    ``input`` should be bfloat16. The operation is performed in float32
    then rounded stochastically back to bfloat16.
    """
    result = input.clone() if input.dtype == torch.float32 else input.to(dtype=torch.float32)
    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(input, result)


__all__ = [
    "set_seed",
    "copy_stochastic_",
    "add_stochastic_",
    "addcdiv_stochastic_",
]
