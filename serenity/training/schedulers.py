"""Learning rate scheduler factory with warmup support."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class SchedulerType(str, Enum):
    """Supported LR scheduler types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_HARD_RESTARTS = "cosine_with_hard_restarts"
    REX = "rex"


# --------------------------------------------------------------------------- #
# Lambda functions (pure, stateless, following OneTrainer's lr_scheduler_util)
# --------------------------------------------------------------------------- #


def _apply_min_factor(value: float, min_factor: float) -> float:
    """Scale value into [min_factor, 1.0] range."""
    return min_factor + (1.0 - min_factor) * value


def _lr_lambda_constant() -> Callable[[int], float]:
    """Constant learning rate (multiplier always 1)."""
    def lr_lambda(current_step: int) -> float:
        return 1.0
    return lr_lambda


def _lr_lambda_linear(
    scheduler_steps: int,
    min_factor: float = 0.0,
) -> Callable[[int], float]:
    """Linear decay from 1.0 to min_factor over scheduler_steps."""
    def lr_lambda(current_step: int) -> float:
        if scheduler_steps <= 0:
            return 1.0
        lin_val = max(0.0, float(scheduler_steps - current_step) / float(scheduler_steps))
        return _apply_min_factor(lin_val, min_factor)
    return lr_lambda


def _lr_lambda_cosine(
    scheduler_steps: int,
    min_factor: float = 0.0,
) -> Callable[[int], float]:
    """Cosine decay from 1.0 to min_factor over scheduler_steps."""
    def lr_lambda(current_step: int) -> float:
        if scheduler_steps <= 0:
            return 1.0
        progress = float(current_step) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * math.pi))
        factor = max(0.0, cos_val)
        return _apply_min_factor(factor, min_factor)
    return lr_lambda


def _lr_lambda_cosine_with_restarts(
    scheduler_steps: int,
    num_cycles: float,
    min_factor: float = 0.0,
) -> Callable[[int], float]:
    """Cosine with smooth restarts (multiple periods)."""
    def lr_lambda(current_step: int) -> float:
        if scheduler_steps <= 0:
            return 1.0
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * 2.0 * math.pi * num_cycles))
        factor = max(0.0, cos_val)
        return _apply_min_factor(factor, min_factor)
    return lr_lambda


def _lr_lambda_cosine_with_hard_restarts(
    scheduler_steps: int,
    num_cycles: float,
    min_factor: float = 0.0,
) -> Callable[[int], float]:
    """Cosine with hard restarts (sawtooth-like)."""
    def lr_lambda(current_step: int) -> float:
        if scheduler_steps <= 0:
            return 1.0
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(((progress * num_cycles) % 1.0) * math.pi))
        factor = max(0.0, cos_val)
        return _apply_min_factor(factor, min_factor)
    return lr_lambda


def _lr_lambda_rex(
    scheduler_steps: int,
    min_factor: float = 0.0,
) -> Callable[[int], float]:
    """REX schedule (arxiv.org/abs/2107.04197)."""
    def lr_lambda(current_step: int) -> float:
        if scheduler_steps <= 0:
            return 1.0
        max_lr = 1.0
        min_lr = 0.0
        d = 0.9
        if current_step < scheduler_steps:
            progress = current_step / scheduler_steps
            div = (1 - d) + (d * (1 - progress))
            val = min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            val = min_lr
        return _apply_min_factor(val, min_factor)
    return lr_lambda


def _lr_lambda_warmup(
    warmup_steps: int,
    lr_lambda: Callable[[int], float],
) -> Callable[[int], float]:
    """Wrap any lr_lambda with a linear warmup prefix."""
    def warmup(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return lr_lambda(current_step - warmup_steps)
    return warmup


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def create_lr_scheduler(
    scheduler_type: SchedulerType | str,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    *,
    num_warmup_steps: int = 0,
    num_cycles: float = 1.0,
    min_lr_factor: float = 0.0,
    last_epoch: int = -1,
) -> LRScheduler:
    """Create a learning rate scheduler with optional warmup.

    Parity with OneTrainer's create_lr_scheduler / lr_scheduler_util.
    All schedules are implemented as LambdaLR for consistency.

    Args:
        scheduler_type: The schedule shape (constant, linear, cosine, etc.).
        optimizer: The optimizer whose LR groups will be scheduled.
        num_training_steps: Total training steps *after* warmup.
        num_warmup_steps: Number of linear warmup steps (0 = no warmup).
        num_cycles: Number of restart cycles (for restart schedules).
        min_lr_factor: Minimum LR multiplier floor (0.0 = decay to zero).
        last_epoch: Passed through to LambdaLR for resuming.

    Returns:
        A PyTorch LRScheduler instance.
    """
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type.lower())

    scheduler_steps = max(1, num_training_steps - num_warmup_steps)

    match scheduler_type:
        case SchedulerType.CONSTANT:
            lr_lambda = _lr_lambda_constant()

        case SchedulerType.LINEAR:
            lr_lambda = _lr_lambda_linear(scheduler_steps, min_lr_factor)

        case SchedulerType.COSINE:
            lr_lambda = _lr_lambda_cosine(scheduler_steps, min_lr_factor)

        case SchedulerType.COSINE_WITH_RESTARTS:
            lr_lambda = _lr_lambda_cosine_with_restarts(
                scheduler_steps, num_cycles, min_lr_factor,
            )

        case SchedulerType.COSINE_WITH_HARD_RESTARTS:
            lr_lambda = _lr_lambda_cosine_with_hard_restarts(
                scheduler_steps, num_cycles, min_lr_factor,
            )

        case SchedulerType.REX:
            lr_lambda = _lr_lambda_rex(scheduler_steps, min_lr_factor)

        case _:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # Apply warmup wrapper
    if num_warmup_steps > 0:
        lr_lambda = _lr_lambda_warmup(num_warmup_steps, lr_lambda)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=last_epoch,
    )


__all__ = [
    "SchedulerType",
    "create_lr_scheduler",
]
