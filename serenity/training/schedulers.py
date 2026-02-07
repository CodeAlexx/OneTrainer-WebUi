"""Learning rate scheduler factory with warmup support.

Parity with OneTrainer's create_lr_scheduler / lr_scheduler_util.
Includes cosine with restarts, hard restarts, REX, Adafactor, and custom.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from enum import Enum

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR


class SchedulerType(str, Enum):
    """Supported LR scheduler types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_HARD_RESTARTS = "cosine_with_hard_restarts"
    REX = "rex"
    ADAFACTOR = "adafactor"
    CUSTOM = "custom"


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
    # Adafactor-specific
    initial_lr: float | None = None,
    # Custom scheduler
    custom_class: str | None = None,
    custom_params: list[dict[str, str]] | None = None,
    learning_rate: float | None = None,
    num_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    gradient_accumulation_steps: int = 1,
) -> LRScheduler:
    """Create a learning rate scheduler with optional warmup.

    Parity with OneTrainer's create_lr_scheduler / lr_scheduler_util.
    All schedules are implemented as LambdaLR for consistency.
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

        case SchedulerType.ADAFACTOR:
            try:
                from transformers.optimization import AdafactorSchedule
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for AdafactorSchedule. "
                    "Install with: pip install transformers"
                ) from exc
            eff_lr = initial_lr
            if eff_lr is None:
                # Fall back to optimizer's param group initial_lr
                eff_lr = optimizer.state_dict()["param_groups"][0].get("initial_lr", 1e-3)
            return AdafactorSchedule(optimizer, initial_lr=eff_lr)

        case SchedulerType.CUSTOM:
            if not custom_class:
                raise ValueError("Must specify custom_class when using CUSTOM scheduler.")
            if "." not in custom_class:
                raise ValueError("custom_class must be in format <module>.<ClassName>")

            klass_name = custom_class.split(".")[-1]
            module_name = custom_class.removesuffix("." + klass_name)
            mod = importlib.import_module(module_name)
            klass = getattr(mod, klass_name)

            # Build kwargs from custom_params
            custom_kwargs: dict[str, object] = {}
            total_steps = num_training_steps
            for pd in (custom_params or []):
                key = pd["key"]
                value = pd["value"]
                # Special value substitutions
                if value == "%LR%":
                    value = learning_rate or 1e-4
                elif value == "%EPOCHS%":
                    value = num_epochs or 1
                elif value == "%STEPS_PER_EPOCH%":
                    value = steps_per_epoch or total_steps
                elif value == "%TOTAL_STEPS%":
                    value = total_steps
                elif value == "%SCHEDULER_STEPS%":
                    value = scheduler_steps
                else:
                    import ast
                    value = ast.literal_eval(value)
                custom_kwargs[key] = value

            scheduler = klass(optimizer=optimizer, last_epoch=last_epoch, **custom_kwargs)

            if num_warmup_steps > 0:
                warmup_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=_lr_lambda_warmup(num_warmup_steps, _lr_lambda_constant()),
                    last_epoch=last_epoch,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[num_warmup_steps],
                    last_epoch=last_epoch,
                )
            return scheduler

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
