"""Optimizer and LR scheduler factory for native Flux 2 training."""

from __future__ import annotations

import math
from typing import Any

import torch

from serenity.cli.utils import (
    as_bool as _as_bool,
    first_config_value as _first_config_value,
    normalize_model_type as _normalize_model_type,
)

__all__ = [
    "_normalize_optimizer_name",
    "_normalize_scheduler_name",
    "_resolve_optimizer_steps",
    "_resolve_warmup_steps",
    "_resolve_scheduler_min_factor",
    "_resolve_scheduler_cycles",
    "_create_optimizer",
    "_create_lr_scheduler",
]

_OPTIMIZER_ALIASES = {
    "adamw": "adamw",
    "adamw_8bit": "adamw",
    "adamw8bit": "adamw",
    "paged_adamw_8bit": "adamw",
    "paged_adamw8bit": "adamw",
    "schedule_free_adamw": "adamw",
    "schedulefree_adamw": "adamw",
    "adam": "adam",
    "adam_8bit": "adam",
    "adam8bit": "adam",
    "paged_adam_8bit": "adam",
    "paged_adam8bit": "adam",
    "sgd": "sgd",
    "adafactor": "adafactor",
    "lion": "lion",
}

_CONSTANT_SCHEDULERS = {"", "none", "off", "constant", "constant_with_warmup", "adafactor"}
_LINEAR_SCHEDULERS = {"linear"}
_COSINE_SCHEDULERS = {"cosine"}
_COSINE_RESTART_SCHEDULERS = {"cosine_with_restarts", "cosine_with_hard_restarts", "cosine_restarts"}
_SCHEDULER_ALIASES = {
    "": "constant",
    "none": "constant",
    "off": "constant",
    "constant_with_warmup": "constant",
    "cosine_with_hard_restart": "cosine_with_hard_restarts",
    "cosine_restart": "cosine_with_restarts",
    "learning_rate_scheduler": "constant",
}


def _normalize_optimizer_name(value: Any, default: str = "adamw") -> str:
    normalized = _normalize_model_type(value or default)
    normalized = _OPTIMIZER_ALIASES.get(normalized, normalized)

    if normalized in _OPTIMIZER_ALIASES:
        return _OPTIMIZER_ALIASES[normalized]
    if "adafactor" in normalized:
        return "adafactor"
    if "lion" in normalized:
        return "lion"
    if normalized.startswith("sgd"):
        return "sgd"
    if normalized.startswith("adamw"):
        return "adamw"
    if normalized.startswith("adam"):
        return "adam"
    return default


def _normalize_scheduler_name(value: Any, default: str = "constant") -> str:
    normalized = _normalize_model_type(value or default)
    normalized = _SCHEDULER_ALIASES.get(normalized, normalized)
    if normalized in _CONSTANT_SCHEDULERS | _LINEAR_SCHEDULERS | _COSINE_SCHEDULERS | _COSINE_RESTART_SCHEDULERS:
        return normalized
    return default


def _resolve_optimizer_steps(max_steps: int, grad_accum: int) -> int:
    return max(1, math.ceil(float(max_steps) / float(max(1, grad_accum))))


def _resolve_warmup_steps(
    config: dict[str, Any],
    scheduler_block: dict[str, Any],
    total_optimizer_steps: int,
) -> int:
    warmup_raw = _first_config_value(
        scheduler_block,
        config,
        ("warmup_steps", "lr_warmup_steps", "learning_rate_warmup_steps"),
        0,
    )
    warmup_steps = int(float(warmup_raw or 0))
    return max(0, warmup_steps)


def _resolve_scheduler_min_factor(config: dict[str, Any], scheduler_block: dict[str, Any]) -> float:
    min_factor_raw = _first_config_value(
        scheduler_block,
        config,
        ("min_factor", "min_lr_factor", "lr_min_factor", "eta_min_ratio", "min_lr_ratio"),
        0.0,
    )
    min_factor = float(min_factor_raw or 0.0)
    return max(0.0, min(min_factor, 1.0))


def _resolve_scheduler_cycles(config: dict[str, Any], scheduler_block: dict[str, Any]) -> float:
    num_cycles_raw = _first_config_value(
        scheduler_block,
        config,
        ("num_cycles", "lr_num_cycles", "cosine_num_cycles"),
        1.0,
    )
    num_cycles = float(num_cycles_raw or 1.0)
    return max(1.0, num_cycles)


def _create_optimizer(
    params: list[torch.nn.Parameter],
    *,
    config: dict[str, Any],
    optimizer_block: dict[str, Any],
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name_raw = _first_config_value(
        optimizer_block,
        config,
        ("optimizer", "optimizer_type"),
        "adamw",
    )
    optimizer_name = _normalize_optimizer_name(optimizer_name_raw)

    weight_decay = float(_first_config_value(optimizer_block, config, ("weight_decay",), 0.0))
    beta1 = float(_first_config_value(optimizer_block, config, ("beta1",), 0.9))
    beta2 = float(_first_config_value(optimizer_block, config, ("beta2",), 0.999))
    eps = float(_first_config_value(optimizer_block, config, ("eps", "epsilon"), 1e-8))
    amsgrad = _as_bool(_first_config_value(optimizer_block, config, ("amsgrad",), False))

    if optimizer_name == "adafactor":
        from transformers import Adafactor

        clip_threshold_raw = _first_config_value(optimizer_block, config, ("clip_threshold",), None)
        adafactor_kwargs: dict[str, Any] = {
            "lr": learning_rate,
            "scale_parameter": _as_bool(_first_config_value(optimizer_block, config, ("scale_parameter",), False)),
            "relative_step": _as_bool(_first_config_value(optimizer_block, config, ("relative_step",), False)),
            "warmup_init": _as_bool(_first_config_value(optimizer_block, config, ("warmup_init",), False)),
            "weight_decay": weight_decay,
            "eps": (eps, 1e-3),
        }
        if clip_threshold_raw is not None:
            adafactor_kwargs["clip_threshold"] = float(clip_threshold_raw)
        return Adafactor(params, **adafactor_kwargs), optimizer_name

    if optimizer_name == "adam":
        return (
            torch.optim.Adam(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            ),
            optimizer_name,
        )

    if optimizer_name == "sgd":
        momentum = float(_first_config_value(optimizer_block, config, ("momentum",), 0.0))
        dampening = float(_first_config_value(optimizer_block, config, ("dampening",), 0.0))
        nesterov = _as_bool(_first_config_value(optimizer_block, config, ("nesterov",), False))
        if momentum <= 0.0:
            nesterov = False
        return (
            torch.optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
            ),
            optimizer_name,
        )

    if optimizer_name == "lion":
        lion_class = getattr(torch.optim, "Lion", None)
        if lion_class is None:
            try:
                from lion_pytorch import Lion as lion_class
            except ImportError:
                print("[native/flux2] warning: Lion optimizer unavailable, falling back to AdamW.")
                optimizer_name = "adamw"
                lion_class = None
        if lion_class is not None:
            return (
                lion_class(
                    params,
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                ),
                optimizer_name,
            )

    return (
        torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        ),
        "adamw",
    )


def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, Any],
    scheduler_block: dict[str, Any],
    total_optimizer_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, str]:
    scheduler_name_raw = _first_config_value(
        scheduler_block,
        config,
        ("scheduler", "lr_scheduler", "learning_rate_scheduler"),
        "constant",
    )
    scheduler_name = _normalize_scheduler_name(scheduler_name_raw)
    warmup_steps = _resolve_warmup_steps(config, scheduler_block, total_optimizer_steps)
    min_factor = _resolve_scheduler_min_factor(config, scheduler_block)
    num_cycles = _resolve_scheduler_cycles(config, scheduler_block)

    use_constant_schedule = scheduler_name in _CONSTANT_SCHEDULERS
    if use_constant_schedule and warmup_steps <= 0 and min_factor <= 0.0:
        return None, scheduler_name

    def _lr_lambda(last_epoch: int) -> float:
        step = max(0, int(last_epoch) + 1)

        if warmup_steps > 0 and step <= warmup_steps:
            warmup_progress = float(step) / float(max(1, warmup_steps))
            return max(min_factor, min(1.0, warmup_progress))

        if total_optimizer_steps <= warmup_steps:
            progress = 1.0
        else:
            progress = (float(step) - float(warmup_steps)) / float(max(1, total_optimizer_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_name in _LINEAR_SCHEDULERS:
            base_factor = 1.0 - progress
        elif scheduler_name in _COSINE_SCHEDULERS:
            base_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler_name in _COSINE_RESTART_SCHEDULERS:
            if progress >= 1.0:
                base_factor = 0.0
            else:
                cycle_position = (num_cycles * progress) % 1.0
                base_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_position))
        else:
            base_factor = 1.0

        factor = min_factor + (1.0 - min_factor) * base_factor
        return min(max(float(factor), min_factor), 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return scheduler, scheduler_name
