"""Optimizer factory with support for standard and optional third-party optimizers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAM = "adam"
    SGD = "sgd"
    PRODIGY = "prodigy"
    ADAFACTOR = "adafactor"
    LION = "lion"

    @property
    def is_adaptive(self) -> bool:
        """Whether this optimizer adapts its own learning rate."""
        return self in {OptimizerType.PRODIGY}

    @property
    def requires_optional_dep(self) -> bool:
        """Whether this optimizer needs a third-party package."""
        return self in {
            OptimizerType.ADAMW_8BIT,
            OptimizerType.PRODIGY,
            OptimizerType.LION,
            OptimizerType.ADAFACTOR,
        }


def create_optimizer(
    optimizer_type: OptimizerType | str,
    parameters: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
    lr: float,
    *,
    # Adam / AdamW family
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    weight_decay: float | None = None,
    amsgrad: bool = False,
    fused: bool = False,
    foreach: bool = False,
    # SGD specific
    momentum: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    # bitsandbytes 8-bit specific
    min_8bit_size: int = 4096,
    percentile_clipping: float = 100.0,
    block_wise: bool = True,
    is_paged: bool = False,
    # Prodigy specific
    beta3: float | None = None,
    d0: float = 1e-6,
    d_coef: float = 1.0,
    growth_rate: float = float("inf"),
    decouple: bool = True,
    use_bias_correction: bool = False,
    safeguard_warmup: bool = False,
    slice_p: int = 1,
    # Adafactor specific
    eps2: float | None = None,
    clip_threshold: float = 1.0,
    decay_rate: float = -0.8,
    scale_parameter: bool = True,
    relative_step: bool = False,
    warmup_init: bool = False,
    # Lion specific
    use_triton: bool = False,
    # Catch-all for forward compatibility
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create an optimizer by type with sensible defaults matching OneTrainer parity.

    All optimizer-specific parameters are keyword-only.  Unknown kwargs are
    logged and silently dropped so callers can forward a superset of options.
    """
    if isinstance(optimizer_type, str):
        optimizer_type = OptimizerType(optimizer_type.lower())

    if kwargs:
        logger.debug("Ignoring unknown optimizer kwargs: %s", list(kwargs.keys()))

    match optimizer_type:

        # ------------------------------------------------------------------ #
        # AdamW (torch)
        # ------------------------------------------------------------------ #
        case OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params=parameters,
                lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                amsgrad=amsgrad,
                foreach=foreach,
                fused=fused,
            )

        # ------------------------------------------------------------------ #
        # Adam (torch)
        # ------------------------------------------------------------------ #
        case OptimizerType.ADAM:
            return torch.optim.Adam(
                params=parameters,
                lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 0.0,
                amsgrad=amsgrad,
                foreach=foreach,
                fused=fused,
            )

        # ------------------------------------------------------------------ #
        # SGD (torch)
        # ------------------------------------------------------------------ #
        case OptimizerType.SGD:
            return torch.optim.SGD(
                params=parameters,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay if weight_decay is not None else 0.0,
                nesterov=nesterov,
            )

        # ------------------------------------------------------------------ #
        # AdamW 8-bit (bitsandbytes)
        # ------------------------------------------------------------------ #
        case OptimizerType.ADAMW_8BIT:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for AdamW-8bit. "
                    "Install with: pip install bitsandbytes"
                ) from exc

            return bnb.optim.AdamW8bit(
                params=parameters,
                lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                min_8bit_size=min_8bit_size,
                percentile_clipping=percentile_clipping,
                block_wise=block_wise,
                is_paged=is_paged,
            )

        # ------------------------------------------------------------------ #
        # Prodigy (prodigyopt)
        # ------------------------------------------------------------------ #
        case OptimizerType.PRODIGY:
            try:
                import prodigyopt
            except ImportError as exc:
                raise ImportError(
                    "prodigyopt is required for Prodigy optimizer. "
                    "Install with: pip install prodigyopt"
                ) from exc

            return prodigyopt.Prodigy(
                params=parameters,
                lr=lr,
                betas=betas or (0.9, 0.999),
                beta3=beta3,
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 0.0,
                decouple=decouple,
                use_bias_correction=use_bias_correction,
                safeguard_warmup=safeguard_warmup,
                d0=d0,
                d_coef=d_coef,
                growth_rate=growth_rate,
                slice_p=slice_p,
            )

        # ------------------------------------------------------------------ #
        # Adafactor (transformers)
        # ------------------------------------------------------------------ #
        case OptimizerType.ADAFACTOR:
            try:
                from transformers.optimization import Adafactor
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for Adafactor optimizer. "
                    "Install with: pip install transformers"
                ) from exc

            # When using relative_step, Adafactor manages its own LR;
            # passing an explicit LR causes a warning.
            effective_lr = None if relative_step else lr

            # Handle per-group LR removal for relative_step mode,
            # matching OneTrainer's behavior.
            if relative_step and isinstance(parameters, list):
                params_list = list(parameters)
                for param in params_list:
                    if isinstance(param, dict) and "lr" in param:
                        param.pop("lr")
                parameters = params_list

            return Adafactor(
                params=parameters,
                lr=effective_lr,
                eps=(
                    eps if eps is not None else 1e-30,
                    eps2 if eps2 is not None else 1e-3,
                ),
                clip_threshold=clip_threshold,
                decay_rate=decay_rate,
                beta1=betas[0] if betas else None,
                weight_decay=weight_decay if weight_decay is not None else 0.0,
                scale_parameter=scale_parameter,
                relative_step=relative_step,
                warmup_init=warmup_init,
            )

        # ------------------------------------------------------------------ #
        # Lion (lion-pytorch)
        # ------------------------------------------------------------------ #
        case OptimizerType.LION:
            try:
                import lion_pytorch as lp
            except ImportError as exc:
                raise ImportError(
                    "lion-pytorch is required for Lion optimizer. "
                    "Install with: pip install lion-pytorch"
                ) from exc

            return lp.Lion(
                params=parameters,
                lr=lr,
                betas=betas or (0.9, 0.99),
                weight_decay=weight_decay if weight_decay is not None else 0.0,
                use_triton=use_triton,
            )

        case _:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


__all__ = [
    "OptimizerType",
    "create_optimizer",
]
