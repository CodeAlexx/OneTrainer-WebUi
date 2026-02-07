"""Optimizer factory with support for standard and optional third-party optimizers.

Parity with OneTrainer's create_optimizer() - all 35+ optimizer types.
Each third-party optimizer uses conditional imports (try/except).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Supported optimizer types matching OneTrainer's Optimizer enum."""

    # Torch built-in
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

    # BNB standard & 8-bit
    ADAM_8BIT = "adam_8bit"
    ADAMW_8BIT = "adamw_8bit"
    SGD_8BIT = "sgd_8bit"
    ADAGRAD = "adagrad"
    ADAGRAD_8BIT = "adagrad_8bit"
    RMSPROP = "rmsprop"
    RMSPROP_8BIT = "rmsprop_8bit"
    LION = "lion"
    LION_8BIT = "lion_8bit"
    LAMB = "lamb"
    LAMB_8BIT = "lamb_8bit"
    LARS = "lars"
    LARS_8BIT = "lars_8bit"
    ADEMAMIX = "ademamix"
    ADEMAMIX_8BIT = "ademamix_8bit"

    # Schedule-free
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"

    # DADAPT
    DADAPT_SGD = "dadapt_sgd"
    DADAPT_ADAM = "dadapt_adam"
    DADAPT_ADAN = "dadapt_adan"
    DADAPT_ADA_GRAD = "dadapt_ada_grad"
    DADAPT_LION = "dadapt_lion"

    # Prodigy family
    PRODIGY = "prodigy"
    PRODIGY_PLUS_SCHEDULE_FREE = "prodigy_plus_schedule_free"

    # Adafactor
    ADAFACTOR = "adafactor"

    # Advanced optimizers (adv_optm)
    ADAMW_ADV = "adamw_adv"
    ADOPT_ADV = "adopt_adv"
    PRODIGY_ADV = "prodigy_adv"
    LION_ADV = "lion_adv"
    LION_PRODIGY_ADV = "lion_prodigy_adv"
    SIMPLIFIED_ADEMAMIX = "simplified_ademamix"
    SIGNSGD_ADV = "signsgd_adv"
    MUON_ADV = "muon_adv"
    ADAMUON_ADV = "adamuon_adv"

    # CAME
    CAME = "came"
    CAME_8BIT = "came_8bit"

    # MUON (upstream)
    MUON = "muon"

    # Timm / pytorch-optimizer
    ADABELIEF = "adabelief"
    TIGER = "tiger"
    AIDA = "aida"
    ADOPT = "adopt"
    YOGI = "yogi"

    @property
    def is_adaptive(self) -> bool:
        """Whether this optimizer adapts its own learning rate."""
        return self in {
            OptimizerType.DADAPT_SGD,
            OptimizerType.DADAPT_ADAM,
            OptimizerType.DADAPT_ADAN,
            OptimizerType.DADAPT_ADA_GRAD,
            OptimizerType.DADAPT_LION,
            OptimizerType.PRODIGY,
            OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
            OptimizerType.PRODIGY_ADV,
            OptimizerType.LION_PRODIGY_ADV,
        }

    @property
    def is_schedule_free(self) -> bool:
        """Whether this optimizer is schedule-free."""
        return self in {
            OptimizerType.SCHEDULE_FREE_ADAMW,
            OptimizerType.SCHEDULE_FREE_SGD,
            OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
        }

    @property
    def requires_optional_dep(self) -> bool:
        """Whether this optimizer needs a third-party package."""
        return self not in {OptimizerType.ADAM, OptimizerType.ADAMW, OptimizerType.SGD}


def _bnb_import():
    try:
        import bitsandbytes as bnb
        return bnb
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required. Install with: pip install bitsandbytes"
        ) from exc


def create_optimizer(
    optimizer_type: OptimizerType | str,
    parameters: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
    lr: float,
    *,
    # Adam / AdamW family
    betas: tuple[float, ...] | None = None,
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
    optim_bits: int = 32,
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
    # AdEMAMix specific
    alpha: float | None = None,
    # Schedule-free specific
    r: float = 0.0,
    weight_lr_power: float = 2.0,
    warmup_steps: int = 0,
    # DADAPT specific
    log_every: int = 0,
    fsdp_in_use: bool = False,
    no_prox: bool = False,
    # LAMB specific
    bias_correction: bool = True,
    adam_w_mode: bool = True,
    max_unorm: float | None = None,
    # RMSprop specific
    centered: bool = False,
    rmsprop_alpha: float = 0.99,
    # Adagrad specific
    lr_decay: float = 0.0,
    initial_accumulator_value: float = 0.0,
    # Advanced optimizer flags
    stochastic_rounding: bool = False,
    fused_back_pass: bool = False,
    # timm/pytorch-optimizer
    decoupled_decay: bool = True,
    fixed_decay: bool = False,
    rectify: bool = True,
    degenerated_to_sgd: bool = True,
    # Catch-all for forward compatibility
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create an optimizer by type with sensible defaults matching OneTrainer parity.

    All optimizer-specific parameters are keyword-only. Unknown kwargs are
    logged and silently dropped so callers can forward a superset of options.
    """
    if isinstance(optimizer_type, str):
        optimizer_type = OptimizerType(optimizer_type.lower())

    if kwargs:
        logger.debug("Ignoring unknown optimizer kwargs: %s", list(kwargs.keys()))

    wd = weight_decay if weight_decay is not None else 0.0

    match optimizer_type:

        # ------------------------------------------------------------------ #
        # Torch built-in
        # ------------------------------------------------------------------ #

        case OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                amsgrad=amsgrad, foreach=foreach, fused=fused,
            )

        case OptimizerType.ADAM:
            return torch.optim.Adam(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd, amsgrad=amsgrad, foreach=foreach, fused=fused,
            )

        case OptimizerType.SGD:
            return torch.optim.SGD(
                params=parameters, lr=lr,
                momentum=momentum, dampening=dampening,
                weight_decay=wd, nesterov=nesterov,
            )

        # ------------------------------------------------------------------ #
        # BNB 8-bit
        # ------------------------------------------------------------------ #

        case OptimizerType.ADAM_8BIT:
            bnb = _bnb_import()
            return bnb.optim.Adam(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8, weight_decay=wd,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=block_wise, is_paged=is_paged,
            )

        case OptimizerType.ADAMW_8BIT:
            bnb = _bnb_import()
            return bnb.optim.AdamW8bit(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=block_wise, is_paged=is_paged,
            )

        case OptimizerType.SGD_8BIT:
            bnb = _bnb_import()
            return bnb.optim.SGD8bit(
                params=parameters, lr=lr,
                momentum=momentum, dampening=dampening,
                weight_decay=wd, nesterov=nesterov,
            )

        case OptimizerType.ADAGRAD:
            bnb = _bnb_import()
            return bnb.optim.Adagrad(
                params=parameters, lr=lr, weight_decay=wd,
                eps=eps if eps is not None else 1e-10,
                lr_decay=lr_decay,
                initial_accumulator_value=initial_accumulator_value,
            )

        case OptimizerType.ADAGRAD_8BIT:
            bnb = _bnb_import()
            return bnb.optim.Adagrad8bit(
                params=parameters, lr=lr, weight_decay=wd,
                eps=eps if eps is not None else 1e-10,
                lr_decay=lr_decay,
                initial_accumulator_value=initial_accumulator_value,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=block_wise,
            )

        case OptimizerType.RMSPROP:
            bnb = _bnb_import()
            return bnb.optim.RMSprop(
                params=parameters, lr=lr, weight_decay=wd,
                eps=eps if eps is not None else 1e-8,
                alpha=rmsprop_alpha, momentum=momentum, centered=centered,
            )

        case OptimizerType.RMSPROP_8BIT:
            bnb = _bnb_import()
            return bnb.optim.RMSprop8bit(
                params=parameters, lr=lr, weight_decay=wd,
                eps=eps if eps is not None else 1e-8,
                alpha=rmsprop_alpha, momentum=momentum, centered=centered,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=block_wise,
            )

        case OptimizerType.LION:
            try:
                import lion_pytorch as lp
            except ImportError as exc:
                raise ImportError("lion-pytorch is required. pip install lion-pytorch") from exc
            return lp.Lion(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.99),
                weight_decay=wd, use_triton=use_triton,
            )

        case OptimizerType.LION_8BIT:
            bnb = _bnb_import()
            return bnb.optim.Lion8bit(
                params=parameters, lr=lr, weight_decay=wd,
                betas=betas or (0.9, 0.999),
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=block_wise, is_paged=is_paged,
            )

        case OptimizerType.LAMB:
            bnb = _bnb_import()
            return bnb.optim.LAMB(
                params=parameters, lr=lr,
                bias_correction=bias_correction,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd, amsgrad=amsgrad,
                adam_w_mode=adam_w_mode, optim_bits=optim_bits,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=False,
                max_unorm=max_unorm if max_unorm is not None else 1.0,
            )

        case OptimizerType.LAMB_8BIT:
            bnb = _bnb_import()
            return bnb.optim.LAMB8bit(
                params=parameters, lr=lr,
                bias_correction=bias_correction,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd, amsgrad=amsgrad,
                adam_w_mode=adam_w_mode,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                block_wise=False,
                max_unorm=max_unorm if max_unorm is not None else 1.0,
            )

        case OptimizerType.LARS:
            bnb = _bnb_import()
            return bnb.optim.LARS(
                params=parameters, lr=lr, weight_decay=wd,
                momentum=momentum, dampening=dampening, nesterov=nesterov,
                max_unorm=max_unorm if max_unorm is not None else 0.02,
            )

        case OptimizerType.LARS_8BIT:
            bnb = _bnb_import()
            return bnb.optim.LARS8bit(
                params=parameters, lr=lr, weight_decay=wd,
                momentum=momentum, dampening=dampening, nesterov=nesterov,
                min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
                max_unorm=max_unorm if max_unorm is not None else 0.02,
            )

        case OptimizerType.ADEMAMIX:
            bnb = _bnb_import()
            return bnb.optim.AdEMAMix(
                params=parameters, lr=lr,
                betas=(
                    betas[0] if betas and len(betas) > 0 else 0.9,
                    betas[1] if betas and len(betas) > 1 else 0.999,
                    beta3 if beta3 is not None else 0.9999,
                ),
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                eps=eps if eps is not None else 1e-8,
                alpha=alpha if alpha is not None else 5,
                optim_bits=optim_bits,
                min_8bit_size=min_8bit_size, is_paged=is_paged,
            )

        case OptimizerType.ADEMAMIX_8BIT:
            bnb = _bnb_import()
            return bnb.optim.AdEMAMix8bit(
                params=parameters, lr=lr,
                betas=(
                    betas[0] if betas and len(betas) > 0 else 0.9,
                    betas[1] if betas and len(betas) > 1 else 0.999,
                    beta3 if beta3 is not None else 0.9999,
                ),
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                eps=eps if eps is not None else 1e-8,
                alpha=alpha if alpha is not None else 5,
                min_8bit_size=min_8bit_size, is_paged=is_paged,
            )

        # ------------------------------------------------------------------ #
        # Schedule-free
        # ------------------------------------------------------------------ #

        case OptimizerType.SCHEDULE_FREE_ADAMW:
            try:
                from schedulefree import AdamWScheduleFree
            except ImportError as exc:
                raise ImportError("schedulefree is required. pip install schedulefree") from exc
            return AdamWScheduleFree(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                weight_decay=weight_decay if weight_decay is not None else 1e-2,
                eps=eps if eps is not None else 1e-8,
                warmup_steps=warmup_steps,
                r=r, weight_lr_power=weight_lr_power,
                foreach=foreach,
            )

        case OptimizerType.SCHEDULE_FREE_SGD:
            try:
                from schedulefree import SGDScheduleFree
            except ImportError as exc:
                raise ImportError("schedulefree is required. pip install schedulefree") from exc
            return SGDScheduleFree(
                params=parameters, lr=lr,
                momentum=momentum, weight_decay=wd,
                warmup_steps=warmup_steps,
                r=r, weight_lr_power=weight_lr_power,
                foreach=foreach,
            )

        # ------------------------------------------------------------------ #
        # DADAPT
        # ------------------------------------------------------------------ #

        case OptimizerType.DADAPT_SGD:
            try:
                import dadaptation as da
            except ImportError as exc:
                raise ImportError("dadaptation is required. pip install dadaptation") from exc
            return da.DAdaptSGD(
                params=parameters, lr=lr,
                momentum=momentum, weight_decay=wd,
                log_every=log_every, d0=d0, growth_rate=growth_rate,
                fsdp_in_use=fsdp_in_use,
            )

        case OptimizerType.DADAPT_ADAM:
            try:
                import dadaptation as da
            except ImportError as exc:
                raise ImportError("dadaptation is required. pip install dadaptation") from exc
            return da.DAdaptAdam(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd, log_every=log_every,
                decouple=decouple, use_bias_correction=use_bias_correction,
                d0=d0, growth_rate=growth_rate, fsdp_in_use=fsdp_in_use,
            )

        case OptimizerType.DADAPT_ADAN:
            try:
                import dadaptation as da
            except ImportError as exc:
                raise ImportError("dadaptation is required. pip install dadaptation") from exc
            return da.DAdaptAdan(
                params=parameters, lr=lr,
                betas=(
                    betas[0] if betas and len(betas) > 0 else 0.98,
                    betas[1] if betas and len(betas) > 1 else 0.92,
                    beta3 if beta3 is not None else 0.99,
                ),
                eps=eps if eps is not None else 1e-8,
                weight_decay=weight_decay if weight_decay is not None else 0.02,
                no_prox=no_prox, log_every=log_every,
                d0=d0, growth_rate=growth_rate,
            )

        case OptimizerType.DADAPT_ADA_GRAD:
            try:
                import dadaptation as da
            except ImportError as exc:
                raise ImportError("dadaptation is required. pip install dadaptation") from exc
            return da.DAdaptAdaGrad(
                params=parameters, lr=lr,
                momentum=momentum, log_every=log_every,
                weight_decay=wd,
                eps=eps if eps is not None else 0.0,
                d0=d0, growth_rate=growth_rate,
            )

        case OptimizerType.DADAPT_LION:
            try:
                import dadaptation as da
            except ImportError as exc:
                raise ImportError("dadaptation is required. pip install dadaptation") from exc
            return da.DAdaptLion(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                weight_decay=wd, log_every=log_every,
                d0=d0, fsdp_in_use=fsdp_in_use,
            )

        # ------------------------------------------------------------------ #
        # Prodigy family
        # ------------------------------------------------------------------ #

        case OptimizerType.PRODIGY:
            try:
                import prodigyopt
            except ImportError as exc:
                raise ImportError("prodigyopt is required. pip install prodigyopt") from exc
            return prodigyopt.Prodigy(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                beta3=beta3, eps=eps if eps is not None else 1e-8,
                weight_decay=wd, decouple=decouple,
                use_bias_correction=use_bias_correction,
                safeguard_warmup=safeguard_warmup,
                d0=d0, d_coef=d_coef, growth_rate=growth_rate,
                fsdp_in_use=fsdp_in_use, slice_p=slice_p,
            )

        case OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE:
            try:
                from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
            except ImportError as exc:
                raise ImportError("prodigyplus is required. pip install prodigyplus") from exc
            return ProdigyPlusScheduleFree(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.99),
                beta3=beta3, weight_decay=wd,
                use_bias_correction=use_bias_correction,
                d0=d0, d_coef=d_coef,
                eps=eps,
                stochastic_rounding=stochastic_rounding,
                fused_back_pass=fused_back_pass,
            )

        # ------------------------------------------------------------------ #
        # Adafactor
        # ------------------------------------------------------------------ #

        case OptimizerType.ADAFACTOR:
            try:
                from transformers.optimization import Adafactor
            except ImportError as exc:
                raise ImportError("transformers is required. pip install transformers") from exc

            effective_lr = None if relative_step else lr
            if relative_step and isinstance(parameters, list):
                params_list = list(parameters)
                for param in params_list:
                    if isinstance(param, dict) and "lr" in param:
                        param.pop("lr")
                parameters = params_list

            return Adafactor(
                params=parameters, lr=effective_lr,
                eps=(eps if eps is not None else 1e-30, eps2 if eps2 is not None else 1e-3),
                clip_threshold=clip_threshold, decay_rate=decay_rate,
                beta1=betas[0] if betas else None,
                weight_decay=wd, scale_parameter=scale_parameter,
                relative_step=relative_step, warmup_init=warmup_init,
            )

        # ------------------------------------------------------------------ #
        # CAME
        # ------------------------------------------------------------------ #

        case OptimizerType.CAME:
            try:
                from came_pytorch import CAME
            except ImportError:
                raise ImportError("came-pytorch is required. pip install came-pytorch")
            return CAME(
                params=parameters, lr=lr,
                eps=(eps if eps is not None else 1e-30, eps2 if eps2 is not None else 1e-16),
                betas=(
                    betas[0] if betas and len(betas) > 0 else 0.9,
                    betas[1] if betas and len(betas) > 1 else 0.999,
                    beta3 if beta3 is not None else 0.9999,
                ),
                weight_decay=wd,
            )

        case OptimizerType.CAME_8BIT:
            try:
                from came_pytorch import CAME8bit
            except ImportError:
                raise ImportError("came-pytorch is required. pip install came-pytorch")
            return CAME8bit(
                params=parameters, lr=lr,
                eps=(eps if eps is not None else 1e-30, eps2 if eps2 is not None else 1e-16),
                betas=(
                    betas[0] if betas and len(betas) > 0 else 0.9,
                    betas[1] if betas and len(betas) > 1 else 0.999,
                    beta3 if beta3 is not None else 0.9999,
                ),
                weight_decay=wd,
                min_8bit_size=min_8bit_size,
            )

        # ------------------------------------------------------------------ #
        # Advanced optimizers (adv_optm)
        # ------------------------------------------------------------------ #

        case OptimizerType.ADAMW_ADV:
            try:
                from adv_optm import AdamW_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return AdamW_adv(
                params=parameters, lr=lr,
                betas=betas or (0.0, 0.99),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd, use_bias_correction=use_bias_correction,
                stochastic_rounding=stochastic_rounding,
            )

        case OptimizerType.ADOPT_ADV:
            try:
                from adv_optm import Adopt_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Adopt_adv(
                params=parameters, lr=lr,
                betas=betas or (0.0, 0.9999),
                eps=eps if eps is not None else 1e-6,
                weight_decay=wd,
                stochastic_rounding=stochastic_rounding,
            )

        case OptimizerType.PRODIGY_ADV:
            try:
                from adv_optm import Prodigy_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Prodigy_adv(
                params=parameters, lr=lr,
                betas=betas or (0.0, 0.99),
                beta3=beta3, eps=eps if eps is not None else 1e-8,
                weight_decay=wd,
                stochastic_rounding=stochastic_rounding,
                d0=d0, d_coef=d_coef, growth_rate=growth_rate, slice_p=slice_p,
            )

        case OptimizerType.LION_ADV:
            try:
                from adv_optm import Lion_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Lion_adv(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.99),
                weight_decay=wd, clip_threshold=clip_threshold,
                stochastic_rounding=stochastic_rounding,
            )

        case OptimizerType.LION_PRODIGY_ADV:
            try:
                from adv_optm import Lion_Prodigy_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Lion_Prodigy_adv(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.99),
                beta3=beta3, weight_decay=wd, clip_threshold=clip_threshold,
                stochastic_rounding=stochastic_rounding,
                d0=d0, d_coef=d_coef, growth_rate=growth_rate, slice_p=slice_p,
            )

        case OptimizerType.SIMPLIFIED_ADEMAMIX:
            try:
                from adv_optm import Simplified_AdEMAMix
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Simplified_AdEMAMix(
                params=parameters, lr=lr,
                betas=betas or (0.99, 0.999),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd,
                use_bias_correction=use_bias_correction,
                stochastic_rounding=stochastic_rounding,
            )

        case OptimizerType.SIGNSGD_ADV:
            try:
                from adv_optm import SignSGD_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return SignSGD_adv(
                params=parameters, lr=lr,
                momentum=momentum, weight_decay=wd,
                stochastic_rounding=stochastic_rounding,
            )

        case OptimizerType.MUON_ADV:
            try:
                from adv_optm import Muon_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return Muon_adv(
                params=parameters, lr=lr,
                weight_decay=wd,
                stochastic_rounding=stochastic_rounding,
                nesterov=nesterov,
            )

        case OptimizerType.ADAMUON_ADV:
            try:
                from adv_optm import AdaMuon_adv
            except ImportError as exc:
                raise ImportError("adv_optm is required. pip install adv_optm") from exc
            return AdaMuon_adv(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.99),
                eps=eps if eps is not None else 1e-8,
                weight_decay=wd,
                stochastic_rounding=stochastic_rounding,
                nesterov=nesterov,
            )

        # ------------------------------------------------------------------ #
        # MUON (upstream)
        # ------------------------------------------------------------------ #

        case OptimizerType.MUON:
            try:
                from muon import SingleDeviceMuonWithAuxAdam
            except ImportError as exc:
                raise ImportError("muon is required. pip install muon") from exc
            return SingleDeviceMuonWithAuxAdam(
                param_groups=parameters if isinstance(parameters, list) else [{"params": list(parameters), "lr": lr}],
            )

        # ------------------------------------------------------------------ #
        # timm / pytorch-optimizer
        # ------------------------------------------------------------------ #

        case OptimizerType.ADABELIEF:
            try:
                from timm.optim.adabelief import AdaBelief
            except ImportError as exc:
                raise ImportError("timm is required. pip install timm") from exc
            return AdaBelief(
                params=parameters, lr=lr, weight_decay=wd,
                betas=betas or (0.9, 0.999),
                eps=eps if eps is not None else 1e-16,
                amsgrad=amsgrad, decoupled_decay=decoupled_decay,
                fixed_decay=fixed_decay, rectify=rectify,
                degenerated_to_sgd=degenerated_to_sgd,
            )

        case OptimizerType.TIGER:
            try:
                from pytorch_optimizer.optimizer.tiger import Tiger
            except ImportError as exc:
                raise ImportError("pytorch-optimizer is required. pip install pytorch-optimizer") from exc
            return Tiger(
                params=parameters, lr=lr, weight_decay=wd,
                beta=betas[0] if betas else 0.9,
                weight_decouple=decoupled_decay, fixed_decay=fixed_decay,
            )

        case OptimizerType.AIDA:
            try:
                from pytorch_optimizer.optimizer.aida import Aida
            except ImportError as exc:
                raise ImportError("pytorch-optimizer is required. pip install pytorch-optimizer") from exc
            return Aida(
                params=parameters, lr=lr, weight_decay=wd,
                betas=betas or (0.9, 0.999),
                weight_decouple=decoupled_decay, fixed_decay=fixed_decay,
                eps=eps if eps is not None else 1e-8,
            )

        case OptimizerType.ADOPT:
            try:
                from pytorch_optimizer.optimizer.adopt import ADOPT
            except ImportError as exc:
                raise ImportError("pytorch-optimizer is required. pip install pytorch-optimizer") from exc
            return ADOPT(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.9999),
                weight_decay=wd,
                weight_decouple=decoupled_decay, fixed_decay=fixed_decay,
                eps=eps if eps is not None else 1e-6,
            )

        case OptimizerType.YOGI:
            try:
                from pytorch_optimizer.optimizer.yogi import Yogi
            except ImportError as exc:
                raise ImportError("pytorch-optimizer is required. pip install pytorch-optimizer") from exc
            return Yogi(
                params=parameters, lr=lr,
                betas=betas or (0.9, 0.999),
                weight_decay=wd,
                weight_decouple=decoupled_decay, fixed_decay=fixed_decay,
                eps=eps if eps is not None else 1e-3,
            )

        case _:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


__all__ = [
    "OptimizerType",
    "create_optimizer",
]
