"""Sampling pipeline for Serenity inference — schedulers, samplers, CFG, and prediction types."""

from __future__ import annotations

from serenity.inference.sampling.cfg import (
    CFGHookRegistry,
    CFGHookType,
    CfgHookFn,
    apply_cfg,
    compute_cfg,
    epsilon_scaling,
    mahiro_correction,
    rescale_cfg,
)
from serenity.inference.sampling.conditioning import Conditioning, create_noise, prepare_conditioning
from serenity.inference.sampling.prediction import (
    ContinuousEDMPrediction,
    ContinuousVPrediction,
    DiscreteFlowPrediction,
    EDMPrediction,
    EpsPrediction,
    FlowPrediction,
    FluxPrediction,
    Prediction,
    PredictionType,
    VPrediction,
    get_prediction,
)
from serenity.inference.sampling.regions import AreaRegion, compose_regional_predictions
from serenity.inference.sampling.sampler import (
    DenoiseFn,
    SamplerType,
    create_model_fn,
    euler_ancestral_sample,
    euler_sample,
    sample,
)
from serenity.inference.sampling.schedulers import SchedulerType, compute_sigmas

__all__ = [
    # prediction
    "PredictionType",
    "Prediction",
    "EpsPrediction",
    "VPrediction",
    "FlowPrediction",
    "FluxPrediction",
    "EDMPrediction",
    "ContinuousEDMPrediction",
    "ContinuousVPrediction",
    "DiscreteFlowPrediction",
    "get_prediction",
    # cfg
    "CFGHookRegistry",
    "CFGHookType",
    "CfgHookFn",
    "compute_cfg",
    "rescale_cfg",
    "mahiro_correction",
    "epsilon_scaling",
    "apply_cfg",
    # regions
    "AreaRegion",
    "compose_regional_predictions",
    # schedulers
    "SchedulerType",
    "compute_sigmas",
    # sampler
    "SamplerType",
    "DenoiseFn",
    "sample",
    "euler_sample",
    "euler_ancestral_sample",
    "create_model_fn",
    # conditioning
    "Conditioning",
    "create_noise",
    "prepare_conditioning",
]
