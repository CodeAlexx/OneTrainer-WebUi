"""Sampling pipeline for Serenity inference — schedulers, samplers, CFG, and prediction types."""

from __future__ import annotations

from serenity.inference.sampling.cfg import apply_cfg, compute_cfg, mahiro_correction, rescale_cfg
from serenity.inference.sampling.conditioning import Conditioning, create_noise, prepare_conditioning
from serenity.inference.sampling.prediction import (
    ContinuousEDMPrediction,
    ContinuousVPrediction,
    DiscreteFlowPrediction,
    EpsPrediction,
    FlowPrediction,
    FluxPrediction,
    Prediction,
    PredictionType,
    VPrediction,
    get_prediction,
)
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
    "ContinuousEDMPrediction",
    "ContinuousVPrediction",
    "DiscreteFlowPrediction",
    "get_prediction",
    # cfg
    "compute_cfg",
    "rescale_cfg",
    "mahiro_correction",
    "apply_cfg",
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
