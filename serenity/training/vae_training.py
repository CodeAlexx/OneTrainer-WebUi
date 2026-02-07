"""VAE training utilities: encoder/decoder fine-tuning, KL + reconstruction loss.

Parity with OneTrainer's VAE fine-tuning support for SD1.5 and SDXL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class VAETrainMode(str, Enum):
    """Which parts of the VAE to train."""
    FULL = "full"
    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"


@dataclass
class VAETrainConfig:
    """Configuration for VAE fine-tuning."""

    enabled: bool = False
    mode: VAETrainMode = VAETrainMode.FULL
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    kl_weight: float = 1e-6
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 0.0
    gradient_checkpointing: bool = False


def setup_vae_finetuning(
    vae: nn.Module,
    config: VAETrainConfig,
) -> None:
    """Configure a VAE module for fine-tuning.

    Freezes/unfreezes encoder and decoder based on *config.mode* and
    optionally enables gradient checkpointing.
    """
    if not config.enabled:
        for param in vae.parameters():
            param.requires_grad_(False)
        vae.eval()
        logger.debug("VAE frozen (finetuning disabled)")
        return

    # Start by freezing everything
    for param in vae.parameters():
        param.requires_grad_(False)

    encoder = getattr(vae, "encoder", None)
    decoder = getattr(vae, "decoder", None)
    quant_conv = getattr(vae, "quant_conv", None)
    post_quant_conv = getattr(vae, "post_quant_conv", None)

    if config.mode in (VAETrainMode.FULL, VAETrainMode.ENCODER_ONLY):
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad_(True)
            logger.debug("VAE encoder unfrozen")
        if quant_conv is not None:
            for param in quant_conv.parameters():
                param.requires_grad_(True)

    if config.mode in (VAETrainMode.FULL, VAETrainMode.DECODER_ONLY):
        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad_(True)
            logger.debug("VAE decoder unfrozen")
        if post_quant_conv is not None:
            for param in post_quant_conv.parameters():
                param.requires_grad_(True)

    vae.train()

    if config.gradient_checkpointing and hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()
        logger.debug("VAE gradient checkpointing enabled")

    total_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vae.parameters())
    logger.info(
        "VAE finetuning: mode=%s, trainable=%d/%d params (lr=%s)",
        config.mode.value,
        total_trainable,
        total_params,
        config.learning_rate,
    )


def compute_vae_loss(
    vae: nn.Module,
    pixel_values: Tensor,
    config: VAETrainConfig,
) -> dict[str, Tensor]:
    """Run a VAE forward pass and compute reconstruction + KL losses.

    Returns a dict with keys: 'loss', 'reconstruction_loss', 'kl_loss'.
    """
    # Encode
    posterior = vae.encode(pixel_values).latent_dist
    z = posterior.sample()

    # Decode
    reconstructed = vae.decode(z).sample

    # Reconstruction loss (L1 + L2)
    l1 = F.l1_loss(
        reconstructed.to(dtype=torch.float32),
        pixel_values.to(dtype=torch.float32),
    )
    l2 = F.mse_loss(
        reconstructed.to(dtype=torch.float32),
        pixel_values.to(dtype=torch.float32),
    )
    reconstruction_loss = 0.5 * l1 + 0.5 * l2

    # KL divergence loss
    kl_loss = posterior.kl().mean()

    # Combined loss
    total = (
        config.reconstruction_weight * reconstruction_loss
        + config.kl_weight * kl_loss
    )

    return {
        "loss": total,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }


def vae_training_step(
    vae: nn.Module,
    pixel_values: Tensor,
    config: VAETrainConfig,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Execute one VAE training step.

    Handles forward pass, loss computation, backward, and optimizer step.
    Returns a dict of scalar loss values for logging.
    """
    optimizer.zero_grad()

    if scaler is not None:
        with torch.amp.autocast("cuda"):
            losses = compute_vae_loss(vae, pixel_values, config)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses = compute_vae_loss(vae, pixel_values, config)
        losses["loss"].backward()
        optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def vae_param_groups(
    vae: nn.Module,
    config: VAETrainConfig,
) -> list[dict[str, Any]]:
    """Build optimizer param groups for VAE fine-tuning."""
    trainable = [p for p in vae.parameters() if p.requires_grad]
    if not trainable:
        return []

    return [
        {
            "params": trainable,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "vae",
        }
    ]


__all__ = [
    "VAETrainMode",
    "VAETrainConfig",
    "setup_vae_finetuning",
    "compute_vae_loss",
    "vae_training_step",
    "vae_param_groups",
]
