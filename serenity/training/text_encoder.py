"""Text encoder training utilities: LoRA, freeze/unfreeze, separate LRs.

Parity with OneTrainer's text encoder training setup across model setups.
Supports CLIP, T5, Llama, and dual-encoder configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TextEncoderType(str, Enum):
    """Supported text encoder architectures."""
    CLIP = "clip"
    T5 = "t5"
    LLAMA = "llama"
    QWEN = "qwen"


@dataclass
class TextEncoderTrainConfig:
    """Configuration for text encoder training."""

    train: bool = False
    learning_rate: float | None = None
    weight_decay: float = 0.0
    train_embedding: bool = False
    dropout_probability: float = 0.0
    layer_skip: int = 0
    stop_training_after: float | None = None
    stop_training_after_unit: str = "never"


@dataclass
class TextEncoderLoRATargets:
    """LoRA target module names for each text encoder type."""

    clip: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
    )
    t5: tuple[str, ...] = (
        "q",
        "k",
        "v",
        "o",
        "wi_0",
        "wi_1",
        "wo",
    )
    llama: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    qwen: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )


_DEFAULT_TARGETS = TextEncoderLoRATargets()


def get_text_encoder_lora_targets(
    encoder_type: TextEncoderType | str,
) -> tuple[str, ...]:
    """Return LoRA target module name patterns for a text encoder type."""
    if isinstance(encoder_type, str):
        encoder_type = TextEncoderType(encoder_type.lower())

    _map = {
        TextEncoderType.CLIP: _DEFAULT_TARGETS.clip,
        TextEncoderType.T5: _DEFAULT_TARGETS.t5,
        TextEncoderType.LLAMA: _DEFAULT_TARGETS.llama,
        TextEncoderType.QWEN: _DEFAULT_TARGETS.qwen,
    }
    targets = _map.get(encoder_type)
    if targets is None:
        raise ValueError(f"Unknown text encoder type: {encoder_type}")
    return targets


def freeze_text_encoder(text_encoder: nn.Module) -> None:
    """Freeze all parameters of a text encoder."""
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_encoder.eval()
    logger.debug("Text encoder frozen (%d parameters)", sum(1 for _ in text_encoder.parameters()))


def unfreeze_text_encoder(
    text_encoder: nn.Module,
    layer_indices: list[int] | None = None,
) -> None:
    """Unfreeze text encoder parameters.

    If *layer_indices* is given, only the specified transformer layers are
    unfrozen (useful for partial fine-tuning).  Otherwise all parameters
    are set to trainable.
    """
    if layer_indices is None:
        for param in text_encoder.parameters():
            param.requires_grad_(True)
        text_encoder.train()
        logger.debug("Text encoder fully unfrozen")
        return

    # Find the encoder layers container
    encoder_layers = _find_encoder_layers(text_encoder)
    if encoder_layers is None:
        logger.warning("Could not locate encoder layers; unfreezing all parameters")
        for param in text_encoder.parameters():
            param.requires_grad_(True)
        text_encoder.train()
        return

    # Freeze everything first
    freeze_text_encoder(text_encoder)
    text_encoder.train()

    # Selectively unfreeze
    for idx in layer_indices:
        if idx < 0:
            idx = len(encoder_layers) + idx
        if 0 <= idx < len(encoder_layers):
            for param in encoder_layers[idx].parameters():
                param.requires_grad_(True)
            logger.debug("Unfroze text encoder layer %d", idx)


def _find_encoder_layers(text_encoder: nn.Module) -> nn.ModuleList | None:
    """Locate the transformer layer list inside a text encoder."""
    # CLIP: text_model.encoder.layers
    layers = _get_nested_attr(text_encoder, "text_model.encoder.layers")
    if layers is not None:
        return layers

    # T5: encoder.block
    layers = _get_nested_attr(text_encoder, "encoder.block")
    if layers is not None:
        return layers

    # Llama / Qwen: model.layers
    layers = _get_nested_attr(text_encoder, "model.layers")
    if layers is not None:
        return layers

    # Direct layers attribute
    layers = _get_nested_attr(text_encoder, "layers")
    if layers is not None:
        return layers

    return None


def _get_nested_attr(module: nn.Module, path: str) -> Any | None:
    """Safely resolve a dotted attribute path."""
    parts = path.split(".")
    current = module
    for part in parts:
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, nn.ModuleList) else None


def setup_text_encoder_training(
    text_encoder: nn.Module,
    encoder_type: TextEncoderType | str,
    config: TextEncoderTrainConfig,
) -> None:
    """Configure a text encoder for training based on config.

    Freezes or unfreezes the encoder as needed and sets training mode.
    """
    if not config.train:
        freeze_text_encoder(text_encoder)
        return

    unfreeze_text_encoder(text_encoder)
    logger.info(
        "Text encoder (%s) set to training mode (lr=%s, wd=%s)",
        encoder_type,
        config.learning_rate,
        config.weight_decay,
    )


def text_encoder_param_groups(
    text_encoder: nn.Module,
    config: TextEncoderTrainConfig,
    default_lr: float,
) -> list[dict[str, Any]]:
    """Build optimizer param groups for a text encoder.

    Returns a list with a single param group using the text encoder's
    learning rate (falling back to *default_lr*).
    """
    trainable = [p for p in text_encoder.parameters() if p.requires_grad]
    if not trainable:
        return []

    lr = config.learning_rate if config.learning_rate is not None else default_lr
    return [
        {
            "params": trainable,
            "lr": lr,
            "weight_decay": config.weight_decay,
            "name": "text_encoder",
        }
    ]


def dual_text_encoder_param_groups(
    text_encoder_1: nn.Module,
    text_encoder_2: nn.Module,
    config_1: TextEncoderTrainConfig,
    config_2: TextEncoderTrainConfig,
    default_lr: float,
) -> list[dict[str, Any]]:
    """Build optimizer param groups for dual text encoders with separate LRs."""
    groups: list[dict[str, Any]] = []

    trainable_1 = [p for p in text_encoder_1.parameters() if p.requires_grad]
    if trainable_1:
        lr_1 = config_1.learning_rate if config_1.learning_rate is not None else default_lr
        groups.append({
            "params": trainable_1,
            "lr": lr_1,
            "weight_decay": config_1.weight_decay,
            "name": "text_encoder_1",
        })

    trainable_2 = [p for p in text_encoder_2.parameters() if p.requires_grad]
    if trainable_2:
        lr_2 = config_2.learning_rate if config_2.learning_rate is not None else default_lr
        groups.append({
            "params": trainable_2,
            "lr": lr_2,
            "weight_decay": config_2.weight_decay,
            "name": "text_encoder_2",
        })

    return groups


__all__ = [
    "TextEncoderType",
    "TextEncoderTrainConfig",
    "TextEncoderLoRATargets",
    "get_text_encoder_lora_targets",
    "freeze_text_encoder",
    "unfreeze_text_encoder",
    "setup_text_encoder_training",
    "text_encoder_param_groups",
    "dual_text_encoder_param_groups",
]
