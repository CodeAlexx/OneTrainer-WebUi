"""Per-component model weight dtype management.

Provides a centralized way to track which dtype each model component
should use, enabling mixed-precision training configurations.

Usage::

    from serenity.core.weight_dtypes import create_weight_dtypes

    dtypes = create_weight_dtypes(config)
    vae_dtype = dtypes.vae  # -> torch.dtype or None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from serenity.core.enums import DataType

if TYPE_CHECKING:
    from serenity.core.config import TrainConfig


# ---------------------------------------------------------------------------
# Dtype conversion helper
# ---------------------------------------------------------------------------

def dtype_from_config_value(value: DataType | str) -> torch.dtype | None:
    """Convert a Serenity ``DataType`` enum (or string) to a ``torch.dtype``.

    Returns ``None`` for quantized types or ``NONE`` since they require
    special loading rather than a simple cast.
    """
    if isinstance(value, str):
        try:
            value = DataType(value)
        except (ValueError, KeyError):
            # Try case-insensitive
            upper = value.upper().replace(" ", "_")
            try:
                value = DataType(upper)
            except (ValueError, KeyError):
                return None

    return value.torch_dtype()


# ---------------------------------------------------------------------------
# ModelWeightDtypes dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelWeightDtypes:
    """Per-component dtype specification for a training run.

    Each field stores the ``torch.dtype`` (or ``None`` for quantized /
    special types) that the corresponding model component should use.
    """

    train_dtype: torch.dtype | None = None
    fallback_train_dtype: torch.dtype | None = None

    unet: torch.dtype | None = None
    prior: torch.dtype | None = None
    transformer: torch.dtype | None = None
    text_encoder: torch.dtype | None = None
    text_encoder_2: torch.dtype | None = None
    text_encoder_3: torch.dtype | None = None
    text_encoder_4: torch.dtype | None = None
    vae: torch.dtype | None = None
    effnet_encoder: torch.dtype | None = None
    decoder: torch.dtype | None = None
    decoder_text_encoder: torch.dtype | None = None
    decoder_vqgan: torch.dtype | None = None
    lora: torch.dtype | None = None
    embedding: torch.dtype | None = None

    def all_dtypes(self) -> list[torch.dtype | None]:
        """Return all component dtypes as a flat list."""
        return [
            self.unet,
            self.prior,
            self.transformer,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.text_encoder_4,
            self.vae,
            self.effnet_encoder,
            self.decoder,
            self.decoder_text_encoder,
            self.decoder_vqgan,
            self.lora,
            self.embedding,
        ]

    def unique_dtypes(self) -> set[torch.dtype]:
        """Return the set of unique non-None dtypes across all components."""
        return {d for d in self.all_dtypes() if d is not None}

    def needs_mixed_precision(self) -> bool:
        """True if components use different dtypes, requiring autocast."""
        all_types = [d for d in self.all_dtypes() + [self.train_dtype] if d is not None]
        return len(set(all_types)) > 1

    @classmethod
    def from_single_dtype(cls, dtype: torch.dtype) -> ModelWeightDtypes:
        """Create an instance where every component uses the same dtype."""
        return cls(
            train_dtype=dtype,
            fallback_train_dtype=dtype,
            unet=dtype,
            prior=dtype,
            transformer=dtype,
            text_encoder=dtype,
            text_encoder_2=dtype,
            text_encoder_3=dtype,
            text_encoder_4=dtype,
            vae=dtype,
            effnet_encoder=dtype,
            decoder=dtype,
            decoder_text_encoder=dtype,
            decoder_vqgan=dtype,
            lora=dtype,
            embedding=dtype,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_weight_dtypes(config: TrainConfig) -> ModelWeightDtypes:
    """Build ``ModelWeightDtypes`` from a ``TrainConfig``.

    Reads the ``weight_dtype`` field from each model-part sub-config
    (unet, transformer, text_encoder, etc.) and converts to torch dtypes.
    """
    def _get_part_dtype(part_name: str) -> torch.dtype | None:
        part = getattr(config, part_name, None)
        if part is None:
            return None
        weight_dtype = getattr(part, "weight_dtype", None)
        if weight_dtype is None:
            return None
        return dtype_from_config_value(weight_dtype)

    return ModelWeightDtypes(
        train_dtype=dtype_from_config_value(config.train_dtype),
        fallback_train_dtype=dtype_from_config_value(config.fallback_train_dtype),
        unet=_get_part_dtype("unet"),
        prior=_get_part_dtype("prior"),
        transformer=_get_part_dtype("transformer"),
        text_encoder=_get_part_dtype("text_encoder"),
        text_encoder_2=_get_part_dtype("text_encoder_2"),
        text_encoder_3=_get_part_dtype("text_encoder_3"),
        text_encoder_4=_get_part_dtype("text_encoder_4"),
        vae=_get_part_dtype("vae"),
        effnet_encoder=_get_part_dtype("effnet_encoder"),
        decoder=_get_part_dtype("decoder"),
        decoder_text_encoder=_get_part_dtype("decoder_text_encoder"),
        decoder_vqgan=_get_part_dtype("decoder_vqgan"),
        lora=dtype_from_config_value(config.lora_weight_dtype),
        embedding=dtype_from_config_value(config.embedding_weight_dtype),
    )


__all__ = [
    "ModelWeightDtypes",
    "create_weight_dtypes",
    "dtype_from_config_value",
]
