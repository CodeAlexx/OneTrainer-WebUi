"""
FLUX.2 Training Module

Provides specialized trainers for FLUX.2 Klein models (4B and 9B variants).
Supports both text-to-image generation and image editing modes.

Training Modes:
- T2I (Text-to-Image): Standard diffusion training with text conditioning
- Edit (Image-to-Image): Multi-reference editing with mask support

Usage:
    # Text-to-Image training
    from eritrainer.training.flux2 import create_trainer

    trainer = create_trainer(
        mode="t2i",
        model_path="/path/to/flux2-klein-4b",
        model_variant="klein-4b",
        resolution=1024,
    )

    # Image editing training
    trainer = create_trainer(
        mode="edit",
        model_path="/path/to/flux2-klein-4b",
        model_variant="klein-4b",
        max_references=2,
        use_masks=True,
    )

    # Inject adapter
    from eritrainer.adapters import create_adapter
    adapter = create_adapter("lokr", rank=16, model_type="flux_klein")
    trainer.inject_adapter(adapter)

    # Training loop
    for batch in dataloader:
        result = trainer.training_step(batch)
        result['loss'].backward()
"""

from typing import Optional, Union

import torch

from .base import Flux2BaseTrainer, Flux2TrainerConfig
from .image_trainer import (
    Flux2ImageTrainer,
    Flux2ImageTrainerConfig,
    create_image_trainer,
)
from .edit_trainer import (
    Flux2EditTrainer,
    Flux2EditTrainerConfig,
    create_edit_trainer,
)


def create_trainer(
    mode: str = "t2i",
    model_path: str = "",
    model_variant: str = "klein-4b",
    resolution: int = 1024,
    learning_rate: float = 1e-4,
    gradient_checkpointing: str = "per_block",
    blocks_to_swap: int = 0,
    train_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> Union[Flux2ImageTrainer, Flux2EditTrainer]:
    """
    Factory function to create a FLUX.2 trainer.

    Automatically selects the appropriate trainer based on mode.

    Args:
        mode: Training mode ("t2i" or "edit")
        model_path: Path to FLUX.2 Klein model
        model_variant: Model variant ("klein-4b", "klein-9b", etc.)
        resolution: Training resolution
        learning_rate: Learning rate
        gradient_checkpointing: Checkpointing mode ("none", "full", "per_block")
        blocks_to_swap: Number of blocks to swap to CPU for VRAM efficiency
        train_dtype: Training dtype (default: bfloat16)
        **kwargs: Mode-specific configuration options

    Returns:
        Configured trainer instance

    Raises:
        ValueError: If mode is not "t2i" or "edit"

    Example:
        # T2I trainer
        trainer = create_trainer(
            mode="t2i",
            model_path="/path/to/model",
            caption_dropout=0.1,  # T2I-specific
        )

        # Edit trainer
        trainer = create_trainer(
            mode="edit",
            model_path="/path/to/model",
            max_references=4,  # Edit-specific
            use_masks=True,    # Edit-specific
        )
    """
    mode = mode.lower()

    if mode == "t2i":
        return create_image_trainer(
            model_path=model_path,
            model_variant=model_variant,
            resolution=resolution,
            learning_rate=learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            blocks_to_swap=blocks_to_swap,
            train_dtype=train_dtype,
            **kwargs,
        )
    elif mode == "edit":
        return create_edit_trainer(
            model_path=model_path,
            model_variant=model_variant,
            resolution=resolution,
            learning_rate=learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            blocks_to_swap=blocks_to_swap,
            train_dtype=train_dtype,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 't2i' or 'edit'"
        )


def create_trainer_from_config(
    config: Union[Flux2ImageTrainerConfig, Flux2EditTrainerConfig],
    model: Optional[object] = None,
) -> Union[Flux2ImageTrainer, Flux2EditTrainer]:
    """
    Create trainer from configuration object.

    Args:
        config: Trainer configuration
        model: Optional pre-loaded model

    Returns:
        Configured trainer instance
    """
    if isinstance(config, Flux2EditTrainerConfig):
        return Flux2EditTrainer(config, model)
    else:
        return Flux2ImageTrainer(config, model)


__all__ = [
    # Base
    "Flux2BaseTrainer",
    "Flux2TrainerConfig",
    # T2I
    "Flux2ImageTrainer",
    "Flux2ImageTrainerConfig",
    "create_image_trainer",
    # Edit
    "Flux2EditTrainer",
    "Flux2EditTrainerConfig",
    "create_edit_trainer",
    # Factory
    "create_trainer",
    "create_trainer_from_config",
]
