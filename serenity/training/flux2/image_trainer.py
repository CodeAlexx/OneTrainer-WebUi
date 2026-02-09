"""
FLUX.2 Text-to-Image Trainer

Specialized trainer for FLUX.2 Klein text-to-image generation training.
Uses standard diffusion training with flow matching objective.

Features:
- Standard T2I training workflow
- Caption-based conditioning
- Optional caption dropout for CFG
- Aspect ratio bucketing support
- Latent and text embedding caching
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .base import Flux2BaseTrainer, Flux2TrainerConfig


@dataclass
class Flux2ImageTrainerConfig(Flux2TrainerConfig):
    """
    Configuration for FLUX.2 text-to-image training.

    Extends base config with T2I-specific settings.
    """
    # Caption handling
    caption_dropout: float = 0.0  # Probability of dropping caption (for CFG)
    caption_shuffle: bool = False  # Randomly shuffle caption words
    caption_prefix: str = ""  # Prefix to add to all captions
    caption_suffix: str = ""  # Suffix to add to all captions

    # Resolution settings
    resolution: int = 1024
    aspect_ratio_buckets: bool = True
    min_bucket_size: int = 512
    max_bucket_size: int = 1536

    # Augmentation
    random_flip: float = 0.0  # Probability of horizontal flip
    center_crop: bool = True

    def __post_init__(self):
        """Set training mode to T2I."""
        self.training_mode = "t2i"
        super().__post_init__()


class Flux2ImageTrainer(Flux2BaseTrainer):
    """
    Text-to-Image trainer for FLUX.2 Klein models.

    Standard diffusion training workflow:
    1. Load image + caption
    2. Encode image to latents (or load from cache)
    3. Encode caption to embeddings (or load from cache)
    4. Sample timestep and add noise
    5. Predict velocity
    6. Compute MSE loss against velocity target

    Usage:
        config = Flux2ImageTrainerConfig(
            model_path="/path/to/flux2-klein-4b",
            model_variant="klein-4b",
            resolution=1024,
        )
        trainer = Flux2ImageTrainer(config)
        trainer.load_model()

        # Inject adapter
        from serenity.adapters import create_adapter
        adapter = create_adapter("lokr", rank=16, model_type="flux_2_klein")
        trainer.inject_adapter(adapter)

        # Training loop
        for batch in dataloader:
            loss_dict = trainer.training_step(batch)
            loss_dict['loss'].backward()
    """

    def __init__(
        self,
        config: Flux2ImageTrainerConfig,
        model: Optional[Any] = None,
    ):
        """
        Initialize T2I trainer.

        Args:
            config: T2I training configuration
            model: Optional pre-loaded Flux2KleinModel
        """
        super().__init__(config, model)
        self.config: Flux2ImageTrainerConfig = config

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Prepare batch for T2I training.

        Expected batch format:
        - 'latents': Pre-encoded latents [B, C, H, W] (if cached)
        - 'pixel_values': Raw images [B, 3, H, W] (if not cached)
        - 'text_embeddings': Pre-encoded embeddings [B, L, D] (if cached)
        - 'input_ids': Tokenized text [B, L] (if not cached)
        - 'captions': Raw caption strings (for on-the-fly encoding)

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with 'latents' and 'text_embeddings'
        """
        result = {}

        # Handle latents
        if 'latents' in batch:
            # Use pre-cached latents
            result['latents'] = batch['latents']
        elif 'pixel_values' in batch:
            # Encode on-the-fly (not recommended for training)
            with torch.no_grad():
                result['latents'] = self._encode_images(batch['pixel_values'])
        else:
            raise ValueError("Batch must contain 'latents' or 'pixel_values'")

        # Handle text embeddings
        if 'text_embeddings' in batch:
            # Use pre-cached embeddings
            result['text_embeddings'] = batch['text_embeddings']
        elif 'captions' in batch:
            # Encode on-the-fly (not recommended for training)
            captions = self._process_captions(batch['captions'])
            with torch.no_grad():
                result['text_embeddings'] = self._encode_text(captions)
        else:
            raise ValueError("Batch must contain 'text_embeddings' or 'captions'")

        return result

    def _encode_images(self, pixel_values: Tensor) -> Tensor:
        """
        Encode images to latents using VAE.

        Args:
            pixel_values: Images [B, 3, H, W] in [-1, 1]

        Returns:
            Latents [B, C, H, W]
        """
        if self.model.vae is None:
            raise RuntimeError("VAE not loaded. Use cached latents or load full model.")

        pixel_values = pixel_values.to(self.train_device, dtype=self.train_dtype)
        latents = self.model.encode_image(pixel_values)
        return latents

    def _encode_text(self, captions: List[str]) -> Tensor:
        """
        Encode captions to text embeddings using Qwen3.

        FLUX.2 Klein uses stacked layer embeddings from Qwen3.

        Args:
            captions: List of caption strings

        Returns:
            Text embeddings [B, L, D]
        """
        if self.model.text_encoder is None:
            raise RuntimeError("Text encoder not loaded. Use cached embeddings or load full model.")

        # Use model's encode method
        embeddings = self.model.encode_prompt(
            captions,
            device=self.train_device,
            max_sequence_length=self.model.max_sequence_length,
        )

        return embeddings

    def _process_captions(self, captions: List[str]) -> List[str]:
        """
        Process captions with dropout, shuffle, prefix/suffix.

        Args:
            captions: Raw caption strings

        Returns:
            Processed captions
        """
        processed = []
        for caption in captions:
            # Caption dropout
            if self._is_training and self.config.caption_dropout > 0:
                if torch.rand(1).item() < self.config.caption_dropout:
                    caption = ""  # Drop caption for CFG training
                    processed.append(caption)
                    continue

            # Add prefix/suffix
            if self.config.caption_prefix:
                caption = f"{self.config.caption_prefix} {caption}"
            if self.config.caption_suffix:
                caption = f"{caption} {self.config.caption_suffix}"

            # Optional shuffle
            if self._is_training and self.config.caption_shuffle:
                words = caption.split()
                import random
                random.shuffle(words)
                caption = " ".join(words)

            processed.append(caption)

        return processed

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Execute T2I training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with 'loss' and metrics
        """
        # Use base implementation
        result = super().training_step(batch)

        # Add T2I-specific metrics
        result['mode'] = 't2i'

        return result

    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute T2I training loss.

        Uses standard MSE velocity loss.

        Args:
            prediction: Velocity prediction [B, C, H, W]
            target: Velocity target [B, C, H, W]
            weights: Optional per-sample weights [B,]

        Returns:
            Scalar loss
        """
        return super().compute_loss(prediction, target, weights)


def create_image_trainer(
    model_path: str,
    model_variant: str = "klein-4b",
    resolution: int = 1024,
    learning_rate: float = 1e-4,
    caption_dropout: float = 0.0,
    gradient_checkpointing: str = "per_block",
    blocks_to_swap: int = 0,
    train_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> Flux2ImageTrainer:
    """
    Factory function to create a T2I trainer.

    Args:
        model_path: Path to FLUX.2 Klein model
        model_variant: Model variant ("klein-4b", "klein-9b", etc.)
        resolution: Training resolution
        learning_rate: Learning rate
        caption_dropout: Caption dropout probability
        gradient_checkpointing: Checkpointing mode
        blocks_to_swap: Number of blocks to swap to CPU
        train_dtype: Training dtype
        **kwargs: Additional config options

    Returns:
        Configured Flux2ImageTrainer
    """
    config = Flux2ImageTrainerConfig(
        model_path=model_path,
        model_variant=model_variant,
        resolution=resolution,
        learning_rate=learning_rate,
        caption_dropout=caption_dropout,
        gradient_checkpointing=gradient_checkpointing,
        blocks_to_swap=blocks_to_swap,
        train_dtype=train_dtype,
        **kwargs,
    )

    return Flux2ImageTrainer(config)
