"""
FLUX.2 Image Editing Trainer

Specialized trainer for FLUX.2 Klein image-to-image editing training.
Supports multi-reference conditioning and mask-based regional editing.

Features:
- Multi-reference image conditioning (1-4 reference images)
- Mask-based regional editing support
- Reference dropout for generalization
- Flexible conditioning injection methods
- Compatible with FLUX.2 Fill/Edit variants

Training modes:
1. Single-reference: Standard I2I editing with one source
2. Multi-reference: Blend multiple references for composite generation
3. Masked editing: Apply changes only to masked regions
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Flux2BaseTrainer, Flux2TrainerConfig


@dataclass
class Flux2EditTrainerConfig(Flux2TrainerConfig):
    """
    Configuration for FLUX.2 image editing training.

    Extends base config with editing-specific settings.
    """
    # Reference image settings
    max_references: int = 4  # Maximum number of reference images
    reference_dropout: float = 0.1  # Probability of dropping each reference
    reference_blend_mode: str = "concat"  # "concat", "average", "attention"

    # Mask settings
    use_masks: bool = True  # Enable mask-based regional editing
    mask_blur: float = 0.0  # Gaussian blur on masks (0 = sharp)
    mask_threshold: float = 0.5  # Binarization threshold for soft masks
    invert_masks: bool = False  # Invert mask regions

    # Conditioning settings
    conditioning_scale: float = 1.0  # Scale for reference conditioning
    conditioning_injection: str = "concat"  # "concat", "add", "cross_attention"

    # Caption handling for edit instructions
    instruction_prefix: str = ""  # Prefix for edit instructions
    instruction_suffix: str = ""  # Suffix for edit instructions
    include_source_description: bool = True  # Include source image description

    # Resolution
    resolution: int = 1024
    aspect_ratio_buckets: bool = True

    def __post_init__(self):
        """Set training mode to edit."""
        self.training_mode = "edit"
        super().__post_init__()


class Flux2EditTrainer(Flux2BaseTrainer):
    """
    Image editing trainer for FLUX.2 Klein models.

    Supports training for image-to-image editing tasks:
    - Style transfer
    - Object manipulation
    - Regional inpainting
    - Multi-reference composition

    Input format:
    - Target image (what we want to generate)
    - Reference image(s) (conditioning input)
    - Optional mask (regions to edit)
    - Edit instruction (text describing the edit)

    Training objective:
    Learn to transform reference(s) into target following instruction.

    Usage:
        config = Flux2EditTrainerConfig(
            model_path="/path/to/flux2-klein-4b",
            model_variant="klein-4b",
            use_masks=True,
            max_references=2,
        )
        trainer = Flux2EditTrainer(config)
        trainer.load_model()

        # Inject adapter
        from serenity.adapters import create_adapter
        adapter = create_adapter("lora", rank=32, model_type="flux_2_klein")
        trainer.inject_adapter(adapter)

        # Training with multi-reference data
        for batch in dataloader:
            loss_dict = trainer.training_step(batch)
            loss_dict['loss'].backward()
    """

    def __init__(
        self,
        config: Flux2EditTrainerConfig,
        model: Optional[Any] = None,
    ):
        """
        Initialize edit trainer.

        Args:
            config: Edit training configuration
            model: Optional pre-loaded Flux2KleinModel
        """
        super().__init__(config, model)
        self.config: Flux2EditTrainerConfig = config

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Prepare batch for edit training.

        Expected batch format:
        - 'target_latents': Target image latents [B, C, H, W]
        - 'reference_latents': Reference latents [B, N, C, H, W] or [B, C, H, W]
        - 'text_embeddings': Edit instruction embeddings [B, L, D]
        - 'masks': Optional edit masks [B, 1, H, W]

        For on-the-fly encoding (not recommended):
        - 'target_images': Target images [B, 3, H, W]
        - 'reference_images': Reference images [B, N, 3, H, W] or list
        - 'instructions': Edit instruction strings

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with training tensors
        """
        result = {}

        # Handle target latents
        if 'target_latents' in batch:
            result['latents'] = batch['target_latents']
        elif 'target_images' in batch:
            with torch.no_grad():
                result['latents'] = self._encode_images(batch['target_images'])
        else:
            raise ValueError("Batch must contain 'target_latents' or 'target_images'")

        # Handle reference latents
        if 'reference_latents' in batch:
            result['reference_latents'] = batch['reference_latents']
        elif 'reference_images' in batch:
            with torch.no_grad():
                result['reference_latents'] = self._encode_references(batch['reference_images'])
        else:
            # No references - pure T2I mode
            result['reference_latents'] = None

        # Handle text embeddings
        if 'text_embeddings' in batch:
            result['text_embeddings'] = batch['text_embeddings']
        elif 'instructions' in batch:
            instructions = self._process_instructions(batch['instructions'])
            with torch.no_grad():
                result['text_embeddings'] = self._encode_text(instructions)
        else:
            raise ValueError("Batch must contain 'text_embeddings' or 'instructions'")

        # Handle masks
        if 'masks' in batch:
            result['masks'] = self._process_masks(batch['masks'])
        else:
            result['masks'] = None

        return result

    def _encode_references(self, reference_images: Union[Tensor, List[Tensor]]) -> Tensor:
        """
        Encode reference images to latents.

        Handles both single and multiple reference images.

        Args:
            reference_images: References [B, N, 3, H, W] or list of [B, 3, H, W]

        Returns:
            Reference latents [B, N, C, H, W]
        """
        if self.model.vae is None:
            raise RuntimeError("VAE not loaded. Use cached latents or load full model.")

        if isinstance(reference_images, list):
            # List of reference images
            ref_latents = []
            for ref in reference_images:
                ref = ref.to(self.train_device, dtype=self.train_dtype)
                latent = self.model.encode_image(ref)
                ref_latents.append(latent)
            # Stack along reference dimension
            return torch.stack(ref_latents, dim=1)  # [B, N, C, H, W]
        elif reference_images.dim() == 5:
            # Already batched [B, N, 3, H, W]
            B, N, C, H, W = reference_images.shape
            ref = reference_images.view(B * N, C, H, W)
            ref = ref.to(self.train_device, dtype=self.train_dtype)
            latents = self.model.encode_image(ref)
            # Reshape back
            _, C_lat, H_lat, W_lat = latents.shape
            return latents.view(B, N, C_lat, H_lat, W_lat)
        else:
            # Single reference [B, 3, H, W]
            ref = reference_images.to(self.train_device, dtype=self.train_dtype)
            latent = self.model.encode_image(ref)
            return latent.unsqueeze(1)  # [B, 1, C, H, W]

    def _encode_images(self, pixel_values: Tensor) -> Tensor:
        """Encode images to latents using VAE."""
        if self.model.vae is None:
            raise RuntimeError("VAE not loaded. Use cached latents or load full model.")
        pixel_values = pixel_values.to(self.train_device, dtype=self.train_dtype)
        return self.model.encode_image(pixel_values)

    def _encode_text(self, instructions: List[str]) -> Tensor:
        """Encode instructions to text embeddings."""
        if self.model.text_encoder is None:
            raise RuntimeError("Text encoder not loaded.")
        return self.model.encode_prompt(
            instructions,
            device=self.train_device,
            max_sequence_length=self.model.max_sequence_length,
        )

    def _process_instructions(self, instructions: List[str]) -> List[str]:
        """Process edit instructions with prefix/suffix."""
        processed = []
        for inst in instructions:
            if self.config.instruction_prefix:
                inst = f"{self.config.instruction_prefix} {inst}"
            if self.config.instruction_suffix:
                inst = f"{inst} {self.config.instruction_suffix}"
            processed.append(inst)
        return processed

    def _process_masks(self, masks: Tensor) -> Tensor:
        """
        Process masks for training.

        Args:
            masks: Raw masks [B, 1, H, W]

        Returns:
            Processed masks [B, 1, H, W]
        """
        masks = masks.to(self.train_device, dtype=self.train_dtype)

        # Apply blur if configured
        if self.config.mask_blur > 0:
            kernel_size = int(self.config.mask_blur * 6) | 1  # Ensure odd
            masks = F.gaussian_blur(masks, kernel_size=kernel_size)

        # Threshold for binarization
        if self.config.mask_threshold > 0:
            masks = (masks > self.config.mask_threshold).float()

        # Invert if configured
        if self.config.invert_masks:
            masks = 1.0 - masks

        return masks

    def _apply_reference_dropout(self, reference_latents: Tensor) -> Tensor:
        """
        Apply dropout to reference images during training.

        Randomly drops references to improve generalization.

        Args:
            reference_latents: References [B, N, C, H, W]

        Returns:
            References with dropout applied
        """
        transformer = getattr(self.model, "transformer", None)
        is_training = bool(getattr(transformer, "training", False))
        if not is_training or self.config.reference_dropout <= 0:
            return reference_latents

        B, N, C, H, W = reference_latents.shape

        # Generate dropout mask [B, N]
        keep_mask = torch.rand(B, N, device=reference_latents.device) > self.config.reference_dropout

        # Ensure at least one reference is kept per sample
        for i in range(B):
            if not keep_mask[i].any():
                keep_mask[i, 0] = True

        # Apply mask
        keep_mask = keep_mask.view(B, N, 1, 1, 1).expand_as(reference_latents)
        return reference_latents * keep_mask.float()

    def _blend_references(self, reference_latents: Tensor) -> Tensor:
        """
        Blend multiple references into single conditioning.

        Args:
            reference_latents: References [B, N, C, H, W]

        Returns:
            Blended references [B, C, H, W]
        """
        if self.config.reference_blend_mode == "average":
            # Simple averaging
            return reference_latents.mean(dim=1)
        elif self.config.reference_blend_mode == "concat":
            # Concatenate along channel dimension
            B, N, C, H, W = reference_latents.shape
            return reference_latents.view(B, N * C, H, W)
        else:
            # Default to first reference
            return reference_latents[:, 0]

    def forward_transformer(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        text_embeddings: Tensor,
        reference_latents: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass with reference conditioning.

        Args:
            noisy_latents: Noisy target latents [B, C, H, W]
            timesteps: Timesteps [B,]
            text_embeddings: Instruction embeddings [B, L, D]
            reference_latents: Reference latents [B, N, C, H, W] or None
            masks: Edit masks [B, 1, H, W] or None

        Returns:
            Velocity prediction [B, C, H, W]
        """
        # Apply reference dropout
        if reference_latents is not None:
            reference_latents = self._apply_reference_dropout(reference_latents)
            # Blend references
            ref_conditioning = self._blend_references(reference_latents)
            ref_conditioning = ref_conditioning * self.config.conditioning_scale
        else:
            ref_conditioning = None

        # Track original latent channels for output slicing after concat
        output_channels = noisy_latents.shape[1]

        # Prepare conditioning injection
        if ref_conditioning is not None and self.config.conditioning_injection == "concat":
            # Concatenate reference with noisy latents
            # This requires model to expect additional channels
            conditioned_latents = torch.cat([noisy_latents, ref_conditioning], dim=1)
        elif ref_conditioning is not None and self.config.conditioning_injection == "add":
            # Add reference as residual
            conditioned_latents = noisy_latents + ref_conditioning
        else:
            conditioned_latents = noisy_latents

        # Prepare for block swap if enabled
        if hasattr(self.model, 'prepare_block_swap_training'):
            self.model.prepare_block_swap_training()

        # Pack latents/text for transformer.
        packed_latents, image_ids = self.model.pack_latents(conditioned_latents)
        packed_text, text_ids = self.model.pack_text(text_embeddings)
        pooled_projections = self.model.pooled_text_projection(packed_text)
        guidance = torch.ones(packed_latents.shape[0], device=self.train_device, dtype=packed_latents.dtype)

        # Get dimensions
        _, _, height, width = noisy_latents.shape

        transformer_call_kwargs: dict[str, Any] = {
            "hidden_states": packed_latents,
            "timestep": timesteps / 1000,  # Normalize discrete timestep
            "guidance": guidance,
            "encoder_hidden_states": packed_text,
            "txt_ids": text_ids,
            "img_ids": image_ids,
            "return_dict": False,
        }
        transformer_params = inspect.signature(self.model.transformer.forward).parameters
        if "pooled_projections" in transformer_params:
            transformer_call_kwargs["pooled_projections"] = pooled_projections

        # Forward through transformer
        output = self.model.transformer(**transformer_call_kwargs)

        if isinstance(output, tuple):
            packed_pred = output[0]
        else:
            packed_pred = output.sample if hasattr(output, 'sample') else output

        # Unpack - handle potential channel expansion from conditioning
        if ref_conditioning is not None and self.config.conditioning_injection == "concat":
            # Slice to original output channels (not hardcoded //2)
            if packed_pred.dim() == 3:
                # Packed format [B, seq, C] — slice last dim to output_channels
                packed_pred = packed_pred[..., :output_channels]
            else:
                # Unpacked [B, C, H, W] — slice channel dim
                packed_pred = packed_pred[:, :output_channels]
            prediction = self.model.unpack_latents(packed_pred, height, width)
        else:
            prediction = self.model.unpack_latents(packed_pred, height, width)

        return prediction

    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute masked edit loss.

        Args:
            prediction: Velocity prediction [B, C, H, W]
            target: Velocity target [B, C, H, W]
            weights: Per-sample weights [B,]
            masks: Edit masks [B, 1, H, W]

        Returns:
            Scalar loss
        """
        # MSE per pixel
        mse = F.mse_loss(prediction, target, reduction='none')

        # Apply mask if provided
        if masks is not None:
            # Resize mask to match latent dimensions
            if masks.shape[-2:] != mse.shape[-2:]:
                masks = F.interpolate(
                    masks,
                    size=mse.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            # Weight loss by mask
            mse = mse * masks

        # Reduce over spatial dimensions
        mse = mse.mean(dim=[1, 2, 3])  # [B]

        # Apply weights
        if weights is not None:
            mse = mse * weights

        return mse.mean()

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Execute edit training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with 'loss' and metrics
        """
        # Prepare batch
        prepared = self.prepare_batch(batch)
        latents = prepared['latents'].to(self.train_device, dtype=self.train_dtype)
        text_embeddings = prepared['text_embeddings'].to(self.train_device, dtype=self.train_dtype)
        reference_latents = prepared.get('reference_latents')
        masks = prepared.get('masks')

        if reference_latents is not None:
            reference_latents = reference_latents.to(self.train_device, dtype=self.train_dtype)

        batch_size = latents.shape[0]

        # Sample timesteps
        timesteps = self.sample_timesteps(batch_size)

        # Sample noise
        noise = torch.randn_like(latents)

        # Compute noisy latents
        noisy_latents, _sigma = self.compute_noisy_latents(latents, noise, timesteps)

        # Compute target
        target = self.compute_velocity_target(latents, noise)

        # Forward pass with references
        prediction = self.forward_transformer(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_embeddings=text_embeddings,
            reference_latents=reference_latents,
            masks=masks,
        )

        # Compute loss
        weights = self.get_velocity_weight(timesteps)
        loss = self.compute_loss(prediction, target, weights, masks)

        # Metrics
        result = {
            'loss': loss,
            'timestep_mean': timesteps.float().mean(),
            'mode': 'edit',
        }

        if reference_latents is not None:
            result['num_references'] = reference_latents.shape[1]

        if masks is not None:
            result['mask_coverage'] = masks.mean()

        return result


def create_edit_trainer(
    model_path: str,
    model_variant: str = "klein-4b",
    resolution: int = 1024,
    learning_rate: float = 1e-4,
    max_references: int = 4,
    reference_dropout: float = 0.1,
    use_masks: bool = True,
    gradient_checkpointing: str = "per_block",
    blocks_to_swap: int = 0,
    train_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> Flux2EditTrainer:
    """
    Factory function to create an edit trainer.

    Args:
        model_path: Path to FLUX.2 Klein model
        model_variant: Model variant
        resolution: Training resolution
        learning_rate: Learning rate
        max_references: Maximum reference images
        reference_dropout: Reference dropout probability
        use_masks: Enable mask-based editing
        gradient_checkpointing: Checkpointing mode
        blocks_to_swap: Blocks to swap to CPU
        train_dtype: Training dtype
        **kwargs: Additional config options

    Returns:
        Configured Flux2EditTrainer
    """
    config = Flux2EditTrainerConfig(
        model_path=model_path,
        model_variant=model_variant,
        resolution=resolution,
        learning_rate=learning_rate,
        max_references=max_references,
        reference_dropout=reference_dropout,
        use_masks=use_masks,
        gradient_checkpointing=gradient_checkpointing,
        blocks_to_swap=blocks_to_swap,
        train_dtype=train_dtype,
        **kwargs,
    )

    return Flux2EditTrainer(config)
