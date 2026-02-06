"""
FLUX.2 Base Trainer

Shared training logic for FLUX.2 Klein models (4B and 9B variants).
Provides common functionality for both T2I and editing modes.

Key features:
- Flow matching training objective (velocity prediction)
- Qwen3 text encoder with stacked layer embeddings
- 32→128 channel VAE patchification
- Conductor-based memory management (like OneTrainer)
- Integrated checkpointing + activation offloading + layer offloading
"""

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import conductor-based checkpointing (OneTrainer-style)
from eritrainer.training.checkpointing import (
    enable_checkpointing_for_flux2_transformer,
    LayerOffloadConductor,
)
from eritrainer.training.torch_util import torch_gc

logger = logging.getLogger(__name__)


class GradientCheckpointingMethod(str, Enum):
    """Gradient checkpointing modes matching OneTrainer."""
    OFF = "off"
    ON = "on"
    CPU_OFFLOADED = "cpu_offloaded"

    def enabled(self) -> bool:
        return self != GradientCheckpointingMethod.OFF

    def offload(self) -> bool:
        return self == GradientCheckpointingMethod.CPU_OFFLOADED


@dataclass
class Flux2TrainerConfig:
    """
    Configuration for FLUX.2 training.

    Shared settings for both T2I and editing modes.
    """
    # Model settings
    model_path: str = ""
    model_variant: str = "klein-4b"  # "klein-4b", "klein-9b", "klein-4b-base", "klein-9b-base"

    # Training mode
    training_mode: str = "t2i"  # "t2i" (text-to-image) or "edit" (image editing)

    # Training settings
    train_dtype: torch.dtype = torch.bfloat16
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Flow matching settings
    timestep_density: str = "logit_normal"  # "uniform", "logit_normal", "sigmoid"
    timestep_bias: float = 0.0  # Bias towards low (negative) or high (positive) timesteps
    velocity_weighting: str = "uniform"  # "uniform", "sigma_sqrt", "snr"

    # VRAM optimization (conductor-based, matching OneTrainer)
    gradient_checkpointing: str = "cpu_offloaded"  # "off", "on", "cpu_offloaded"
    enable_activation_offloading: bool = True  # Offload activations to CPU during forward
    layer_offload_fraction: float = 0.0  # Fraction of layers to keep offloaded (0.9 = 90% on CPU)
    enable_async_offloading: bool = True  # Use async CUDA streams for transfers

    # Legacy block swap (deprecated - use layer_offload_fraction instead)
    blocks_to_swap: int = 0  # Number of transformer blocks to swap to CPU

    # Caching
    cache_latents: bool = True
    cache_text_embeddings: bool = True
    cache_dir: Optional[str] = None

    # Adapter settings (injected externally)
    adapter_type: str = "lora"
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    adapter_dropout: float = 0.0
    adapter_target_modules: List[str] = field(default_factory=list)

    # Quantization
    quantize_transformer: Optional[str] = None  # "int8", "fp8", None

    # Device settings
    train_device: str = "cuda"
    temp_device: str = "cpu"

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize model variant
        self.model_variant = self.model_variant.lower().replace("_", "-")

        # Validate training mode
        if self.training_mode not in ("t2i", "edit"):
            raise ValueError(f"Invalid training_mode: {self.training_mode}. Must be 't2i' or 'edit'")

        # Validate gradient checkpointing - support both old and new names
        valid_gc = ("off", "on", "cpu_offloaded", "none", "full", "per_block")
        if self.gradient_checkpointing not in valid_gc:
            raise ValueError(f"Invalid gradient_checkpointing: {self.gradient_checkpointing}")
        # Normalize old names to new
        gc_mapping = {"none": "off", "full": "on", "per_block": "cpu_offloaded"}
        if self.gradient_checkpointing in gc_mapping:
            self.gradient_checkpointing = gc_mapping[self.gradient_checkpointing]

    @property
    def is_4b(self) -> bool:
        """Check if using 4B variant."""
        return "4b" in self.model_variant

    @property
    def is_9b(self) -> bool:
        """Check if using 9B variant."""
        return "9b" in self.model_variant

    @property
    def is_base(self) -> bool:
        """Check if using undistilled base variant."""
        return "base" in self.model_variant

    def get_gradient_checkpointing_method(self) -> GradientCheckpointingMethod:
        """Get gradient checkpointing method as enum."""
        return GradientCheckpointingMethod(self.gradient_checkpointing)


class Flux2BaseTrainer(ABC):
    """
    Abstract base trainer for FLUX.2 models.

    Provides shared functionality for both T2I and editing modes:
    - Model loading and setup
    - Flow matching loss calculation
    - Timestep sampling
    - Gradient checkpointing
    - Block swapping
    """

    def __init__(
        self,
        config: Flux2TrainerConfig,
        model: Optional[Any] = None,
    ):
        """
        Initialize base trainer.

        Args:
            config: Training configuration
            model: Optional pre-loaded Flux2KleinModel
        """
        self.config = config
        self.model = model
        self.adapter = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Device setup
        self.train_device = torch.device(config.train_device)
        self.temp_device = torch.device(config.temp_device)
        self.train_dtype = config.train_dtype

        # Conductor for memory management (None until setup_optimizations called)
        self.transformer_offload_conductor: Optional[LayerOffloadConductor] = None

        # Initialize if model provided
        if model is not None:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Set up model for training with conductor-based memory management.

        This implements OneTrainer's unified approach where:
        1. Gradient checkpointing wraps each transformer block
        2. Activation offloading saves activations to CPU during forward
        3. Layer offloading moves layers between CPU/GPU as needed
        4. All managed by a single LayerOffloadConductor

        For 9B models on 24GB VRAM, use:
            gradient_checkpointing="cpu_offloaded"
            enable_activation_offloading=True
            layer_offload_fraction=0.9
        """
        if self.model is None:
            return

        gc_method = self.config.get_gradient_checkpointing_method()

        if gc_method.enabled():
            # Get the unwrapped transformer for checkpointing
            transformer = self._get_unwrapped_transformer()

            if transformer is None:
                logger.warning("Could not find transformer for checkpointing setup")
                return

            # Create conductor and wrap blocks with OffloadCheckpointLayer
            # This is the key integration - checkpointing + offloading in one system
            self.transformer_offload_conductor = self._create_offload_conductor(transformer)

            if self.transformer_offload_conductor is not None:
                # Attach to model for access during forward pass
                self.model.transformer_offload_conductor = self.transformer_offload_conductor

                logger.info("Gradient checkpointing enabled for Flux 2 transformer")
                if gc_method.offload() and self.config.enable_activation_offloading:
                    logger.info("Activation offloading: CPU")
                if self.config.layer_offload_fraction > 0:
                    logger.info(f"Layer offloading: {self.config.layer_offload_fraction * 100:.1f}%")

        # Legacy block swap (for backwards compatibility)
        # Prefer layer_offload_fraction instead
        if self.config.blocks_to_swap > 0 and self.config.layer_offload_fraction == 0:
            logger.warning(
                "blocks_to_swap is deprecated. Use layer_offload_fraction instead "
                "for integrated checkpointing + offloading."
            )
            self.model.enable_block_swap(
                blocks_to_swap=self.config.blocks_to_swap,
                device=self.train_device,
            )

    def _get_unwrapped_transformer(self) -> Optional[nn.Module]:
        """Get the unwrapped transformer module (handles PEFT wrapping)."""
        if self.model is None or self.model.transformer is None:
            return None

        transformer = self.model.transformer

        # Unwrap PEFT model if needed
        unwrapped = transformer
        while hasattr(unwrapped, 'base_model') or hasattr(unwrapped, 'model'):
            if hasattr(unwrapped, 'base_model'):
                unwrapped = unwrapped.base_model
            elif hasattr(unwrapped, 'model'):
                unwrapped = unwrapped.model

        return unwrapped

    def _create_offload_conductor(self, transformer: nn.Module) -> Optional[LayerOffloadConductor]:
        """
        Create LayerOffloadConductor with proper block wrapping.

        This follows OneTrainer's enable_checkpointing_for_flux2_transformer pattern:
        - Wraps transformer_blocks with OffloadCheckpointLayer
        - Wraps single_transformer_blocks with OffloadCheckpointLayer
        - Creates conductor that manages all memory operations
        """
        gc_method = self.config.get_gradient_checkpointing_method()

        # Check if transformer has the expected block structure
        has_transformer_blocks = hasattr(transformer, 'transformer_blocks')
        has_single_blocks = hasattr(transformer, 'single_transformer_blocks')

        if not has_transformer_blocks and not has_single_blocks:
            logger.warning(
                f"Transformer {type(transformer).__name__} does not have "
                "transformer_blocks or single_transformer_blocks. "
                "Cannot enable conductor-based checkpointing."
            )
            return None

        # Create conductor with proper settings
        conductor = LayerOffloadConductor(
            module=transformer,
            train_device=self.train_device,
            temp_device=self.temp_device,
            layer_offload_fraction=self.config.layer_offload_fraction,
            offload_activations=gc_method.offload() and self.config.enable_activation_offloading,
            enable_async=self.config.enable_async_offloading,
        )

        # Import checkpoint wrapper
        from eritrainer.training.checkpointing import create_checkpoint

        layer_index = 0

        # Wrap transformer_blocks (double-stream blocks)
        if has_transformer_blocks:
            for i, block in enumerate(transformer.transformer_blocks):
                transformer.transformer_blocks[i] = create_checkpoint(
                    orig_module=block,
                    train_device=self.train_device,
                    include_from_offload_param_names=["hidden_states", "encoder_hidden_states"],
                    conductor=conductor if gc_method.offload() else None,
                    layer_index=layer_index,
                    compile=False,
                )
                layer_index += 1

        # Wrap single_transformer_blocks (single-stream blocks)
        if has_single_blocks:
            for i, block in enumerate(transformer.single_transformer_blocks):
                transformer.single_transformer_blocks[i] = create_checkpoint(
                    orig_module=block,
                    train_device=self.train_device,
                    include_from_offload_param_names=["hidden_states"],
                    conductor=conductor if gc_method.offload() else None,
                    layer_index=layer_index,
                    compile=False,
                )
                layer_index += 1

        logger.info(f"Wrapped {layer_index} transformer blocks with checkpointing")

        return conductor

    def move_transformer_to_train_device(self) -> None:
        """
        Move transformer to train device with proper conductor initialization.

        Call this AFTER model loading but BEFORE training starts.
        The conductor will allocate static buffers and position layers optimally.
        """
        if self.transformer_offload_conductor is not None:
            # Conductor handles all device movement and buffer allocation
            self.transformer_offload_conductor.to(self.train_device)
            logger.info("Transformer moved to train device via conductor")
        elif self.model is not None and self.model.transformer is not None:
            # No conductor - simple move
            self.model.transformer.to(self.train_device)
            logger.info("Transformer moved to train device (no conductor)")

        torch_gc()

    def move_transformer_to_temp_device(self) -> None:
        """Move transformer to temp device (CPU) to free VRAM."""
        if self.transformer_offload_conductor is not None:
            self.transformer_offload_conductor.to(self.temp_device)
        elif self.model is not None and self.model.transformer is not None:
            self.model.transformer.to(self.temp_device)

        torch_gc()

    def inject_adapter(self, adapter: Any) -> None:
        """
        Inject adapter into the model.

        IMPORTANT: This should be called AFTER _setup_model() has wrapped
        the transformer blocks with OffloadCheckpointLayer. The checkpoint
        wrappers are applied to the underlying blocks, so PEFT wrapping
        the transformer still preserves our memory management.

        Args:
            adapter: Adapter instance (from eritrainer.adapters)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Inject into transformer - PEFT/LyCORIS may wrap the module or apply hooks in-place.
        injected_transformer = adapter.inject(self.model.transformer)
        if injected_transformer is not None and injected_transformer is not self.model.transformer:
            with suppress(AttributeError):
                self.model.transformer = injected_transformer
        self.adapter = adapter

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (adapter params)."""
        if self.model is None or self.model.transformer is None:
            return []
        if self.adapter is None:
            trainable = [param for param in self.model.transformer.parameters() if param.requires_grad]
            return trainable if trainable else list(self.model.transformer.parameters())
        return list(self.adapter.get_trainable_params())

    # =========================================================================
    # Timestep Sampling
    # =========================================================================

    def sample_timesteps(self, batch_size: int, latent_height: int = 64, latent_width: int = 64) -> Tensor:
        """
        Sample timesteps for flow matching training.

        FLUX.2 uses DISCRETE timesteps with resolution-dependent shift.
        OneTrainer formula: timestep = N * shift * u / ((shift - 1) * u + N)

        Args:
            batch_size: Number of timesteps to sample
            latent_height: Latent height for shift calculation (unpatchified)
            latent_width: Latent width for shift calculation (unpatchified)

        Returns:
            Timesteps tensor [B,] (discrete integers 0 to 999)
        """
        import math

        # Calculate resolution-dependent shift (matching OneTrainer)
        # Uses scheduler config defaults: base_shift=0.5, max_shift=1.15
        base_seq_len = 256
        max_seq_len = 4096
        base_shift = 0.5
        max_shift = 1.15
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        shift = math.exp(mu)

        num_train_timesteps = 1000
        u = torch.rand(batch_size, device=self.train_device)

        # Apply proper shift transformation (matching OneTrainer)
        # First scale u to [0, N], then apply shift
        t_scaled = u * num_train_timesteps
        t = num_train_timesteps * shift * t_scaled / ((shift - 1) * t_scaled + num_train_timesteps)
        t = t.long().clamp(0, num_train_timesteps - 1)

        # Legacy density options (now combined with shift)
        if self.config.timestep_density == "uniform":
            pass  # Already using uniform with shift
        elif self.config.timestep_density == "logit_normal":
            # Apply logit-normal on top of shifted distribution
            u = torch.randn(batch_size, device=self.train_device)
            u = u * 1.0 + self.config.timestep_bias
            logit_normal = torch.sigmoid(u)
            t = (logit_normal * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        elif self.config.timestep_density == "sigmoid":
            # Sigmoid of uniform (then discretize)
            u = torch.rand(batch_size, device=self.train_device) * 2 - 1  # [-1, 1]
            sigmoid_u = torch.sigmoid(u * 3)  # Squash to focus on middle
            t = (sigmoid_u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        else:
            raise ValueError(f"Unknown timestep_density: {self.config.timestep_density}")

        return t

    def get_velocity_weight(self, timesteps: Tensor) -> Tensor:
        """
        Get weighting for velocity loss at each timestep.

        Args:
            timesteps: Timesteps tensor [B,]

        Returns:
            Weights tensor [B,]
        """
        if self.config.velocity_weighting == "uniform":
            return torch.ones_like(timesteps)
        elif self.config.velocity_weighting == "sigma_sqrt":
            # Weight by sqrt(sigma) to emphasize low noise
            sigma = timesteps
            return torch.sqrt(sigma + 1e-6)
        elif self.config.velocity_weighting == "snr":
            # SNR-based weighting
            sigma = timesteps
            snr = (1 - sigma) / (sigma + 1e-6)
            return snr / (snr + 1)
        else:
            return torch.ones_like(timesteps)

    # =========================================================================
    # Flow Matching
    # =========================================================================

    def compute_noisy_latents(
        self,
        latents: Tensor,
        noise: Tensor,
        timesteps: Tensor,
        num_train_timesteps: int = 1000,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute noisy latents using flow matching interpolation.

        OneTrainer formula: sigma = (timestep + 1) / num_train_timesteps
        Interpolation: x_t = sigma * noise + (1 - sigma) * latents

        Args:
            latents: Clean latents [B, C, H, W]
            noise: Random noise [B, C, H, W]
            timesteps: Discrete timesteps [B,] (integers 0 to 999)
            num_train_timesteps: Total timesteps (default 1000)

        Returns:
            Tuple of (noisy latents [B, C, H, W], sigma [B, 1, 1, 1])
        """
        # Compute sigma from discrete timestep (matches OneTrainer)
        # +1 ensures sigma is never 0 (ranges from 1/1000 to 1.0)
        sigma = ((timesteps.float() + 1) / num_train_timesteps).view(-1, 1, 1, 1)

        # Flow matching interpolation: noise * sigma + latent * (1 - sigma)
        noisy_latents = sigma * noise + (1 - sigma) * latents
        return noisy_latents, sigma

    def compute_velocity_target(
        self,
        latents: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """
        Compute velocity prediction target.

        For flow matching: v = noise - x_0

        Args:
            latents: Clean latents [B, C, H, W]
            noise: Random noise [B, C, H, W]

        Returns:
            Velocity target [B, C, H, W]
        """
        return noise - latents

    def compute_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute flow matching loss.

        Args:
            prediction: Model velocity prediction [B, C, H, W]
            target: Velocity target [B, C, H, W]
            weights: Optional per-sample weights [B,]

        Returns:
            Scalar loss tensor
        """
        # MSE loss per sample
        mse = F.mse_loss(prediction, target, reduction='none')

        # Reduce over spatial dimensions
        mse = mse.mean(dim=[1, 2, 3])  # [B]

        # Apply weights
        if weights is not None:
            mse = mse * weights

        # Mean over batch
        return mse.mean()

    # =========================================================================
    # Forward Pass
    # =========================================================================

    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Prepare batch for training.

        Subclasses implement mode-specific batch preparation:
        - T2I: Text embeddings + latents
        - Edit: Text embeddings + latents + reference images + masks

        Args:
            batch: Raw batch from dataloader

        Returns:
            Prepared tensors dict
        """
        pass

    def forward_transformer(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        text_embeddings: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass through FLUX.2 transformer.

        Args:
            noisy_latents: Noisy latents [B, C, H, W]
            timesteps: Timesteps [B,]
            text_embeddings: Qwen3 embeddings [B, L, D]
            **kwargs: Additional mode-specific inputs

        Returns:
            Velocity prediction [B, C, H, W]
        """
        # Memory management:
        # - If conductor is active: OffloadCheckpointLayer handles everything
        # - If legacy block swap is active: call prepare_block_swap_training
        # - If neither: standard forward pass
        if self.transformer_offload_conductor is None:
            # Legacy block swap path (deprecated)
            if hasattr(self.model, 'prepare_block_swap_training'):
                self.model.prepare_block_swap_training()

        # Pack latents/text for transformer.
        packed_latents, image_ids = self.model.pack_latents(noisy_latents)
        packed_text, text_ids = self.model.pack_text(text_embeddings)
        pooled_projections = kwargs.pop("pooled_projections", None)
        if pooled_projections is None:
            if hasattr(self.model, "pooled_text_projection"):
                pooled_projections = self.model.pooled_text_projection(packed_text)
            else:
                # FLUX.2 transformer time_text_embed expects pooled text projections.
                pooled_projections = packed_text.mean(dim=1)

        # Get latent dimensions for unpacking.
        _, _, height, width = noisy_latents.shape

        # Forward through transformer
        # Note: FLUX.2 Klein uses different interface than FLUX.1
        # timesteps should be discrete integers, normalized to [0, 1] by /1000
        output = self.model.transformer(
            hidden_states=packed_latents,
            timestep=timesteps / 1000,  # Normalize discrete timestep
            guidance=None,
            encoder_hidden_states=packed_text,
            pooled_projections=pooled_projections,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )

        if isinstance(output, tuple):
            packed_pred = output[0]
        else:
            packed_pred = output.sample if hasattr(output, 'sample') else output

        # Unpack prediction [B, H*W//4, C*4] -> [B, C, H, W]
        prediction = self.model.unpack_latents(packed_pred, height, width)

        return prediction

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Execute single training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with 'loss' and optional metrics
        """
        # Prepare batch (mode-specific)
        prepared = self.prepare_batch(batch)
        latents = prepared['latents'].to(self.train_device, dtype=self.train_dtype)
        text_embeddings = prepared['text_embeddings'].to(self.train_device, dtype=self.train_dtype)

        batch_size = latents.shape[0]
        _, _, latent_height, latent_width = latents.shape

        # Sample timesteps (discrete, with resolution-dependent shift)
        timesteps = self.sample_timesteps(batch_size, latent_height * 2, latent_width * 2)

        # Sample noise
        noise = torch.randn_like(latents)

        # Compute noisy latents (returns tuple of noisy_latents, sigma)
        noisy_latents, _sigma = self.compute_noisy_latents(latents, noise, timesteps)

        # Compute velocity target
        target = self.compute_velocity_target(latents, noise)

        # Forward pass
        prediction = self.forward_transformer(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_embeddings=text_embeddings,
            **{k: v for k, v in prepared.items() if k not in ('latents', 'text_embeddings')},
        )

        # Compute loss
        weights = self.get_velocity_weight(timesteps)
        loss = self.compute_loss(prediction, target, weights)

        return {
            'loss': loss,
            'timestep_mean': timesteps.float().mean(),
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def save_adapter(self, path: Union[str, Path]) -> None:
        """Save adapter weights."""
        if self.adapter is None:
            raise RuntimeError("No adapter to save")
        self.adapter.save(str(path))

    def load_adapter(self, path: Union[str, Path]) -> None:
        """Load adapter weights."""
        if self.adapter is None:
            raise RuntimeError("No adapter to load into")
        self.adapter.load(str(path))

    def to_train_mode(self) -> None:
        """
        Set model to training mode and initialize conductor.

        This should be called after:
        1. Model loading
        2. Adapter injection (if using)
        3. Optimizer creation

        The conductor will allocate static memory buffers and position
        layers optimally for the training loop.
        """
        if self.model is not None:
            # Initialize conductor memory management if not already done
            # This allocates buffers and positions layers on CPU/GPU
            if self.transformer_offload_conductor is not None:
                self.move_transformer_to_train_device()

            self.model.transformer.train()

            # Keep frozen inference-only components out of the optimizer.
            for attr_name in ("vae", "text_encoder", "text_encoder_2"):
                component = getattr(self.model, attr_name, None)
                if component is None or not hasattr(component, "parameters"):
                    continue
                component.eval()
                for param in component.parameters():
                    param.requires_grad = False

            if self.adapter is None:
                # Native full-finetune mode: train the full transformer.
                for param in self.model.transformer.parameters():
                    param.requires_grad = True
            else:
                # Adapter mode: freeze backbone and train adapter params only.
                for param in self.model.transformer.parameters():
                    param.requires_grad = False
                for param in self.adapter.get_trainable_params():
                    param.requires_grad = True

    def to_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.transformer.eval()
