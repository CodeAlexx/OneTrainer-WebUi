"""
BaseWanSetup - Common setup logic for Wan video model training

Adapted from BaseHunyuanVideoSetup for Wan's architecture.
Uses flow matching loss similar to Flux/Z-Image.
"""

from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.WanModel import WanModel, WanModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import enable_checkpointing_for_wan_transformer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
import torch.nn.functional as F

# Presets for targeting specific layers
PRESETS = {
    "attn-mlp": ["attn", "ff"],
    "attn-only": ["attn"],
    "blocks": ["blocks"],
    "full": [],
}


class BaseWanSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta
):
    """
    Base setup class for Wan video model training.
    
    Handles:
    - Gradient checkpointing
    - Autocast context setup
    - Text encoding
    - Flow matching forward pass
    - Loss calculation
    """

    def setup_optimizations(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Configure model optimizations like checkpointing, block swapping, and autocast."""

        # Block swapping for low VRAM training (ported from diffusion-pipe)
        # Must be done BEFORE moving model to device and before checkpointing
        if config.blocks_to_swap > 0:
            print(f"Enabling block swapping with {config.blocks_to_swap} blocks...")
            model.enable_block_swap(
                blocks_to_swap=config.blocks_to_swap,
                device=torch.device(config.train_device),
                reentrant_activation_checkpointing=False,  # TODO: make configurable
            )
            # Prepare for training
            model.prepare_block_swap_training()

        # Gradient checkpointing with layer offload for Wan transformer
        # This enables training large models by offloading layers to CPU
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_wan_transformer(model.transformer, config)

        # Setup autocast context
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache
        )

        model.transformer_autocast_context, model.transformer_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().transformer,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )

        # Quantize layers if configured
        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)

    def predict(
            self,
            model: WanModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        """
        Forward pass for training.
        
        Wan uses flow matching similar to Flux:
        - Sample timestep t
        - Create noisy latent: x_t = (1-t) * x_1 + t * x_0
        - Predict velocity/flow
        - Target is x_0 - x_1
        """
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            # Get latents (should be pre-encoded video latents)
            latents = batch['latent_image']  # Shape: (B, C, F, H, W)
            
            # Ensure 5D for video
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)  # Add frame dimension

            bs = latents.shape[0]
            device = latents.device
            dtype = model.train_dtype.torch_dtype()

            # Encode text if not cached
            text_embeddings, seq_lens = model.encode_text(
                train_device=self.train_device,
                batch_size=bs,
                rand=rand,
                text=batch.get('prompt'),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
            )

            # Sample timesteps (flow matching uses [0, 1])
            t = torch.rand(bs, device=device, generator=generator)
            
            # Apply shift if configured (like Flux)
            shift = config.transformer.get('shift', None) if hasattr(config.transformer, 'get') else None
            if shift:
                t = (t * shift) / (1 + (shift - 1) * t)

            # Create noise
            noise = torch.randn_like(latents)

            # Flow matching interpolation: x_t = (1-t) * x_1 + t * x_0
            t_expanded = t.view(-1, 1, 1, 1, 1)
            noisy_latents = (1 - t_expanded) * latents + t_expanded * noise

            # Target is flow: x_0 - x_1 = noise - latents
            target = noise - latents

            # Scale timestep for model input (Wan uses [0, 1000])
            timestep = t * 1000

            # Prepare model inputs
            # Wan expects: x, y (i2v conditioning), t, text_embeddings, seq_lens, clip_fea
            y = batch.get('i2v_conditioning')  # Optional for i2v
            clip_fea = batch.get('clip_features')  # Optional for i2v

            with model.transformer_autocast_context:
                # Forward through transformer
                # TODO: Adapt this to Wan's exact forward signature
                predicted_flow = model.transformer(
                    hidden_states=noisy_latents.to(dtype),
                    timestep=timestep.to(dtype),
                    encoder_hidden_states=text_embeddings.to(dtype) if text_embeddings is not None else None,
                    return_dict=False,
                )[0]

            model_output_data = {
                'loss_type': 'target',
                'timestep': t,  # Use normalized timestep for loss weighting
                'predicted': predicted_flow,
                'target': target,
            }

        return model_output_data

    def calculate_loss(
            self,
            model: WanModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        """Calculate flow matching loss."""
        predicted = data['predicted']
        target = data['target']
        
        # Simple MSE loss for flow matching
        loss = F.mse_loss(predicted.float(), target.float(), reduction='mean')
        
        return loss
