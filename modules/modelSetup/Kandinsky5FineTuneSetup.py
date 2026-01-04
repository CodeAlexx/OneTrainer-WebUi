"""
Kandinsky 5 Fine-Tune Setup

Sets up a Kandinsky 5 model for fine-tuning.
"""

import torch

from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress


class Kandinsky5FineTuneSetup(BaseModelSetup):
    """Setup for Kandinsky 5 fine-tuning."""

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(train_device, temp_device, debug_mode)

    def create_parameters(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        
        # Add transformer parameters if available
        if model.transformer is not None:
            self._create_model_part_parameters(
                parameter_group_collection,
                "transformer",
                model.transformer,
                config.transformer,
            )
        
        return parameter_group_collection

    def setup_optimizations(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ):
        # Setup autocast context for mixed precision
        model.transformer_autocast_context, model.transformer_train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [config.weight_dtypes().transformer],
            config.enable_autocast_cache,
        )

        # Setup Gradient Checkpointing if enabled
        if config.gradient_checkpointing.enabled():
            if model.transformer is not None:
                if hasattr(model.transformer, 'visual_transformer_blocks'):
                    for block in model.transformer.visual_transformer_blocks:
                        block.gradient_checkpointing = True
                if hasattr(model.transformer, 'text_transformer_blocks'):
                    for block in model.transformer.text_transformer_blocks:
                        block.gradient_checkpointing = True

        # Quantize layers (NF4, INT8, etc.) if configured
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)

    def setup_model(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ):
        # Freeze VAE and text encoders by default
        if model.vae is not None:
            model.vae.requires_grad_(False)
        if model.text_encoder_qwen is not None:
            model.text_encoder_qwen.requires_grad_(False)
        if model.text_encoder_clip is not None:
            model.text_encoder_clip.requires_grad_(False)

    def setup_train_device(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ):
        model.to(self.train_device)

    def predict(
            self,
            model: Kandinsky5Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        """
        Forward pass for training.

        DiffusionTransformer3D.forward signature (from kandinsky-5-code):
            x: noisy latents [T, H, W, C] - NO batch dimension!
            text_embed: Qwen text embeddings [seq_len, 3584]
            pooled_text_embed: CLIP pooled embeddings [768]
            time: timesteps [1] - float 0-1 range
            visual_rope_pos: tuple of (pos_t, pos_h, pos_w) index tensors
            text_rope_pos: LongTensor position indices
            scale_factor: (1.0, 1.0, 1.0) default
            sparse_params: None
            attention_mask: None
        """
        dtype = model.transformer_train_dtype.torch_dtype() if model.transformer_train_dtype else torch.bfloat16

        # Get latents from batch - shape [B, C, T, H, W]
        latents = batch['latent_image'].to(self.train_device, dtype=dtype)
        batch_size = latents.shape[0]

        # kandinsky-5-code expects NO batch dimension, process one sample at a time
        # For now, only support batch_size=1
        if batch_size != 1:
            raise ValueError(f"Kandinsky 5 training currently only supports batch_size=1, got {batch_size}")

        # Reshape latents: [B=1, C, T, H, W] -> [T, H, W, C]
        latents = latents[0].permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
        num_frames, height, width, channels = latents.shape

        # Get text embeddings and remove batch dimension
        text_embed = batch.get('text_embed_qwen')
        if text_embed is not None:
            text_embed = text_embed[0].to(self.train_device, dtype=dtype)  # [seq_len, 3584]

        pooled_text_embed = batch.get('pooled_text_embed_clip')
        if pooled_text_embed is not None:
            pooled_text_embed = pooled_text_embed[0].to(self.train_device, dtype=dtype)  # [768]

        # Create RoPE positions
        # Visual: tuple of position INDEX tensors for RoPE3D
        patch_size = model.dit_config.get('patch_size', (1, 2, 2))
        patched_t = num_frames // patch_size[0]
        patched_h = height // patch_size[1]
        patched_w = width // patch_size[2]

        # visual_rope_pos is a TUPLE of position indices that index into precomputed RoPE buffers
        visual_rope_pos = (
            torch.arange(patched_t, device=self.train_device),
            torch.arange(patched_h, device=self.train_device),
            torch.arange(patched_w, device=self.train_device),
        )

        # Text RoPE: LongTensor of position indices
        text_len = text_embed.shape[0] if text_embed is not None else 256
        text_rope_pos = torch.arange(text_len, device=self.train_device, dtype=torch.long)

        # Sample noise with same shape as latents [T, H, W, C]
        noise = torch.randn_like(latents)

        # Sample timesteps (flow matching: 0 to 1) - keep as FLOAT
        if deterministic:
            timesteps = torch.tensor([0.5], device=self.train_device, dtype=dtype)
        else:
            timesteps = torch.rand((1,), device=self.train_device, dtype=dtype)

        # Create noisy latents using flow matching interpolation
        # x_t = (1 - t) * x_0 + t * noise
        t = timesteps.view(1, 1, 1, 1)  # Broadcast over [T, H, W, C]
        noisy_latents = (1 - t) * latents + t * noise

        # Forward pass through transformer
        # Note: time stays as float (0-1), the model handles conversion internally
        if model.transformer:
            with model.transformer_autocast_context:
                model_pred = model.transformer(
                    x=noisy_latents,
                    text_embed=text_embed,
                    pooled_text_embed=pooled_text_embed,
                    time=timesteps,  # Float 0-1, NOT integer!
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=(1.0, 1.0, 1.0),
                    sparse_params=None,
                    attention_mask=None,
                )
        else:
            model_pred = torch.zeros_like(noisy_latents)

        # Target for flow matching: velocity = noise - data
        target = noise - latents

        # Reshape predictions and targets back to [B, C, T, H, W] for loss calculation
        # model_pred is [T, H, W, C] -> [1, C, T, H, W]
        model_pred = model_pred.permute(3, 0, 1, 2).unsqueeze(0)
        target = target.permute(3, 0, 1, 2).unsqueeze(0)

        return {
            "model_pred": model_pred,
            "target": target,
            "timesteps": timesteps,
            "loss_weight": batch.get('loss_weight', torch.ones(batch_size, device=self.train_device)),
        }

    def calculate_loss(
            self,
            model: Kandinsky5Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> torch.Tensor:
        pred = data['model_pred']
        target = data['target']
        
        # Simple MSE
        loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction='none')
        loss = loss.mean()
        
        return loss

    def after_optimizer_step(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass
