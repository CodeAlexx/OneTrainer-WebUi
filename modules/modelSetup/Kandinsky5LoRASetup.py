"""
Kandinsky 5 LoRA Setup

Sets up a Kandinsky 5 model for LoRA training using the official DiffusionTransformer3D.
"""

import torch
import torch.nn.functional as F
from torch import nn

from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import create_peft_wrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
# Delayed import to avoid circular dependency - imported inside setup_model()
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress
from modules.util.checkpointing_util import enable_checkpointing

# Import K5 block types for checkpointing
import sys
sys.path.insert(0, 'models/kandinsky-5-code')
from kandinsky.models.dit import TransformerEncoderBlock, TransformerDecoderBlock

class Kandinsky5LoRASetup(BaseModelSetup):
    """Setup for Kandinsky 5 LoRA training."""

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

        # Collect LoRA parameters from transformer
        if model.transformer is not None and model.transformer_lora is not None:
            self._create_model_part_parameters(
                parameter_group_collection,
                "transformer_lora",
                model.transformer_lora,
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

        # Setup Gradient Checkpointing and Layer Offload if enabled
        if config.gradient_checkpointing.enabled():
            if model.transformer is not None:
                # Use OneTrainer's layer offload conductor for K5 blocks
                # TransformerEncoderBlock: text processing (x is the main tensor)
                # TransformerDecoderBlock: visual processing (visual_embed is main, text_embed is context)
                model.transformer_offload_conductor = enable_checkpointing(
                    model.transformer, config, False, [  # compile=False due to K5's @torch.compile decorators
                        (TransformerEncoderBlock, ["x"]),
                        (TransformerDecoderBlock, ["visual_embed", "text_embed"]),
                    ]
                )
                print("Gradient checkpointing + layer offload enabled for Kandinsky 5 transformer")

        # Quantize layers (NF4, INT8, etc.) if configured
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)

    def setup_model(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ):
        # 1. Freeze everything first
        model.requires_grad_(False)

        # 2. Inject LoRA into transformer using standard factory
        if model.transformer is not None:
            layer_filter = config.layer_filter.split(',') if config.layer_filter else None
            model.transformer_lora = create_peft_wrapper(
                model.transformer, "lora_transformer", config, layer_filter
            )

            # 3. Enable gradients for LoRA parameters only
            if model.transformer_lora is not None:
                model.transformer_lora.requires_grad_(True)
                model.transformer_lora.hook_to_module()

                # Convert LoRA to training dtype
                lora_dtype = config.lora_weight_dtype.torch_dtype() if config.lora_weight_dtype else torch.bfloat16
                model.transformer_lora.to(dtype=lora_dtype)

                trainable_params = sum(p.numel() for p in model.transformer_lora.parameters() if p.requires_grad)
                print(f"LoRA injected: {trainable_params:,} trainable parameters (dtype={lora_dtype})")

        # 4. Initialize model parameters for optimizer
        from modules.util.optimizer_util import init_model_parameters
        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
    ):
        # Move VAE and text encoders to temp device (they're frozen)
        if model.vae is not None:
            model.vae.to(self.temp_device)
        if model.text_encoder_qwen is not None:
            model.text_encoder_qwen.to(self.temp_device)
        if model.text_encoder_clip is not None:
            model.text_encoder_clip.to(self.temp_device)

        # Move transformer to train device
        if model.transformer is not None:
            model.transformer.to(self.train_device)

        # Move LoRA weights to train device
        if model.transformer_lora is not None:
            model.transformer_lora.to(self.train_device)

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
        Forward pass for LoRA training.

        Uses the same approach as FineTuneSetup - DiffusionTransformer3D expects
        NO batch dimension in the forward pass.
        """
        dtype = model.transformer_train_dtype.torch_dtype() if model.transformer_train_dtype else torch.bfloat16

        # Get latents from batch - shape [B, C, T, H, W]
        latents = batch['latent_image'].to(self.train_device, dtype=dtype)
        batch_size = latents.shape[0]

        # kandinsky-5-code expects NO batch dimension, process one sample at a time
        if batch_size != 1:
            raise ValueError(f"Kandinsky 5 training currently only supports batch_size=1, got {batch_size}")

        # Reshape latents: [B=1, C, T, H, W] -> [T, H, W, C]
        latents = latents[0].permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
        num_frames, height, width, channels = latents.shape

        # Encode text on-the-fly (not cached to avoid thread-safety issues)
        prompt = batch.get('prompt')
        if prompt is not None:
            prompt_text = prompt[0] if isinstance(prompt, list) else prompt
            model.text_encoder_to(self.train_device)
            text_embed, pooled_text_embed = model.encode_text([prompt_text], self.train_device)
            if text_embed is not None:
                text_embed = text_embed[0].to(dtype=dtype)  # [seq_len, 3584]
            if pooled_text_embed is not None:
                pooled_text_embed = pooled_text_embed[0].to(dtype=dtype)  # [768]
            model.text_encoder_to(self.temp_device)
        else:
            text_embed = None
            pooled_text_embed = None

        # Create RoPE positions
        patch_size = model.dit_config.get('patch_size', (1, 2, 2))
        patched_t = num_frames // patch_size[0]
        patched_h = height // patch_size[1]
        patched_w = width // patch_size[2]

        visual_rope_pos = (
            torch.arange(patched_t, device=self.train_device),
            torch.arange(patched_h, device=self.train_device),
            torch.arange(patched_w, device=self.train_device),
        )

        text_len = text_embed.shape[0] if text_embed is not None else 256
        text_rope_pos = torch.arange(text_len, device=self.train_device, dtype=torch.long)

        # Sample noise with same shape as latents [T, H, W, C]
        noise = torch.randn_like(latents)

        # Sample timesteps (flow matching: 0 to 1)
        if deterministic:
            timesteps = torch.tensor([0.5], device=self.train_device, dtype=dtype)
        else:
            timesteps = torch.rand((1,), device=self.train_device, dtype=dtype)

        # Create noisy latents using flow matching interpolation
        # x_t = (1 - t) * x_0 + t * noise
        t = timesteps.view(1, 1, 1, 1)
        noisy_latents = (1 - t) * latents + t * noise

        # Visual conditioning padding if model has visual_cond
        if model.dit_config.get('visual_cond', False):
            visual_cond = torch.zeros_like(noisy_latents)
            visual_cond_mask = torch.zeros(
                num_frames, height, width, 1,
                device=self.train_device, dtype=dtype
            )
            noisy_latents = torch.cat([noisy_latents, visual_cond, visual_cond_mask], dim=-1)

        # Forward pass through transformer
        if model.transformer:
            with model.transformer_autocast_context:
                model_pred = model.transformer(
                    x=noisy_latents,
                    text_embed=text_embed,
                    pooled_text_embed=pooled_text_embed,
                    time=timesteps,
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=(1.0, 1.0, 1.0),
                    sparse_params=None,
                    attention_mask=None,
                )
        else:
            model_pred = torch.zeros_like(noisy_latents[..., :channels])

        # If visual_cond was used, slice back to original channels
        if model.dit_config.get('visual_cond', False):
            model_pred = model_pred[..., :channels]

        # Target for flow matching: velocity = noise - data
        target = noise - latents

        # Reshape predictions and targets back to [B, C, T, H, W] for loss calculation
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
        loss_weight = data.get('loss_weight', 1.0)

        # MSE loss for flow matching
        loss = F.mse_loss(pred.float(), target.float(), reduction='none')

        # Average over all dimensions except batch
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        # Apply loss weight if provided
        if isinstance(loss_weight, torch.Tensor):
            loss = loss * loss_weight

        return loss.mean()

    def after_optimizer_step(
            self,
            model: Kandinsky5Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass
