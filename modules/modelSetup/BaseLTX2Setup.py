"""
Base LTX-2 Setup for OneTrainer

Implements flow matching training for LTX-2 video model.
"""
from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.LTX2Model import LTX2Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import enable_checkpointing_for_ltx2_transformer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

# LTX-2 imports
try:
    from ltx_core.model.transformer.modality import Modality
    LTX_CORE_AVAILABLE = True
except ImportError:
    LTX_CORE_AVAILABLE = False
    Modality = None


class BaseLTX2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        # Enable gradient checkpointing with layer offloading
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_ltx2_transformer(model.transformer, config)

        # Autocast context for training
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ], config.enable_autocast_cache
        )

        # Text encoder autocast (disable fp16 to avoid instability)
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().text_encoder,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )

        # Transformer autocast
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

        # Apply quantization to layers
        if model.text_encoder is not None:
            quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        if model.vae_encoder is not None:
            quantize_layers(model.vae_encoder, self.train_device, model.train_dtype, config)
        if model.vae_decoder is not None:
            quantize_layers(model.vae_decoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)

    def predict(
            self,
            model: LTX2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        """Forward pass for LTX-2 training.

        LTX-2 uses flow matching with the target: velocity = noise - latents
        """
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            # Get video latents from batch
            # Expected shape: [B, C, T, H, W]
            latent_video = batch['latent_video']
            batch_size = latent_video.shape[0]

            # Scale latents if needed
            scaled_latent_video = model.scale_latents(latent_video)

            # Get or encode text embeddings
            if 'text_encoder_hidden_state' in batch:
                text_embeddings = batch['text_encoder_hidden_state']
                text_mask = batch.get('text_encoder_mask', None)
            else:
                text_embeddings, text_mask = model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch_size,
                    rand=rand,
                    text=batch.get('prompt'),
                    text_encoder_dropout_probability=config.text_encoder.dropout_probability,
                )

            # Create noise
            latent_noise = self._create_noise(scaled_latent_video, config, generator)

            # Sample timesteps (0-1 range for flow matching)
            # Use shifted logit normal sampling like ltx-trainer
            timestep = self._get_timestep_continuous(
                deterministic,
                generator,
                batch_size,
                config,
            )

            # Add noise to latents using flow matching formula
            # noisy = (1 - sigma) * clean + sigma * noise
            sigma = timestep.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1] for 5D video
            noisy_latent_video = (1 - sigma) * scaled_latent_video + sigma * latent_noise

            # Create Modality for transformer input
            video_modality = model.create_modality(
                latents=noisy_latent_video.to(dtype=model.transformer_train_dtype.torch_dtype()),
                timesteps=timestep,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
            )

            # Forward pass through transformer
            with model.transformer_autocast_context:
                # LTX-2 forward: (video: Modality, audio: Modality, perturbations)
                predicted_video, _ = model.transformer(
                    video=video_modality,
                    audio=None,
                    perturbations=None,
                )

            # Unpatchify prediction back to latent shape
            if model.patchifier is not None:
                predicted_flow = model.patchifier.unpatchify(
                    predicted_video,
                    noisy_latent_video.shape[2:],  # (T, H, W)
                )
            else:
                predicted_flow = predicted_video

            # Target is velocity: noise - latents
            target_flow = latent_noise - scaled_latent_video

            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': target_flow,
            }

            if config.debug_mode:
                self._save_debug_output(
                    model, batch, train_progress, config,
                    latent_noise, noisy_latent_video, predicted_flow, target_flow, sigma
                )

        return model_output_data

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: torch.Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        """Sample timesteps using shifted logit-normal distribution."""
        if deterministic:
            # Use fixed timestep for deterministic runs
            return torch.full(
                (batch_size,),
                0.5,
                device=self.train_device,
                dtype=torch.float32,
            )

        # Shifted logit-normal sampling (like LTX-2 trainer)
        # Sample from normal distribution
        u = torch.randn(batch_size, generator=generator, device=self.train_device)

        # Apply logit-normal transformation with shift
        shift = getattr(config, 'timestep_shift', 0.0)
        scale = getattr(config, 'timestep_scale', 1.0)

        # Transform to (0, 1) range
        t = torch.sigmoid(u * scale + shift)

        return t

    def calculate_loss(
            self,
            model: LTX2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        """Calculate flow matching loss."""
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=None,  # LTX-2 doesn't use scheduler sigmas in the same way
        ).mean()

    def _save_debug_output(
            self,
            model: LTX2Model,
            batch: dict,
            train_progress: TrainProgress,
            config: TrainConfig,
            latent_noise: Tensor,
            noisy_latent: Tensor,
            predicted_flow: Tensor,
            target_flow: Tensor,
            sigma: Tensor,
    ):
        """Save debug visualizations during training."""
        with torch.no_grad():
            # For video, we can save the middle frame
            mid_frame = latent_noise.shape[2] // 2

            # Save noise (middle frame)
            self._save_image(
                self._project_latent_to_image(latent_noise[:, :, mid_frame]),
                config.debug_dir + "/training_batches",
                "1-noise",
                train_progress.global_step,
            )

            # Save noisy video (middle frame)
            self._save_image(
                self._project_latent_to_image(noisy_latent[:, :, mid_frame]),
                config.debug_dir + "/training_batches",
                "2-noisy_video",
                train_progress.global_step,
            )

            # Save predicted flow (middle frame)
            self._save_image(
                self._project_latent_to_image(predicted_flow[:, :, mid_frame]),
                config.debug_dir + "/training_batches",
                "3-predicted_flow",
                train_progress.global_step,
            )

            # Save target flow (middle frame)
            self._save_image(
                self._project_latent_to_image(target_flow[:, :, mid_frame]),
                config.debug_dir + "/training_batches",
                "4-target_flow",
                train_progress.global_step,
            )
