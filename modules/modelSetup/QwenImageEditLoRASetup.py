"""
QwenImageEditLoRASetup - Sets up LoRA training for Qwen-Image-Edit.

Based on QwenLoRASetup but for image editing model.
Handles control image conditioning during training.
"""

from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.QwenImageEditModel import QwenImageEditModel
from modules.modelSetup.BaseQwenSetup import BaseQwenSetup
from modules.module.LoRAModule import create_peft_wrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class QwenImageEditLoRASetup(
    BaseQwenSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: QwenImageEditModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "text_encoder", model.text_encoder_lora, config.text_encoder)
        self._create_model_part_parameters(parameter_group_collection, "transformer",  model.transformer_lora,  config.transformer)

        if config.train_any_embedding() or config.train_any_output_embedding():
            raise NotImplementedError("Embeddings not implemented for Qwen-Image-Edit")

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: QwenImageEditModel,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder", model.text_encoder_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: QwenImageEditModel,
            config: TrainConfig,
    ):
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "text_encoder")

        if model.text_encoder is not None:
            model.text_encoder_lora = create_peft_wrapper(
                model.text_encoder, "text_encoder", config
            ) if create_te else None

        model.transformer_lora = create_peft_wrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: QwenImageEditModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: QwenImageEditModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)

    def predict(
            self,
            model: QwenImageEditModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        """
        Training forward pass for Qwen-Image-Edit.

        Based on SimpleTuner's _model_predict_edit_v1:
        - Control (source) latents are packed alongside target latents
        - Transformer receives concatenated tokens with img_shapes describing both
        - Only target portion of output is used for loss
        """
        with model.autocast_context:
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(train_progress.global_step)
            rand = Random(train_progress.global_step)

            is_align_prop_step = getattr(config, 'align_prop', False) and (rand.random() < getattr(config, 'align_prop_probability', 0))

            vae_scale_factor = 8  # Qwen Image uses 8x downscaling
            latent_channels = model.vae.config.z_dim  # VAE latent dimension (16 for Qwen)

            # Get target latent image
            latent_image = batch['latent_image']
            batch_size = latent_image.shape[0]

            # Get control/conditioning latent image
            control_latents = batch.get('latent_conditioning_image')
            if control_latents is None:
                raise ValueError(
                    "Qwen-Image-Edit training requires control (conditioning) images. "
                    "Use -condlabel suffix files or enable custom_conditioning_image."
                )

            # Handle both 4D and 5D latents (Qwen VAE can output 5D for video)
            if latent_image.dim() == 5:
                latent_image = latent_image.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
            if control_latents.dim() == 5:
                control_latents = control_latents.squeeze(2)

            _, _, latent_height, latent_width = latent_image.shape
            _, _, control_height, control_width = control_latents.shape

            # Scale latents (Qwen VAE uses specific normalization)
            if hasattr(model.vae.config, 'latents_mean') and model.vae.config.latents_mean is not None:
                latents_mean = torch.tensor(model.vae.config.latents_mean).view(1, -1, 1, 1).to(
                    device=latent_image.device, dtype=latent_image.dtype
                )
                latents_std = 1.0 / torch.tensor(model.vae.config.latents_std).view(1, -1, 1, 1).to(
                    device=latent_image.device, dtype=latent_image.dtype
                )
                scaled_latent_image = (latent_image - latents_mean) * latents_std
                scaled_control_latents = (control_latents - latents_mean) * latents_std
            else:
                scaling_factor = model.vae.config.scaling_factor
                scaled_latent_image = latent_image * scaling_factor
                scaled_control_latents = control_latents * scaling_factor

            # Generate timesteps
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config.num_train_timesteps,
                deterministic,
                generator,
                batch_size,
                config,
            )

            # Add noise to target image using flow matching approach
            latent_noise = self._create_noise(scaled_latent_image, config, generator)
            noisy_latents, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # Pack latents using Qwen's pack_latents function
            # From diffusers QwenImagePipeline
            def pack_latents(latents, batch_size, num_channels, height, width):
                latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
                latents = latents.permute(0, 2, 4, 1, 3, 5)
                latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
                return latents

            def unpack_latents(latents, height, width, vae_scale_factor):
                batch_size, num_patches, channels = latents.shape
                latent_height = height // vae_scale_factor
                latent_width = width // vae_scale_factor
                latents = latents.view(batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2)
                latents = latents.permute(0, 3, 1, 4, 2, 5)
                latents = latents.reshape(batch_size, channels // 4, latent_height, latent_width)
                return latents

            # Pack noisy target latents
            packed_noisy = pack_latents(
                noisy_latents,
                batch_size,
                latent_channels,
                latent_height,
                latent_width,
            )

            # Pack control latents
            packed_control = pack_latents(
                scaled_control_latents,
                batch_size,
                latent_channels,
                control_height,
                control_width,
            )

            # Concatenate: target tokens + control tokens
            transformer_inputs = torch.cat([packed_noisy, packed_control], dim=1)

            # Prepare text embeddings
            text_encoder_output = batch.get('text_encoder_hidden_state')
            if text_encoder_output is None:
                text_encoder_output = self._encode_text_for_qwen(model, batch, config)

            # Ensure correct shape and device
            prompt_embeds = text_encoder_output.to(
                device=self.train_device,
                dtype=model.train_dtype.torch_dtype(),
            )
            if prompt_embeds.dim() == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)

            # Get attention mask
            prompt_embeds_mask = batch.get('tokens_mask')
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(self.train_device, dtype=torch.int64)
                if prompt_embeds_mask.dim() == 1:
                    prompt_embeds_mask = prompt_embeds_mask.unsqueeze(0)

            # Prepare img_shapes: describes token structure for each batch item
            # Format: [(frames, patch_height, patch_width), ...] for each image in sequence
            img_shapes = [
                [
                    (1, latent_height // 2, latent_width // 2),      # Target
                    (1, control_height // 2, control_width // 2),   # Control
                ]
                for _ in range(batch_size)
            ]

            # Sequence lengths for text
            txt_seq_lens = [prompt_embeds.shape[1]] * batch_size

            # Normalize timestep to [0, 1]
            timestep_normalized = timestep.float() / 1000.0

            # Forward pass through transformer
            model_output = model.transformer(
                hidden_states=transformer_inputs.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep_normalized,
                guidance=None,  # Qwen doesn't use guidance during training
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            # Extract only target portion (first set of tokens)
            target_token_count = packed_noisy.shape[1]
            model_output = model_output[:, :target_token_count]

            # Unpack to spatial format
            pixel_height = latent_height * vae_scale_factor
            pixel_width = latent_width * vae_scale_factor
            model_output = unpack_latents(model_output, pixel_height, pixel_width, vae_scale_factor)

            # For flow matching, target is the velocity (noise - clean)
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
            }

            if config.model_type.is_flow_matching():
                # Flow matching target: velocity = noise - latent
                target = latent_noise - scaled_latent_image
            else:
                # Standard diffusion: predict noise
                target = latent_noise

            # Latent mask for masked training
            if config.masked_training:
                latent_mask = batch.get('latent_mask')
                if latent_mask is not None:
                    model_output_data['latent_mask'] = latent_mask

            model_output_data['predicted'] = model_output
            model_output_data['target'] = target

        return model_output_data

    def _encode_text_for_qwen(
            self,
            model: QwenImageEditModel,
            batch: dict,
            config: TrainConfig,
    ) -> Tensor:
        """Encode text when not using cached embeddings."""
        from modules.model.QwenImageEditModel import (
            DEFAULT_PROMPT_TEMPLATE,
            DEFAULT_PROMPT_TEMPLATE_CROP_START,
            PROMPT_MAX_LENGTH,
        )

        prompt = batch.get('prompt', [''])[0] if isinstance(batch.get('prompt'), list) else batch.get('prompt', '')

        # Format prompt
        formatted_prompt = DEFAULT_PROMPT_TEMPLATE.format(prompt=prompt)

        # Tokenize
        tokens = model.tokenizer(
            formatted_prompt,
            padding='max_length',
            max_length=PROMPT_MAX_LENGTH,
            truncation=True,
            return_tensors='pt',
        )

        input_ids = tokens['input_ids'].to(self.train_device)
        attention_mask = tokens['attention_mask'].to(self.train_device)

        # Encode
        with torch.no_grad():
            encoder_output = model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Get last hidden state and crop
        hidden_state = encoder_output.hidden_states[-1]
        hidden_state = hidden_state[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:]

        return hidden_state
