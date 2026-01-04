"""
WanLoRASetup - LoRA training setup for Wan video models

Sets up LoRA adapters on the Wan transformer for training.
"""

import copy

from modules.model.WanModel import WanModel
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.module.LoRAModule import create_peft_wrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection

from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch


class WanLoRASetup(BaseWanSetup):
    """LoRA training setup for Wan video models."""
    
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
            model: WanModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        """Create parameter groups for optimizer."""
        parameter_group_collection = NamedParameterGroupCollection()

        # Text encoder LoRA (if enabled)
        self._create_model_part_parameters(
            parameter_group_collection,
            "text_encoder_lora",
            model.text_encoder_lora,
            config.text_encoder
        )

        # Transformer LoRA
        self._create_model_part_parameters(
            parameter_group_collection,
            "transformer_lora",
            model.transformer_lora,
            config.transformer
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Configure requires_grad for all model parts."""
        # Freeze base models
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        if model.vae is not None:
            model.vae.requires_grad_(False)
        if model.clip is not None:
            model.clip.requires_grad_(False)
        model.transformer.requires_grad_(False)

        # Setup LoRA requires_grad
        self._setup_model_part_requires_grad(
            "text_encoder_lora",
            model.text_encoder_lora,
            config.text_encoder,
            model.train_progress
        )
        self._setup_model_part_requires_grad(
            "transformer_lora",
            model.transformer_lora,
            config.transformer,
            model.train_progress
        )

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Setup LoRA adapters on the model."""
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te")

        # Create text encoder LoRA if configured
        if model.text_encoder is not None and create_te:
            model.text_encoder_lora = create_peft_wrapper(
                model.text_encoder, "lora_te", config
            )
        else:
            model.text_encoder_lora = None

        # Create transformer LoRA - target WanAttentionBlock layers
        model.transformer_lora = create_peft_wrapper(
            model.transformer,
            "lora_transformer",
            config,
            config.layer_filter.split(",") if config.layer_filter else None
        )

        # Load existing LoRA state dict if provided
        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        # Configure text encoder LoRA
        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        # Configure transformer LoRA
        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        # Restore tokenizer
        model.tokenizer = copy.deepcopy(model.orig_tokenizer) if model.orig_tokenizer else None

        # Setup requires_grad
        self.__setup_requires_grad(model, config)

        # Initialize parameters
        from modules.util.optimizer_util import init_model_parameters
        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Move model components to appropriate devices."""
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        # Set training/eval mode
        if model.text_encoder is not None:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        if model.vae is not None:
            model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: WanModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        """Called after each optimizer step."""
        self.__setup_requires_grad(model, config)
