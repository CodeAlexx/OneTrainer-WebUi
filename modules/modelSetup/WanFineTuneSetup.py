"""
WanFineTuneSetup - Full fine-tuning setup for Wan video models
"""

import copy

from modules.model.WanModel import WanModel
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection

from modules.util.TrainProgress import TrainProgress

import torch


class WanFineTuneSetup(BaseWanSetup):
    """Full fine-tuning setup for Wan video models."""
    
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

        # Text encoder parameters (if training)
        if config.text_encoder.train and model.text_encoder is not None:
            self._add_all_param_groups(
                model.text_encoder.parameters(),
                parameter_group_collection,
                config.text_encoder.learning_rate,
                "text_encoder"
            )

        # Transformer parameters
        if config.transformer.train:
            self._add_all_param_groups(
                model.transformer.parameters(),
                parameter_group_collection,
                config.transformer.learning_rate,
                "transformer"
            )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Configure requires_grad for all model parts."""
        # Freeze VAE always
        if model.vae is not None:
            model.vae.requires_grad_(False)
        if model.clip is not None:
            model.clip.requires_grad_(False)

        # Text encoder
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(config.text_encoder.train)

        # Transformer
        model.transformer.requires_grad_(config.transformer.train)

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        """Setup model for fine-tuning."""
        # Restore tokenizer
        model.tokenizer = copy.deepcopy(model.orig_tokenizer) if model.orig_tokenizer else None

        # Setup requires_grad
        self.__setup_requires_grad(model, config)

        # Initialize parameters
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
            config.text_encoder.train \
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
