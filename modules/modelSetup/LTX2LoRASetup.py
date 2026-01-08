"""
LTX-2 LoRA Training Setup for OneTrainer

Configures LoRA training for LTX-2 video model.
"""
from modules.model.LTX2Model import LTX2Model
from modules.modelSetup.BaseLTX2Setup import BaseLTX2Setup
from modules.module.LoRAModule import create_peft_wrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class LTX2LoRASetup(BaseLTX2Setup):
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
            model: LTX2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        """Create parameter groups for optimizer."""
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection,
            "transformer_lora",
            model.transformer_lora,
            config.transformer
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        """Configure which parameters require gradients."""
        # Freeze base models
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        if model.vae_encoder is not None:
            model.vae_encoder.requires_grad_(False)
        if model.vae_decoder is not None:
            model.vae_decoder.requires_grad_(False)

        # Only train LoRA parameters
        self._setup_model_part_requires_grad(
            "transformer_lora",
            model.transformer_lora,
            config.transformer,
            model.train_progress
        )

    def setup_model(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        """Setup LoRA adapters on the transformer."""
        # LTX-2 transformer attention layers to target
        # Based on ltx-trainer's LoRA config:
        # target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
        layer_filter = config.layer_filter.split(",") if config.layer_filter else []

        model.transformer_lora = create_peft_wrapper(
            model.transformer,
            "transformer",
            config,
            layer_filter
        )

        # Load existing LoRA state if provided
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

        # Debug: verify LoRA parameters
        adapter_params = list(model.transformer_lora.parameters())
        print(f"[LTX2] LoRA adapter ({config.peft_type.value}) params count: {len(adapter_params)}")
        if adapter_params:
            total_params = sum(p.numel() for p in adapter_params)
            print(f"[LTX2] Total LoRA parameters: {total_params:,}")
            print(f"[LTX2] First param requires_grad: {adapter_params[0].requires_grad}, "
                  f"dtype: {adapter_params[0].dtype}, device: {adapter_params[0].device}")

    def setup_train_device(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        """Move model components to appropriate devices."""
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        # Move text encoder
        if model.text_encoder is not None:
            model.text_encoder_to(
                self.train_device if text_encoder_on_train_device else self.temp_device
            )
            model.text_encoder.eval()

        # Move VAE
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        if model.vae_encoder is not None:
            model.vae_encoder.eval()
        if model.vae_decoder is not None:
            model.vae_decoder.eval()

        # Move transformer to train device
        model.transformer_to(self.train_device)

        # Set transformer to train mode if training
        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: LTX2Model,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        """Called after each optimizer step."""
        self.__setup_requires_grad(model, config)


class LTX2FineTuneSetup(BaseLTX2Setup):
    """Full fine-tuning setup for LTX-2 (not recommended for 2B model)."""

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
            model: LTX2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection,
            "transformer",
            model.transformer,
            config.transformer
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        # Freeze non-training components
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        if model.vae_encoder is not None:
            model.vae_encoder.requires_grad_(False)
        if model.vae_decoder is not None:
            model.vae_decoder.requires_grad_(False)

        # Enable gradients for transformer
        self._setup_model_part_requires_grad(
            "transformer",
            model.transformer,
            config.transformer,
            model.train_progress
        )

    def setup_model(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

        # Debug output
        trainable_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.transformer.parameters())
        print(f"[LTX2] Fine-tune: {trainable_params:,} / {total_params:,} parameters trainable")

    def setup_train_device(
            self,
            model: LTX2Model,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        if model.text_encoder is not None:
            model.text_encoder_to(
                self.train_device if text_encoder_on_train_device else self.temp_device
            )
            model.text_encoder.eval()

        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)

        model.transformer_to(self.train_device)

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: LTX2Model,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)
