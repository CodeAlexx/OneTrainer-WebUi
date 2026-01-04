"""
Wan LoRA Model Saver
"""

from modules.model.WanModel import WanModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch
import safetensors.torch


class WanLoRAModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: WanModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        # Save LoRA weights
        if model.transformer_lora is not None:
            lora_state_dict = model.transformer_lora.get_state_dict()
            
            # Convert to specified dtype if provided
            if dtype is not None:
                lora_state_dict = {k: v.to(dtype) for k, v in lora_state_dict.items()}
            
            # Add diffusion_model prefix for ComfyUI compatibility
            lora_state_dict = {'diffusion_model.' + k: v for k, v in lora_state_dict.items()}
            
            # Save as safetensors
            safetensors.torch.save_file(
                lora_state_dict,
                output_model_destination,
                metadata={'format': 'pt'}
            )

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
