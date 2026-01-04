"""
Wan Fine-Tune Model Saver
"""

from modules.model.WanModel import WanModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch
import safetensors.torch


class WanFineTuneModelSaver(
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
        # Save full model weights
        if model.transformer is not None:
            state_dict = {}
            for name, param in model.transformer.named_parameters():
                if dtype is not None:
                    state_dict[name] = param.detach().to(dtype)
                else:
                    state_dict[name] = param.detach()
            
            # Save as safetensors
            safetensors.torch.save_file(
                state_dict,
                output_model_destination,
                metadata={'format': 'pt'}
            )

        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
