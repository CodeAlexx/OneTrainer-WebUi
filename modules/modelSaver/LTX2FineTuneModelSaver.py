"""
LTX-2 Fine Tune Model Saver for OneTrainer

Saves fine-tuned LTX-2 models.
"""
from modules.model.LTX2Model import LTX2Model
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch
from safetensors.torch import save_file


class LTX2FineTuneModelSaver(
    BaseModelSaver,
    InternalModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: LTX2Model,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        # Get transformer state dict
        state_dict = {}
        if model.transformer is not None:
            transformer_state = model.transformer.state_dict()
            if dtype is not None:
                transformer_state = {k: v.to(dtype) for k, v in transformer_state.items()}
            state_dict.update(transformer_state)

        # Save based on format
        if output_model_format == ModelFormat.SAFETENSORS:
            save_file(state_dict, output_model_destination)
        elif output_model_format == ModelFormat.INTERNAL:
            save_file(state_dict, output_model_destination)
            self._save_internal_data(model, output_model_destination)
