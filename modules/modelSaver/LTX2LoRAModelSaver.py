"""
LTX-2 LoRA Model Saver for OneTrainer

Saves LoRA models trained on LTX-2.
"""
from modules.model.LTX2Model import LTX2Model
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.ltx2.LTX2LoRASaver import LTX2LoRASaver
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch


class LTX2LoRAModelSaver(
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
        lora_model_saver = LTX2LoRASaver()

        lora_model_saver.save(model, output_model_format, output_model_destination, dtype)
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)
