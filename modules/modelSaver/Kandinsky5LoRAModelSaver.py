"""
Kandinsky 5 LoRA Model Saver

Saves LoRA adapters for Kandinsky 5 models.
"""

import os
from safetensors.torch import save_file

from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.enum.ModelType import ModelType


class Kandinsky5LoRAModelSaver(BaseModelSaver):
    """Saver for Kandinsky 5 LoRA adapters."""

    def save(
            self,
            model: Kandinsky5Model,
            path: str,
            metadata: dict = None,
    ):
        """
        Save LoRA adapters to a safetensors file.
        
        Args:
            model: The Kandinsky5Model with LoRA adapters
            path: Destination path for the saved file
            metadata: Optional metadata to include
        """
        state_dict = {}
        
        # Collect adapter weights if available
        if hasattr(model, 'transformer_lora') and model.transformer_lora is not None:
            state_dict.update(model.transformer_lora.state_dict())
            
        if hasattr(model, 'text_encoder_lora') and model.text_encoder_lora is not None:
            state_dict.update(model.text_encoder_lora.state_dict())
            
        if not state_dict:
            print("Warning: No LoRA weights to save")
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(state_dict, path)
