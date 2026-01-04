
import os
import torch
from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.util.enum.ModelType import ModelType

class Kandinsky5FineTuneModelSaver(BaseModelSaver):
    def __init__(self):
        super().__init__()

    def save_model(self, model: Kandinsky5Model, path: str, metadata: dict = None):
        # Create directory structure similar to loading
        os.makedirs(path, exist_ok=True)
        
        # 1. Save Text Encoders
        if model.text_encoder_qwen:
            model.text_encoder_qwen.save_pretrained(os.path.join(path, "text_encoder"))
            if model.tokenizer_qwen:
                model.tokenizer_qwen.save_pretrained(os.path.join(path, "tokenizer"))
                
        if model.text_encoder_clip:
            model.text_encoder_clip.save_pretrained(os.path.join(path, "text_encoder_2"))
            if model.tokenizer_clip:
                model.tokenizer_clip.save_pretrained(os.path.join(path, "tokenizer_2"))

        # 2. Save VAE
        if model.vae:
            model.vae.save_pretrained(os.path.join(path, "vae"))

        # 3. Save Transformer
        if model.transformer:
            # If it's a diffusers model or has save_pretrained
            if hasattr(model.transformer, "save_pretrained"):
                model.transformer.save_pretrained(os.path.join(path, "transformer")) # or .
            else:
                # Fallback to torch save
                torch.save(model.transformer.state_dict(), os.path.join(path, "transformer_state_dict.pt"))
                
        # 4. Save Scheduler
        if model.noise_scheduler:
            model.noise_scheduler.save_pretrained(os.path.join(path, "scheduler"))

    @staticmethod
    def get_model_types() -> list[ModelType]:
        return [ModelType.KANDINSKY_5]
