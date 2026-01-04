from modules.model.BaseModel import BaseModel
from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class WanLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        # Wan uses peft-style LoRA, no key conversion needed
        return None

    def load(
            self,
            model: WanModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
