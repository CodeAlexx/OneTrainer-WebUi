from modules.model.QwenImageEditModel import QwenImageEditModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.qwen.QwenLoRALoader import QwenLoRALoader
from modules.modelLoader.qwen.QwenImageEditModelLoader import QwenImageEditModelLoader
from modules.util.enum.ModelType import ModelType

QwenImageEditLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.QWEN_IMAGE_EDIT: "resources/sd_model_spec/qwen-image-edit-lora.json"},
    model_class=QwenImageEditModel,
    model_loader_class=QwenImageEditModelLoader,
    embedding_loader_class=None,
    lora_loader_class=QwenLoRALoader,  # Reuse existing Qwen LoRA loader
)
