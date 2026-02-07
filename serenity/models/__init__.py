"""Model registry for Serenity."""

from serenity.models.flux1 import Flux1Model
from serenity.models.flux_schnell import FluxSchnellModel
from serenity.models.flux2 import Flux2Model
from serenity.models.flux2_klein import Flux2KleinModel
from serenity.models.flux_klein import FluxKleinModel
from serenity.models.zimage import ZImageModel
from serenity.models.sd15 import SD15Model
from serenity.models.sd3 import SD3Model, SD35Model
from serenity.models.sdxl import SDXLModel
from serenity.models.qwen import QwenModel, QwenImageEditModel
from serenity.models.ltx2 import LTX2Model
from serenity.models.chroma import ChromaModel
from serenity.models.hunyuan_video import HunyuanVideoModel

__all__ = [
    "Flux1Model",
    "FluxSchnellModel",
    "Flux2Model",
    "Flux2KleinModel",
    "FluxKleinModel",
    "ZImageModel",
    "SD15Model",
    "SD3Model",
    "SD35Model",
    "SDXLModel",
    "QwenModel",
    "QwenImageEditModel",
    "LTX2Model",
    "ChromaModel",
    "HunyuanVideoModel",
]
