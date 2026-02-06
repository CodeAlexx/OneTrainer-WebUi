"""Model registry for EriTrainer."""

from eritrainer.models.flux1 import Flux1Model
from eritrainer.models.flux2 import Flux2Model
from eritrainer.models.flux2_klein import Flux2KleinModel
from eritrainer.models.flux_klein import FluxKleinModel
from eritrainer.models.zimage import ZImageModel
from eritrainer.models.sd15 import SD15Model
from eritrainer.models.sd3 import SD3Model, SD35Model
from eritrainer.models.sdxl import SDXLModel
from eritrainer.models.ltx2 import LTX2Model

__all__ = [
    "Flux1Model",
    "Flux2Model",
    "Flux2KleinModel",
    "FluxKleinModel",
    "ZImageModel",
    "SD15Model",
    "SD3Model",
    "SD35Model",
    "SDXLModel",
    "LTX2Model",
]
