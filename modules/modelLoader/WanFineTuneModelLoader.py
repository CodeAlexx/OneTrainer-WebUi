from modules.model.WanModel import WanModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.wan.WanModelLoader import WanModelLoader
from modules.util.enum.ModelType import ModelType

WanFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.WAN_T2V: "resources/sd_model_spec/wan_t2v.json",
        ModelType.WAN_I2V: "resources/sd_model_spec/wan_i2v.json",
    },
    model_class=WanModel,
    model_loader_class=WanModelLoader,
    embedding_loader_class=None,  # TODO: Add embedding loader if needed
)
