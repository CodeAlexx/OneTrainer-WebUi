"""Shared enum definitions for the Serenity training framework."""

from __future__ import annotations

from enum import Enum


class GradientCheckpointingMethod(str, Enum):
    """Controls gradient checkpointing behavior."""

    OFF = "off"
    ON = "on"
    CPU_OFFLOADED = "cpu_offloaded"

    def enabled(self) -> bool:
        return self in (
            GradientCheckpointingMethod.ON,
            GradientCheckpointingMethod.CPU_OFFLOADED,
        )

    def offload(self) -> bool:
        return self == GradientCheckpointingMethod.CPU_OFFLOADED


class Optimizer(str, Enum):
    """Supported optimizer types."""

    # BNB Standard & 8-bit
    ADAGRAD = "ADAGRAD"
    ADAGRAD_8BIT = "ADAGRAD_8BIT"

    ADAM = "ADAM"
    ADAM_8BIT = "ADAM_8BIT"

    ADAMW = "ADAMW"
    ADAMW_8BIT = "ADAMW_8BIT"
    ADAMW_ADV = "ADAMW_ADV"

    ADEMAMIX = "AdEMAMix"
    ADEMAMIX_8BIT = "AdEMAMix_8BIT"
    SIMPLIFIED_ADEMAMIX = "SIMPLIFIED_AdEMAMix"

    ADOPT = "ADOPT"
    ADOPT_ADV = "ADOPT_ADV"

    LAMB = "LAMB"
    LAMB_8BIT = "LAMB_8BIT"

    LARS = "LARS"
    LARS_8BIT = "LARS_8BIT"

    LION = "LION"
    LION_8BIT = "LION_8BIT"
    LION_ADV = "LION_ADV"

    RMSPROP = "RMSPROP"
    RMSPROP_8BIT = "RMSPROP_8BIT"

    SGD = "SGD"
    SGD_8BIT = "SGD_8BIT"
    SIGNSGD_ADV = "SIGNSGD_ADV"

    # Schedule-free
    SCHEDULE_FREE_ADAMW = "SCHEDULE_FREE_ADAMW"
    SCHEDULE_FREE_SGD = "SCHEDULE_FREE_SGD"

    # DADAPT
    DADAPT_ADA_GRAD = "DADAPT_ADA_GRAD"
    DADAPT_ADAM = "DADAPT_ADAM"
    DADAPT_ADAN = "DADAPT_ADAN"
    DADAPT_LION = "DADAPT_LION"
    DADAPT_SGD = "DADAPT_SGD"

    # Prodigy
    PRODIGY = "PRODIGY"
    PRODIGY_PLUS_SCHEDULE_FREE = "PRODIGY_PLUS_SCHEDULE_FREE"
    PRODIGY_ADV = "PRODIGY_ADV"
    LION_PRODIGY_ADV = "LION_PRODIGY_ADV"

    # ADAFACTOR
    ADAFACTOR = "ADAFACTOR"

    # CAME
    CAME = "CAME"
    CAME_8BIT = "CAME_8BIT"

    # MUON
    MUON = "MUON"
    MUON_ADV = "MUON_ADV"
    ADAMUON_ADV = "ADAMUON_ADV"

    # Pytorch optimizers
    ADABELIEF = "ADABELIEF"
    TIGER = "TIGER"
    AIDA = "AIDA"
    YOGI = "YOGI"

    @property
    def is_adaptive(self) -> bool:
        return self in (
            Optimizer.DADAPT_SGD,
            Optimizer.DADAPT_ADAM,
            Optimizer.DADAPT_ADAN,
            Optimizer.DADAPT_ADA_GRAD,
            Optimizer.DADAPT_LION,
            Optimizer.PRODIGY,
            Optimizer.PRODIGY_PLUS_SCHEDULE_FREE,
            Optimizer.PRODIGY_ADV,
            Optimizer.LION_PRODIGY_ADV,
        )

    @property
    def is_schedule_free(self) -> bool:
        return self in (
            Optimizer.SCHEDULE_FREE_ADAMW,
            Optimizer.SCHEDULE_FREE_SGD,
            Optimizer.PRODIGY_PLUS_SCHEDULE_FREE,
        )

    def supports_fused_back_pass(self) -> bool:
        return self in (
            Optimizer.ADAFACTOR,
            Optimizer.CAME,
            Optimizer.CAME_8BIT,
            Optimizer.ADAM,
            Optimizer.ADAMW,
            Optimizer.ADAMW_ADV,
            Optimizer.ADOPT_ADV,
            Optimizer.SIMPLIFIED_ADEMAMIX,
            Optimizer.PRODIGY_PLUS_SCHEDULE_FREE,
            Optimizer.PRODIGY_ADV,
            Optimizer.LION_ADV,
            Optimizer.LION_PRODIGY_ADV,
            Optimizer.MUON_ADV,
            Optimizer.ADAMUON_ADV,
            Optimizer.SIGNSGD_ADV,
        )


class ModelFormat(str, Enum):
    """Model file format."""

    DIFFUSERS = "DIFFUSERS"
    CKPT = "CKPT"
    SAFETENSORS = "SAFETENSORS"
    LEGACY_SAFETENSORS = "LEGACY_SAFETENSORS"
    COMFY_LORA = "COMFY_LORA"
    INTERNAL = "INTERNAL"

    def file_extension(self) -> str:
        _map = {
            ModelFormat.DIFFUSERS: "",
            ModelFormat.CKPT: ".ckpt",
            ModelFormat.SAFETENSORS: ".safetensors",
            ModelFormat.LEGACY_SAFETENSORS: ".safetensors",
            ModelFormat.COMFY_LORA: ".safetensors",
            ModelFormat.INTERNAL: "",
        }
        return _map.get(self, "")

    def is_single_file(self) -> bool:
        return self.file_extension() != ""


class LossWeight(str, Enum):
    """Loss weighting function."""

    CONSTANT = "CONSTANT"
    P2 = "P2"
    MIN_SNR_GAMMA = "MIN_SNR_GAMMA"
    DEBIASED_ESTIMATION = "DEBIASED_ESTIMATION"
    SIGMA = "SIGMA"

    def supports_flow_matching(self) -> bool:
        return self in (LossWeight.CONSTANT, LossWeight.SIGMA)


class LossScaler(str, Enum):
    """Loss scaling mode."""

    NONE = "NONE"
    BATCH = "BATCH"
    GLOBAL_BATCH = "GLOBAL_BATCH"
    GRADIENT_ACCUMULATION = "GRADIENT_ACCUMULATION"
    BOTH = "BOTH"
    GLOBAL_BOTH = "GLOBAL_BOTH"

    def get_scale(self, batch_size: int, accumulation_steps: int, world_size: int = 1) -> int:
        _map = {
            LossScaler.NONE: 1,
            LossScaler.BATCH: batch_size,
            LossScaler.GLOBAL_BATCH: batch_size * world_size,
            LossScaler.GRADIENT_ACCUMULATION: accumulation_steps,
            LossScaler.BOTH: accumulation_steps * batch_size,
            LossScaler.GLOBAL_BOTH: accumulation_steps * batch_size * world_size,
        }
        if self not in _map:
            raise ValueError(f"Unknown loss scaler: {self}")
        return _map[self]


class DataType(str, Enum):
    """Model/training data types."""

    NONE = "NONE"
    FLOAT_8 = "FLOAT_8"
    FLOAT_16 = "FLOAT_16"
    FLOAT_32 = "FLOAT_32"
    BFLOAT_16 = "BFLOAT_16"
    TFLOAT_32 = "TFLOAT_32"
    INT_8 = "INT_8"
    NFLOAT_4 = "NFLOAT_4"
    FLOAT_W8A8 = "FLOAT_W8A8"
    INT_W8A8 = "INT_W8A8"
    GGUF = "GGUF"
    GGUF_A8_FLOAT = "GGUF_A8_FLOAT"
    GGUF_A8_INT = "GGUF_A8_INT"

    def torch_dtype(self, supports_quantization: bool = True):
        """Return the corresponding torch dtype, or None for quantized types."""
        import torch

        if self.is_quantized() and not supports_quantization:
            return torch.float16

        _map = {
            DataType.FLOAT_16: torch.float16,
            DataType.FLOAT_32: torch.float32,
            DataType.BFLOAT_16: torch.bfloat16,
            DataType.TFLOAT_32: torch.float32,
        }
        return _map.get(self)

    def enable_tf(self) -> bool:
        return self == DataType.TFLOAT_32

    def is_quantized(self) -> bool:
        return self in (
            DataType.FLOAT_8,
            DataType.INT_8,
            DataType.FLOAT_W8A8,
            DataType.INT_W8A8,
            DataType.NFLOAT_4,
        )

    def is_gguf(self) -> bool:
        return self in (DataType.GGUF, DataType.GGUF_A8_FLOAT, DataType.GGUF_A8_INT)


class LearningRateScheduler(str, Enum):
    """Learning rate scheduler type."""

    CONSTANT = "CONSTANT"
    LINEAR = "LINEAR"
    COSINE = "COSINE"
    COSINE_WITH_RESTARTS = "COSINE_WITH_RESTARTS"
    COSINE_WITH_HARD_RESTARTS = "COSINE_WITH_HARD_RESTARTS"
    REX = "REX"
    ADAFACTOR = "ADAFACTOR"
    CUSTOM = "CUSTOM"


class LearningRateScaler(str, Enum):
    """Learning rate scaling mode."""

    NONE = "NONE"
    BATCH = "BATCH"
    GLOBAL_BATCH = "GLOBAL_BATCH"
    GRADIENT_ACCUMULATION = "GRADIENT_ACCUMULATION"
    BOTH = "BOTH"
    GLOBAL_BOTH = "GLOBAL_BOTH"

    def get_scale(self, batch_size: int, accumulation_steps: int, world_size: int = 1) -> int:
        _map = {
            LearningRateScaler.NONE: 1,
            LearningRateScaler.BATCH: batch_size,
            LearningRateScaler.GLOBAL_BATCH: batch_size * world_size,
            LearningRateScaler.GRADIENT_ACCUMULATION: accumulation_steps,
            LearningRateScaler.BOTH: accumulation_steps * batch_size,
            LearningRateScaler.GLOBAL_BOTH: accumulation_steps * batch_size * world_size,
        }
        if self not in _map:
            raise ValueError(f"Unknown learning rate scaler: {self}")
        return _map[self]


class TimestepDistribution(str, Enum):
    """Timestep sampling distribution."""

    UNIFORM = "UNIFORM"
    SIGMOID = "SIGMOID"
    LOGIT_NORMAL = "LOGIT_NORMAL"
    HEAVY_TAIL = "HEAVY_TAIL"
    COS_MAP = "COS_MAP"
    INVERTED_PARABOLA = "INVERTED_PARABOLA"


class NoiseScheduler(str, Enum):
    """Noise scheduler for sampling."""

    DDIM = "DDIM"
    EULER = "EULER"
    EULER_A = "EULER_A"
    DPMPP = "DPMPP"
    DPMPP_SDE = "DPMPP_SDE"
    UNIPC = "UNIPC"
    EULER_KARRAS = "EULER_KARRAS"
    DPMPP_KARRAS = "DPMPP_KARRAS"
    DPMPP_SDE_KARRAS = "DPMPP_SDE_KARRAS"
    UNIPC_KARRAS = "UNIPC_KARRAS"


class EMAMode(str, Enum):
    """EMA weight averaging mode."""

    OFF = "OFF"
    GPU = "GPU"
    CPU = "CPU"


class ImageFormat(str, Enum):
    """Output image format."""

    PNG = "PNG"
    JPG = "JPG"

    def extension(self) -> str:
        _map = {ImageFormat.PNG: ".png", ImageFormat.JPG: ".jpg"}
        return _map.get(self, "")

    def pil_format(self) -> str:
        _map = {ImageFormat.PNG: "PNG", ImageFormat.JPG: "JPEG"}
        return _map.get(self, "")


class TimeUnit(str, Enum):
    """Time unit for training milestones."""

    NEVER = "NEVER"
    EPOCH = "EPOCH"
    STEP = "STEP"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    ALWAYS = "ALWAYS"


class PeftType(str, Enum):
    """PEFT adapter type."""

    LORA = "LORA"
    LOKR = "LOKR"


__all__ = [
    "GradientCheckpointingMethod",
    "Optimizer",
    "ModelFormat",
    "LossWeight",
    "LossScaler",
    "DataType",
    "LearningRateScheduler",
    "LearningRateScaler",
    "TimestepDistribution",
    "NoiseScheduler",
    "EMAMode",
    "ImageFormat",
    "TimeUnit",
    "PeftType",
]
