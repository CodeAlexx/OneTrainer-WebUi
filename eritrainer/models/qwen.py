"""Qwen Image native model adapters."""

from __future__ import annotations

import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl

import torch

_PROMPT_MAX_LENGTH = 512
_REQUIRED_SUBFOLDERS: dict[str, tuple[str, ...]] = {
    "scheduler": ("scheduler_config.json", "config.json"),
    "tokenizer": ("tokenizer_config.json", "tokenizer.json"),
    "text_encoder": ("config.json",),
    "vae": ("config.json",),
    "transformer": ("config.json",),
}
_WEIGHT_MARKERS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".msgpack",
)


def _validate_qwen_model_path(model_path: str) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Qwen model path does not exist: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen model path must be a directory: {root}")

    missing_subfolders: list[str] = []
    invalid_components: list[str] = []
    for subfolder, required_files in _REQUIRED_SUBFOLDERS.items():
        component_path = root / subfolder
        if not component_path.exists():
            missing_subfolders.append(subfolder)
            continue
        if not any((component_path / required).exists() for required in required_files):
            invalid_components.append(subfolder)

    if missing_subfolders:
        raise FileNotFoundError(
            f"Qwen model at {root} is missing required components: {', '.join(sorted(missing_subfolders))}"
        )
    if invalid_components:
        raise FileNotFoundError(
            f"Qwen model at {root} has invalid components (missing config files): "
            f"{', '.join(sorted(invalid_components))}"
        )

    return root


def _component_has_weights(component_dir: Path) -> bool:
    return any(entry.is_file() and entry.name.endswith(_WEIGHT_MARKERS) for entry in component_dir.iterdir())


def _load_pipeline_from_candidates(
    model_path: str,
    candidates: tuple[str, ...],
    dtype: torch.dtype,
):
    import diffusers

    load_errors: list[str] = []
    for class_name in candidates:
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            load_errors.append(f"{class_name}: unavailable")
            continue
        try:
            pipeline = pipeline_cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
            return pipeline
        except Exception as exc:  # pragma: no cover - environment-dependent
            load_errors.append(f"{class_name}: {exc}")

    details = " | ".join(load_errors) if load_errors else "no pipeline candidates"
    raise RuntimeError(f"Could not load Qwen pipeline from {model_path}: {details}")


class QwenBaseModel(BaseModelImpl):
    """Shared native Qwen behavior for text-to-image and edit variants."""

    family = "qwen"
    resolution_multiple = 64
    train_module_attr = "transformer"
    flow_objective = True

    _INT8_ALIASES = {
        "int8",
        "int_8",
        "int-8",
        "int",
        "int_w8a8",
        "intw8a8",
        "w8a8_int",
        "w8a8",
    }
    _FP8_ALIASES = {
        "fp8",
        "fp_8",
        "fp-8",
        "float8",
        "float_8",
        "float-8",
        "float_w8a8",
        "fpw8a8",
        "w8a8_float",
    }

    @classmethod
    def normalize_quantization_mode(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"", "none", "off", "false", "null"}:
            return None
        if normalized in cls._INT8_ALIASES:
            return "int8"
        if normalized in cls._FP8_ALIASES:
            return "fp8"
        return normalized

    @staticmethod
    def is_dispatched_module(module: Any) -> bool:
        device_map = getattr(module, "hf_device_map", None)
        return isinstance(device_map, dict) and len(device_map) > 0

    @staticmethod
    def build_bnb_max_memory(
        train_device: torch.device,
        requested_gpu_budget_gib: int | None = None,
    ) -> dict[Any, str]:
        if train_device.type != "cuda" or not torch.cuda.is_available():
            return {"cpu": "120GiB"}

        device_index = train_device.index if train_device.index is not None else 0
        props = torch.cuda.get_device_properties(device_index)
        total_gib = max(1, int(props.total_memory // (1024**3)))
        if requested_gpu_budget_gib is not None:
            gpu_budget_gib = max(6, min(total_gib - 2, int(requested_gpu_budget_gib)))
        else:
            gpu_budget_gib = max(8, min(total_gib - 4, int(total_gib * 0.45)))
        return {
            device_index: f"{gpu_budget_gib}GiB",
            "cpu": "120GiB",
        }

    def _pipeline_candidates(self) -> tuple[str, ...]:
        if self.model_type == ModelType.QWEN_IMAGE_EDIT:
            return ("QwenImageImg2ImgPipeline", "QwenImageEditPipeline", "QwenImagePipeline")
        return ("QwenImagePipeline", "QwenImageImg2ImgPipeline")

    def _build_component_pipeline(
        self,
        *,
        model_root: Path,
        scheduler: Any,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        transformer: Any,
    ):
        import diffusers

        if self.model_type == ModelType.QWEN_IMAGE_EDIT:
            if hasattr(diffusers, "QwenImageImg2ImgPipeline"):
                pipeline_cls = diffusers.QwenImageImg2ImgPipeline
                return pipeline_cls(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                )

            # QwenImageEditPipeline requires a processor; only use it when local files provide one.
            if hasattr(diffusers, "QwenImageEditPipeline") and (model_root / "processor").exists():
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    str(model_root),
                    subfolder="processor",
                    local_files_only=True,
                )
                pipeline_cls = diffusers.QwenImageEditPipeline
                return pipeline_cls(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    processor=processor,
                    transformer=transformer,
                )

        if hasattr(diffusers, "QwenImagePipeline"):
            pipeline_cls = diffusers.QwenImagePipeline
            return pipeline_cls(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )

        raise RuntimeError("No compatible Qwen pipeline class is available in this diffusers build.")

    def _build_quantization_configs(self, quantization_mode: str) -> tuple[Any, Any]:
        if quantization_mode == "int8":
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError("Qwen INT8 mode requires bitsandbytes installed in the active environment.") from exc

            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

            return (
                TransformersBitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
                DiffusersBitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
            )

        if quantization_mode == "fp8":
            try:
                from diffusers import QuantoConfig as DiffusersQuantoConfig
                from transformers import QuantoConfig as TransformersQuantoConfig
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "Qwen FP8 mode requires Quanto support in diffusers/transformers."
                ) from exc

            return (
                TransformersQuantoConfig(weights="float8"),
                DiffusersQuantoConfig(weights_dtype="float8"),
            )

        raise ValueError(f"Unsupported Qwen quantization mode: {quantization_mode}")

    def _load_pipeline_quantized(
        self,
        model_root: Path,
        dtype: torch.dtype,
        train_device: torch.device,
        *,
        quantization_mode: str,
        requested_gpu_budget_gib: int | None = None,
    ):
        from diffusers import (
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
            QwenImageTransformer2DModel,
        )
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2Tokenizer,
        )

        if not _component_has_weights(model_root / "transformer"):
            raise FileNotFoundError(f"No transformer weights found under {model_root / 'transformer'}")
        if not _component_has_weights(model_root / "text_encoder"):
            raise FileNotFoundError(f"No text-encoder weights found under {model_root / 'text_encoder'}")

        text_quantization_config, transformer_quantization_config = self._build_quantization_configs(quantization_mode)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_root),
            subfolder="scheduler",
            local_files_only=True,
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer",
            local_files_only=True,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_root),
            subfolder="text_encoder",
            quantization_config=text_quantization_config,
            device_map={"": "cpu"},
            local_files_only=True,
        )
        vae = AutoencoderKLQwenImage.from_pretrained(
            str(model_root),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            str(model_root),
            subfolder="transformer",
            torch_dtype=dtype,
            quantization_config=transformer_quantization_config,
            device_map="auto",
            max_memory=self.build_bnb_max_memory(
                train_device,
                requested_gpu_budget_gib=requested_gpu_budget_gib,
            ),
            local_files_only=True,
        )

        return self._build_component_pipeline(
            model_root=model_root,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

    def _load_pipeline_int8(
        self,
        model_root: Path,
        dtype: torch.dtype,
        train_device: torch.device,
        requested_gpu_budget_gib: int | None = None,
    ):
        return self._load_pipeline_quantized(
            model_root,
            dtype,
            train_device,
            quantization_mode="int8",
            requested_gpu_budget_gib=requested_gpu_budget_gib,
        )

    def _load_pipeline_fp8(
        self,
        model_root: Path,
        dtype: torch.dtype,
        train_device: torch.device,
        requested_gpu_budget_gib: int | None = None,
    ):
        return self._load_pipeline_quantized(
            model_root,
            dtype,
            train_device,
            quantization_mode="fp8",
            requested_gpu_budget_gib=requested_gpu_budget_gib,
        )

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        *,
        quantization_mode: str | None = None,
        requested_gpu_budget_gib: int | None = None,
        **_: Any,
    ):
        model_root = _validate_qwen_model_path(model_path)
        normalized_quantization = self.normalize_quantization_mode(quantization_mode)
        if normalized_quantization == "int8":
            return self._load_pipeline_int8(
                model_root,
                dtype,
                train_device,
                requested_gpu_budget_gib=requested_gpu_budget_gib,
            )
        if normalized_quantization == "fp8":
            return self._load_pipeline_fp8(
                model_root,
                dtype,
                train_device,
                requested_gpu_budget_gib=requested_gpu_budget_gib,
            )

        if normalized_quantization:
            print(
                f"[native/diffusion] warning: unsupported Qwen quantization mode "
                f"'{normalized_quantization}', loading unquantized"
            )

        pipeline = _load_pipeline_from_candidates(
            str(model_root),
            self._pipeline_candidates(),
            dtype=dtype,
        )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)

        z_dim = int(getattr(pipeline.vae.config, "z_dim", latents.shape[1]))
        latents_mean = torch.tensor(
            pipeline.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            pipeline.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
        conditioning_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        params = inspect.signature(pipeline.encode_prompt).parameters
        prompt_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "device": device,
            "num_images_per_prompt": 1,
            "max_sequence_length": _PROMPT_MAX_LENGTH,
        }
        if conditioning_image is not None and "image" in params:
            prompt_kwargs["image"] = conditioning_image.to(device=device, dtype=torch.float32)

        prompt_embeds, prompt_mask = pipeline.encode_prompt(**prompt_kwargs)
        return prompt_embeds, None, prompt_mask

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None and self.is_dispatched_module(text_encoder):
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        if self.is_dispatched_module(text_encoder):
            return
        text_encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None or self.is_dispatched_module(text_encoder):
            return
        with suppress(Exception):
            text_encoder.to("cpu")

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = latents.shape
        if frames != 1:
            raise ValueError(f"Qwen latent frames must be 1, got {frames}")

        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        return latents

    @staticmethod
    def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, height * 2, width * 2)
        return latents


class QwenModel(QwenBaseModel):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.QWEN)


class QwenImageEditModel(QwenBaseModel):
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.QWEN_IMAGE_EDIT)
