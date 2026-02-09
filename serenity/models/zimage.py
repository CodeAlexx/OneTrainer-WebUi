"""Native Z-Image model adapter."""

from __future__ import annotations

import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

import torch

_PROMPT_MAX_LENGTH = 512
_REQUIRED_SUBFOLDERS: dict[str, tuple[str, ...]] = {
    "tokenizer": ("tokenizer_config.json", "tokenizer.json"),
    "text_encoder": ("config.json",),
    "vae": ("config.json",),
    "transformer": ("config.json",),
}
_OPTIONAL_SUBFOLDERS: dict[str, tuple[str, ...]] = {
    "scheduler": ("scheduler_config.json", "config.json"),
}
_WEIGHT_MARKERS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".msgpack",
)


def _validate_zimage_model_path(model_path: str) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Z-Image model path does not exist: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Z-Image model path must be a directory: {root}")

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
            f"Z-Image model at {root} is missing required components: {', '.join(sorted(missing_subfolders))}"
        )
    if invalid_components:
        raise FileNotFoundError(
            f"Z-Image model at {root} has invalid components (missing config files): "
            f"{', '.join(sorted(invalid_components))}"
        )

    optional_invalid_components: list[str] = []
    for subfolder, required_files in _OPTIONAL_SUBFOLDERS.items():
        component_path = root / subfolder
        if not component_path.exists():
            continue
        if not any((component_path / required).exists() for required in required_files):
            optional_invalid_components.append(subfolder)

    if optional_invalid_components:
        raise FileNotFoundError(
            f"Z-Image model at {root} has invalid optional components (missing config files): "
            f"{', '.join(sorted(optional_invalid_components))}"
        )

    return root


def _component_has_weights(component_dir: Path) -> bool:
    return any(entry.is_file() and entry.name.endswith(_WEIGHT_MARKERS) for entry in component_dir.iterdir())


def _load_transformer(
    model_root: Path,
    dtype: torch.dtype,
):
    from diffusers import ZImageTransformer2DModel

    from accelerate import init_empty_weights
    from accelerate.utils import load_checkpoint_in_model

    transformer_dir = model_root / "transformer"
    if not _component_has_weights(transformer_dir):
        raise FileNotFoundError(f"No transformer weights found under {transformer_dir}")

    config = ZImageTransformer2DModel.load_config(
        str(model_root),
        subfolder="transformer",
        local_files_only=True,
    )

    with init_empty_weights():
        transformer = ZImageTransformer2DModel.from_config(config)

    # Direct checkpoint load avoids the temporary duplicate allocations from plain from_pretrained.
    load_checkpoint_in_model(
        transformer,
        checkpoint=str(transformer_dir),
        device_map={"": "cpu"},
        dtype=dtype,
        strict=False,
    )
    transformer.to(dtype=dtype)
    return transformer


def _format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    params = inspect.signature(tokenizer.apply_chat_template).parameters
    call_kwargs: dict[str, Any] = {"tokenize": False}
    if "add_generation_prompt" in params:
        call_kwargs["add_generation_prompt"] = True
    if "enable_thinking" in params:
        call_kwargs["enable_thinking"] = True

    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, **call_kwargs)
        if isinstance(formatted, str) and formatted.strip():
            return formatted
    except (RuntimeError, ValueError, TypeError):
        import logging
        logging.debug("Failed to format prompt with chat template, using raw prompt")

    return prompt


class ZImageModel(BaseModelImpl):
    """Native Z-Image behavior used by Serenity training paths."""

    family = "zimage"
    resolution_multiple = 64
    train_module_attr = "transformer"
    flow_objective = True

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.ZIMAGE)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_root = _validate_zimage_model_path(model_path)

        scheduler_dir = model_root / "scheduler"
        if scheduler_dir.exists():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(model_root),
                subfolder="scheduler",
                local_files_only=True,
            )
        else:
            # Some local HF snapshots (non-turbo variants) omit scheduler config;
            # use local default scheduler construction in that case.
            scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer",
            local_files_only=True,
            use_fast=False,
        )
        text_encoder = AutoModelForCausalLM.from_pretrained(
            str(model_root),
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(
            str(model_root),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        transformer = _load_transformer(model_root, dtype)

        pipeline = ZImagePipeline(
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
        )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        shift_factor = float(getattr(pipeline.vae.config, "shift_factor", 0.0))
        scaling_factor = float(getattr(pipeline.vae.config, "scaling_factor", 1.0))
        return (latents - shift_factor) * scaling_factor

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        tokenizer = getattr(pipeline, "tokenizer", None)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            raise RuntimeError("Z-Image pipeline is missing tokenizer or text_encoder")

        if text_encoder.device != device:
            text_encoder.to(device)

        formatted_prompt = _format_chat_prompt(tokenizer, prompt)
        tokens = tokenizer(
            [formatted_prompt],
            max_length=_PROMPT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) >= 2:
            embeddings = hidden_states[-2]
        else:
            embeddings = output.last_hidden_state

        token_mask = attention_mask.bool()[0]
        prompt_embeds = embeddings[0][token_mask]
        return prompt_embeds, None, None

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        if train_device.type == "cuda":
            # Keep Qwen text encoding on CPU during cache generation to preserve VRAM for transformer training.
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        if device.type == "cuda":
            # Z-Image text encoder is large; prefer CPU caching for stable 24GB operation.
            return
        encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        with suppress(Exception):
            encoder.to("cpu")
