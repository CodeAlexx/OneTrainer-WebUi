"""Stable Diffusion 3 / 3.5 native model adapters."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

import torch

_PROMPT_MAX_LENGTH = 256
_REQUIRED_SUBFOLDERS: dict[str, tuple[str, ...]] = {
    "scheduler": ("scheduler_config.json", "config.json"),
    "tokenizer": ("tokenizer_config.json", "tokenizer.json"),
    "tokenizer_2": ("tokenizer_config.json", "tokenizer.json"),
    "tokenizer_3": ("tokenizer_config.json", "tokenizer.json"),
    "text_encoder": ("config.json",),
    "text_encoder_2": ("config.json",),
    "text_encoder_3": ("config.json",),
    "transformer": ("config.json",),
    "vae": ("config.json",),
}
_WEIGHT_MARKERS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".msgpack",
)


def _validate_sd3_model_path(model_path: str) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"SD3 model path does not exist: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"SD3 model path must be a directory: {root}")

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
            f"SD3 model at {root} is missing required components: {', '.join(sorted(missing_subfolders))}"
        )
    if invalid_components:
        raise FileNotFoundError(
            f"SD3 model at {root} has invalid components (missing config files): "
            f"{', '.join(sorted(invalid_components))}"
        )

    return root


def _component_has_weights(component_dir: Path) -> bool:
    return any(entry.is_file() and entry.name.endswith(_WEIGHT_MARKERS) for entry in component_dir.iterdir())


def _link_file(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with suppress(Exception):
        target.symlink_to(source)
        return
    if source.is_file():
        target.write_bytes(source.read_bytes())


def _resolve_transformer_checkpoint_dir(transformer_dir: Path) -> Path:
    if any((transformer_dir / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")):
        return transformer_dir
    if any((transformer_dir / filename).exists() for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json")):
        return transformer_dir

    single_diffusers = transformer_dir / "diffusion_pytorch_model.safetensors"
    sharded_diffusers_index = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not single_diffusers.exists() and not sharded_diffusers_index.exists():
        return transformer_dir

    token = hashlib.sha1(str(transformer_dir).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    compat_dir = Path.home() / ".cache" / "serenity" / "sd3_transformer_compat" / token
    compat_dir.mkdir(parents=True, exist_ok=True)

    if single_diffusers.exists():
        _link_file(single_diffusers, compat_dir / "model.safetensors")
        return compat_dir

    _link_file(sharded_diffusers_index, compat_dir / "model.safetensors.index.json")

    try:
        index_data = json.loads(sharded_diffusers_index.read_text())
        shard_names = sorted(set(index_data.get("weight_map", {}).values()))
    except Exception:
        shard_names = []

    for shard_name in shard_names:
        source = transformer_dir / shard_name
        if source.exists():
            _link_file(source, compat_dir / shard_name)

    return compat_dir


def _load_transformer(
    model_root: Path,
    dtype: torch.dtype,
):
    from diffusers import SD3Transformer2DModel

    from accelerate import init_empty_weights
    from accelerate.utils import load_checkpoint_in_model

    transformer_dir = model_root / "transformer"
    if not _component_has_weights(transformer_dir):
        raise FileNotFoundError(f"No transformer weights found under {transformer_dir}")
    checkpoint_dir = _resolve_transformer_checkpoint_dir(transformer_dir)

    config = SD3Transformer2DModel.load_config(
        str(model_root),
        subfolder="transformer",
        local_files_only=True,
    )

    with init_empty_weights():
        transformer = SD3Transformer2DModel.from_config(config)

    # Direct checkpoint load avoids the temporary duplicate allocations from plain from_pretrained.
    load_checkpoint_in_model(
        transformer,
        checkpoint=str(checkpoint_dir),
        device_map={"": "cpu"},
        dtype=dtype,
        strict=False,
    )
    transformer.to(dtype=dtype)
    return transformer


class SD3Model(BaseModelImpl):
    """Native SD3 model behavior used by Serenity training paths."""

    family = "sd3"
    resolution_multiple = 16
    train_module_attr = "transformer"
    flow_objective = True

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SD3)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer

        model_root = _validate_sd3_model_path(model_path)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_root),
            subfolder="scheduler",
            local_files_only=True,
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer",
            local_files_only=True,
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer_2",
            local_files_only=True,
        )
        tokenizer_3 = T5Tokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer_3",
            local_files_only=True,
            use_fast=False,
        )

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            str(model_root),
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(model_root),
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            local_files_only=True,
        )
        text_encoder_3 = T5EncoderModel.from_pretrained(
            str(model_root),
            subfolder="text_encoder_3",
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

        pipeline = StableDiffusion3Pipeline(
            scheduler=scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
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
        encoded = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=_PROMPT_MAX_LENGTH,
        )
        values = list(encoded) if isinstance(encoded, tuple | list) else [encoded]
        prompt_embeds = values[0]
        pooled_prompt_embeds: torch.Tensor | None = None

        for value in values[1:]:
            if not torch.is_tensor(value):
                continue
            if pooled_prompt_embeds is None and value.dim() == 2 and value.is_floating_point():
                pooled_prompt_embeds = value
                continue

        return prompt_embeds, pooled_prompt_embeds, None

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        if train_device.type == "cuda":
            # Keep SD3 text encoders on CPU during cache generation to preserve VRAM for transformer training.
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        if device.type == "cuda":
            # Avoid loading the full SD3 text stack onto VRAM during cache generation.
            return
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, attr, None)
            if encoder is None:
                continue
            encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipeline, attr, None)
            if encoder is None:
                continue
            with suppress(Exception):
                encoder.to("cpu")


class SD35Model(SD3Model):
    """SD3.5 variant that shares SD3 loading/encoding behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.model_type = ModelType.SD35
