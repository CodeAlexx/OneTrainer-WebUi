"""Native Flux 1 model adapters."""

from __future__ import annotations

import inspect
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

import torch


def _validate_flux_path(model_path: str) -> None:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Flux model path does not exist: {path}")
    if path.is_file():
        return

    required = ("transformer", "vae")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Flux model at {path} is missing required components: {', '.join(missing)}")


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
            return pipeline_cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            load_errors.append(f"{class_name}: {exc}")

    details = " | ".join(load_errors) if load_errors else "no pipeline candidates"
    raise RuntimeError(f"Could not load Flux pipeline from {model_path}: {details}")


def _resolve_snapshot_path(path: str | Path) -> Path | None:
    resolved = Path(path).expanduser()

    # Resolve an HF repo-id directory by `refs/main` to current snapshot.
    refs_main = resolved / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = resolved / "snapshots" / revision
        if snapshot.exists():
            return snapshot

    snapshots_dir = resolved / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
    if snapshots:
        return snapshots[-1]

    if resolved.exists():
        return resolved

    return None


def _resolve_fallback_t5_path() -> Path | None:
    env_path = os.environ.get("SERENITY_FLUX_T5_PATH")
    if env_path:
        snapshot = _resolve_snapshot_path(env_path)
        if snapshot is not None:
            return snapshot

    candidates = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--google--t5-v1_1-xxl",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--google--flan-t5-xl",
    )
    for candidate in candidates:
        snapshot = _resolve_snapshot_path(candidate)
        if snapshot is not None:
            return snapshot
    return None


def _resolve_fallback_clip_path() -> Path | None:
    env_path = os.environ.get("SERENITY_FLUX_CLIP_PATH")
    if env_path:
        snapshot = _resolve_snapshot_path(env_path)
        if snapshot is not None:
            return snapshot

    candidates = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32",
    )
    for candidate in candidates:
        snapshot = _resolve_snapshot_path(candidate)
        if snapshot is not None:
            return snapshot
    return None


def _has_transformers_weights(checkpoint_dir: Path) -> bool:
    marker_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((checkpoint_dir / marker).exists() for marker in marker_files)


def _resolve_local_t5_safetensor() -> Path | None:
    env_path = os.environ.get("SERENITY_FLUX_T5_SAFETENSORS")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    candidates = (
        Path("/home/alex/eriui/comfyui/ComfyUI/models/clip/t5xxl_fp16.safetensors"),
        Path("/home/alex/eriui/comfyui/ComfyUI/models/clip/t5xxl_enconly.safetensors"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _materialize_single_file_checkpoint(
    checkpoint_dir: Path,
    config_source: Path,
    weight_source: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_target = checkpoint_dir / "config.json"
    weights_target = checkpoint_dir / "model.safetensors"

    if not config_target.exists():
        config_target.write_text(config_source.read_text())

    if not weights_target.exists():
        print(f"[serenity/flux] linking fallback weights {weight_source} -> {weights_target}")
        weights_target.symlink_to(weight_source)

    return checkpoint_dir


class Flux1Model(BaseModelImpl):
    """Flux 1.x adapter used by native Serenity training."""

    family = "flux"
    resolution_multiple = 64
    train_module_attr = "transformer"
    flow_objective = True
    text_id_dim = 3

    def __init__(self, *, model_type: ModelType = ModelType.FLUX_DEV) -> None:
        super().__init__(model_type=model_type)

    def _pipeline_candidates(self) -> tuple[str, ...]:
        if self.model_type == ModelType.FLUX_FILL_DEV:
            return ("FluxFillPipeline", "FluxPipeline")
        return ("FluxPipeline",)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        _validate_flux_path(model_path)
        try:
            pipeline = _load_pipeline_from_candidates(model_path, self._pipeline_candidates(), dtype)
        except RuntimeError as exc:
            if self.model_type == ModelType.FLUX_FILL_DEV:
                raise
            pipeline = self._load_pipeline_with_text_encoder_fallback(model_path, dtype, exc)

        pipeline.to("cpu")
        return pipeline

    def _load_pipeline_with_text_encoder_fallback(
        self,
        model_path: str,
        dtype: torch.dtype,
        original_error: Exception,
    ):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

        fallback_clip_path = _resolve_fallback_clip_path()
        fallback_t5_path = _resolve_fallback_t5_path()
        if fallback_clip_path is None:
            raise RuntimeError(
                "Flux text_encoder weights are missing in the model snapshot and no fallback CLIP cache was found. "
                "Set SERENITY_FLUX_CLIP_PATH to a local CLIP checkpoint directory."
            ) from original_error
        if fallback_t5_path is None or not _has_transformers_weights(fallback_t5_path):
            local_t5_weights = _resolve_local_t5_safetensor()
            if local_t5_weights is not None:
                config_source = Path(model_path) / "text_encoder_2" / "config.json"
                if config_source.exists():
                    fallback_t5_path = _materialize_single_file_checkpoint(
                        Path.home() / ".cache" / "serenity" / "flux_fallback" / "t5_v1_1_xxl",
                        config_source,
                        local_t5_weights,
                    )

        if fallback_t5_path is None or not _has_transformers_weights(fallback_t5_path):
            raise RuntimeError(
                "Flux text_encoder_2 weights are missing in the model snapshot and no fallback T5 cache was found. "
                "Set SERENITY_FLUX_T5_PATH to a local T5 checkpoint directory or "
                "SERENITY_FLUX_T5_SAFETENSORS to a local safetensors weight file."
            ) from original_error

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=True,
        )
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                local_files_only=True,
            )
        except (OSError, ValueError):
            tokenizer = CLIPTokenizer.from_pretrained(
                str(fallback_clip_path),
                local_files_only=True,
            )
        try:
            tokenizer_2 = T5Tokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer_2",
                local_files_only=True,
            )
        except (OSError, ValueError):
            tokenizer_2 = T5Tokenizer.from_pretrained(
                str(fallback_t5_path),
                local_files_only=True,
            )
        text_encoder = CLIPTextModel.from_pretrained(
            str(fallback_clip_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            str(fallback_t5_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True,
        )

        return FluxPipeline(
            scheduler=scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            transformer=transformer,
        )

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
        params = inspect.signature(pipeline.encode_prompt).parameters
        encode_kwargs: dict[str, Any] = {}
        if "prompt" in params:
            encode_kwargs["prompt"] = prompt
        if "prompt_2" in params:
            encode_kwargs["prompt_2"] = prompt
        if "device" in params:
            encode_kwargs["device"] = device
        if "num_images_per_prompt" in params:
            encode_kwargs["num_images_per_prompt"] = 1
        if "do_classifier_free_guidance" in params:
            encode_kwargs["do_classifier_free_guidance"] = False

        encoded = pipeline.encode_prompt(**encode_kwargs)

        values = list(encoded) if isinstance(encoded, tuple | list) else [encoded]
        prompt_embeds = values[0]
        pooled_prompt_embeds: torch.Tensor | None = None
        text_ids: torch.Tensor | None = None
        for value in values[1:]:
            if not torch.is_tensor(value):
                continue
            if value.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            } or (value.dim() >= 2 and value.shape[-1] in {3, 4}):
                if text_ids is None:
                    text_ids = value
                continue
            if pooled_prompt_embeds is None and value.dim() == 2 and value.is_floating_point():
                pooled_prompt_embeds = value

        if text_ids is None:
            text_ids = self.prepare_text_ids(prompt_embeds)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        if train_device.type == "cuda":
            # Flux text encoders (especially T5-XXL) can dominate VRAM on 24GB cards.
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        if device.type == "cuda":
            # Keep Flux text encoders on CPU for cache generation stability.
            return
        for attr in ("text_encoder", "text_encoder_2"):
            encoder = getattr(pipeline, attr, None)
            if encoder is None:
                continue
            encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        for attr in ("text_encoder", "text_encoder_2"):
            encoder = getattr(pipeline, attr, None)
            if encoder is None:
                continue
            with suppress(Exception):
                encoder.to("cpu")

    @staticmethod
    def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        seq_len = int(prompt_embeds.shape[1])
        text_ids = torch.zeros((seq_len, 3), device=prompt_embeds.device, dtype=torch.long)
        return text_ids

    @staticmethod
    def prepare_latent_image_ids(
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        image_ids = torch.zeros((latent_height // 2, latent_width // 2, 3), device=device, dtype=dtype)
        image_ids[..., 1] = image_ids[..., 1] + torch.arange(latent_height // 2, device=device, dtype=dtype)[:, None]
        image_ids[..., 2] = image_ids[..., 2] + torch.arange(latent_width // 2, device=device, dtype=dtype)[None, :]
        return image_ids.reshape(-1, 3)

    def pack_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = latents.shape
        packed = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        packed = packed.permute(0, 2, 4, 1, 3, 5)
        packed = packed.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        image_ids = self.prepare_latent_image_ids(height, width, latents.device, latents.dtype)
        return packed, image_ids

    @staticmethod
    def unpack_latents(packed: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
        batch_size, _, channels = packed.shape
        packed_h = latent_height // 2
        packed_w = latent_width // 2

        latents = packed.view(batch_size, packed_h, packed_w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, packed_h * 2, packed_w * 2)
        return latents
