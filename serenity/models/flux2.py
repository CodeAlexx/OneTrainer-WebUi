"""Native Flux 2 model adapter."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

import torch
import torch.nn.functional as F


def _validate_flux2_path(model_path: str) -> None:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Flux 2 model path does not exist: {path}")
    if path.is_file():
        return

    required = ("transformer", "vae", "text_encoder")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Flux 2 model at {path} is missing required components: {', '.join(missing)}")


def _load_flux2_pipeline(model_path: str, dtype: torch.dtype):
    import diffusers

    load_errors: list[str] = []
    for class_name in ("Flux2Pipeline", "Flux2KleinPipeline"):
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
    raise RuntimeError(f"Could not load Flux 2 pipeline from {model_path}: {details}")


class Flux2Model(BaseModelImpl):
    """Flux 2 adapter used by native Serenity training."""

    family = "flux2"
    resolution_multiple = 64
    train_module_attr = "transformer"
    flow_objective = True
    text_id_dim = 4

    def __init__(self, *, model_type: ModelType = ModelType.FLUX_2_DEV) -> None:
        super().__init__(model_type=model_type)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        _validate_flux2_path(model_path)
        pipeline = _load_flux2_pipeline(model_path, dtype)
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    @classmethod
    def normalize_patchified_latents(cls, pipeline: Any, latents: torch.Tensor) -> torch.Tensor:
        vae = getattr(pipeline, "vae", None)
        if vae is None:
            return latents

        bn = getattr(vae, "bn", None)
        if bn is None and hasattr(vae, "encoder"):
            bn = getattr(vae.encoder, "bn", None)
        if bn is None:
            return latents

        bn_channels = int(bn.running_mean.shape[0])
        latent_channels = int(latents.shape[1])

        if bn_channels == latent_channels:
            mean = bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            var = bn.running_var.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            return (latents - mean) / torch.sqrt(var + float(bn.eps))

        # Some checkpoints keep BN stats in unpatchified channel space.
        if bn_channels == (latent_channels // 4):
            unpatchified = cls.unpatchify_latents(latents)
            mean = bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            var = bn.running_var.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            normalized = (unpatchified - mean) / torch.sqrt(var + float(bn.eps))
            return cls.patchify_latents(normalized)

        return latents

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        if latents.dim() != 4:
            raise ValueError(f"Expected Flux2 latents with 4 dims, got {latents.shape}")
        patchified = self.patchify_latents(latents)
        return self.normalize_patchified_latents(pipeline, patchified)

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encoded = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
        )
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
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        text_encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        with suppress(Exception):
            text_encoder.to("cpu")

    @staticmethod
    def prepare_latent_image_ids(
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        t = torch.arange(1, device=device, dtype=torch.long)
        h = torch.arange(latent_height, device=device, dtype=torch.long)
        w = torch.arange(latent_width, device=device, dtype=torch.long)
        layer_idx = torch.arange(1, device=device, dtype=torch.long)
        ids = torch.cartesian_prod(t, h, w, layer_idx)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = prompt_embeds.shape
        t = torch.arange(1, device=prompt_embeds.device, dtype=torch.long)
        h = torch.arange(1, device=prompt_embeds.device, dtype=torch.long)
        w = torch.arange(1, device=prompt_embeds.device, dtype=torch.long)
        layer_idx = torch.arange(seq_len, device=prompt_embeds.device, dtype=torch.long)
        ids = torch.cartesian_prod(t, h, w, layer_idx)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def pack_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = latents.shape
        packed = latents.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        image_ids = self.prepare_latent_image_ids(batch_size, height, width, latents.device)
        return packed, image_ids

    @staticmethod
    def unpack_latents(packed: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
        batch_size, _, channels = packed.shape
        return packed.reshape(batch_size, latent_height, latent_width, channels).permute(0, 3, 1, 2)

    def pooled_text_projection(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = pooled_prompt_embeds if pooled_prompt_embeds is not None else prompt_embeds.mean(dim=1)

        expected_dim = None
        time_text_embed = getattr(getattr(self, "transformer", None), "time_text_embed", None)
        text_embedder = getattr(time_text_embed, "text_embedder", None)
        linear_1 = getattr(text_embedder, "linear_1", None)
        if linear_1 is not None and hasattr(linear_1, "in_features"):
            expected_dim = int(linear_1.in_features)

        if expected_dim is None or expected_dim <= 0 or pooled.shape[-1] == expected_dim:
            return pooled
        if pooled.shape[-1] > expected_dim:
            return pooled[..., :expected_dim]
        return F.pad(pooled, (0, expected_dim - pooled.shape[-1]))
