"""Stable Diffusion XL native model adapter."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl


class SDXLModel(BaseModelImpl):
    """Native SDXL behavior used by Serenity training paths."""

    family = "sdxl"
    resolution_multiple = 8
    train_module_attr = "unet"
    flow_objective = False

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SDXL)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.unet

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        return latents * float(pipeline.vae.config.scaling_factor)

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encoded = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = encoded[0]
        pooled_prompt_embeds = encoded[2] if len(encoded) > 2 else None
        return prompt_embeds, pooled_prompt_embeds, None

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
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
