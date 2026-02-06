"""Stable Diffusion 1.5 native model adapter."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl


class SD15Model(BaseModelImpl):
    """Native SD1.5 behavior used by EriTrainer training paths."""

    family = "sd15"
    resolution_multiple = 8
    train_module_attr = "unet"
    flow_objective = False

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.SD15)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        from diffusers import StableDiffusionPipeline

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
        }
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("safety_checker", None)
            load_kwargs.pop("feature_extractor", None)
            load_kwargs.pop("requires_safety_checker", None)
            pipeline = StableDiffusionPipeline.from_pretrained(model_path, **load_kwargs)
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
        tokens = pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        prompt_embeds = pipeline.text_encoder(input_ids)[0]
        return prompt_embeds, None, None

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        with suppress(Exception):
            encoder.to("cpu")
