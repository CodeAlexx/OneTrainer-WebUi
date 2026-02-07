"""Chroma model adapter (T5 + ChromaTransformer)."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl


# LoRA target modules for Chroma transformer and text encoder
CHROMA_TRANSFORMER_LORA_TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

CHROMA_TEXT_ENCODER_LORA_TARGETS: tuple[str, ...] = (
    "q",
    "k",
    "v",
    "o",
    "wi_0",
    "wi_1",
    "wo",
)

CHROMA_LAYER_PRESETS: dict[str, list[str]] = {
    "attn-mlp": ["attn", "ff.net"],
    "attn-only": ["attn"],
    "blocks": ["transformer_block"],
    "full": [],
}


class ChromaModel(BaseModelImpl):
    """Native Chroma behavior used by Serenity training paths.

    Architecture: T5 text encoder + ChromaTransformer2DModel + VAE.
    Uses flow-matching objective similar to Flux.
    """

    family = "chroma"
    resolution_multiple = 64
    train_module_attr = "transformer"
    flow_objective = True
    text_id_dim = 3

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.CHROMA_1)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        from diffusers import ChromaPipeline

        pipeline = ChromaPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
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
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        if text_encoder.device != device:
            text_encoder.to(device)

        tokenizer_output = tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = tokenizer_output.input_ids.to(device)
        attention_mask = tokenizer_output.attention_mask.to(device)

        outputs = text_encoder(
            input_ids=tokens,
            attention_mask=attention_mask.float(),
            output_hidden_states=True,
            return_dict=True,
        )
        # Use second-to-last hidden state (default layer -1 in OT)
        hidden_states = outputs.hidden_states
        text_encoder_output = hidden_states[-1] if hidden_states else outputs.last_hidden_state

        # Prune padding tokens
        bool_mask = attention_mask.bool()
        seq_len = bool_mask.sum(dim=1).max().item()
        # Pad to multiple of 16 for attention compatibility
        if seq_len % 16 > 0:
            seq_len += 16 - seq_len % 16
        text_encoder_output = text_encoder_output[:, :seq_len, :]

        text_ids = torch.zeros(
            (text_encoder_output.shape[1], 3),
            device=device,
            dtype=text_encoder_output.dtype,
        )
        return text_encoder_output, None, text_ids

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        if train_device.type == "cuda":
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        if device.type == "cuda":
            return
        encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        encoder = getattr(pipeline, "text_encoder", None)
        if encoder is None:
            return
        with suppress(Exception):
            encoder.to("cpu")

    @staticmethod
    def prepare_latent_image_ids(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build positional image IDs for packed latents."""
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] += torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] += torch.arange(width // 2)[None, :]
        return latent_image_ids.reshape(-1, 3).to(device=device, dtype=dtype)

    def pack_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack 4D latents into sequence format for transformer."""
        batch_size, channels, height, width = latents.shape
        packed = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        packed = packed.permute(0, 2, 4, 1, 3, 5)
        packed = packed.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        image_ids = self.prepare_latent_image_ids(height, width, latents.device, latents.dtype)
        return packed, image_ids

    @staticmethod
    def unpack_latents(
        packed: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """Unpack sequence format back to 4D latents."""
        batch_size, _, channels = packed.shape
        h = latent_height // 2
        w = latent_width // 2
        latents = packed.view(batch_size, h, w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, h * 2, w * 2)


__all__ = [
    "ChromaModel",
    "CHROMA_TRANSFORMER_LORA_TARGETS",
    "CHROMA_TEXT_ENCODER_LORA_TARGETS",
    "CHROMA_LAYER_PRESETS",
]
