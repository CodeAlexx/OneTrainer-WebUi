"""HunyuanVideo model adapter (Llama + CLIP, 3D Transformer, 3D VAE)."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl


DEFAULT_PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
DEFAULT_PROMPT_TEMPLATE_CROP_START = 95

# LoRA target modules
HUNYUAN_VIDEO_TRANSFORMER_LORA_TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

HUNYUAN_VIDEO_TEXT_ENCODER_1_LORA_TARGETS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

HUNYUAN_VIDEO_TEXT_ENCODER_2_LORA_TARGETS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)

HUNYUAN_VIDEO_LAYER_PRESETS: dict[str, list[str]] = {
    "attn-mlp": ["attn", "ff.net"],
    "attn-only": ["attn"],
    "blocks": ["transformer_block"],
    "full": [],
}


class HunyuanVideoModel(BaseModelImpl):
    """Native HunyuanVideo behavior used by Serenity training paths.

    Architecture: Llama text encoder + CLIP text encoder +
    HunyuanVideoTransformer3DModel + AutoencoderKLHunyuanVideo (3D VAE).
    Uses flow-matching objective.
    """

    family = "hunyuan_video"
    resolution_multiple = 16
    train_module_attr = "transformer"
    flow_objective = True

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.HUNYUAN_VIDEO)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **_: Any,
    ):
        del train_device
        from diffusers import HunyuanVideoPipeline

        pipeline = HunyuanVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to latent space using the 3D VAE."""
        # HunyuanVideo expects 5D input: (B, C, T, H, W)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        scaling_factor = float(getattr(pipeline.vae.config, "scaling_factor", 1.0))
        return latents * scaling_factor

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode text using both Llama and CLIP text encoders."""
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        tokenizer_1 = pipeline.tokenizer
        tokenizer_2 = pipeline.tokenizer_2

        # Llama text encoder (encoder 1)
        text_encoder_1_output = None
        if text_encoder_1 is not None and tokenizer_1 is not None:
            if text_encoder_1.device != device:
                text_encoder_1.to(device)
            llama_text = DEFAULT_PROMPT_TEMPLATE.format(prompt)
            tok_out = tokenizer_1(
                [llama_text],
                padding="max_length",
                truncation=True,
                max_length=77 + DEFAULT_PROMPT_TEMPLATE_CROP_START,
                return_tensors="pt",
            )
            tokens = tok_out.input_ids.to(device)
            attention_mask = tok_out.attention_mask.to(device)

            outputs = text_encoder_1(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Use layer -3 as default
            hidden_states = outputs.hidden_states
            if hidden_states is not None and len(hidden_states) >= 3:
                text_encoder_1_output = hidden_states[-3]
            else:
                text_encoder_1_output = outputs.last_hidden_state

            # Crop prompt template prefix
            if text_encoder_1_output.shape[1] > DEFAULT_PROMPT_TEMPLATE_CROP_START:
                text_encoder_1_output = text_encoder_1_output[
                    :, DEFAULT_PROMPT_TEMPLATE_CROP_START:, :
                ]

        if text_encoder_1_output is None:
            text_encoder_1_output = torch.zeros(
                (1, 77, 768), device=device, dtype=torch.float32
            )

        # CLIP text encoder (encoder 2) - pooled output
        pooled_output = None
        if text_encoder_2 is not None and tokenizer_2 is not None:
            if text_encoder_2.device != device:
                text_encoder_2.to(device)
            tok_out_2 = tokenizer_2(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tok_out_2.input_ids.to(device)
            outputs_2 = text_encoder_2(tokens_2, output_hidden_states=True, return_dict=True)
            pooled_output = outputs_2.pooler_output

        if pooled_output is None:
            pooled_output = torch.zeros((1, 768), device=device, dtype=torch.float32)

        return text_encoder_1_output, pooled_output, None

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        if train_device.type == "cuda":
            return torch.device("cpu")
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        if device.type == "cuda":
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


__all__ = [
    "HunyuanVideoModel",
    "HUNYUAN_VIDEO_TRANSFORMER_LORA_TARGETS",
    "HUNYUAN_VIDEO_TEXT_ENCODER_1_LORA_TARGETS",
    "HUNYUAN_VIDEO_TEXT_ENCODER_2_LORA_TARGETS",
    "HUNYUAN_VIDEO_LAYER_PRESETS",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_PROMPT_TEMPLATE_CROP_START",
]
