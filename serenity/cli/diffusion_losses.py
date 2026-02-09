"""Loss computation for native diffusion training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from serenity.cli.utils import as_bool as _as_bool
from serenity.core.interfaces import ModelType
from serenity.models.qwen import QwenBaseModel

if TYPE_CHECKING:
    from serenity.cli.diffusion_data import _Batch

__all__ = [
    "_compute_vae_loss",
    "_sample_flow_timesteps",
    "_flow_add_noise",
    "_calculate_timestep_shift",
    "_qwen_pack_latents",
    "_qwen_unpack_latents",
    "_compute_loss",
    "_FLOW_FAMILIES",
    "_is_qwen_edit_type",
]

_FLOW_FAMILIES = {"sd3", "zimage", "qwen", "flux", "flux2", "ltx2"}


def _is_qwen_edit_type(model_type: ModelType) -> bool:
    return model_type == ModelType.QWEN_IMAGE_EDIT


def _compute_vae_loss(
    vae: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    kl_weight: float,
) -> torch.Tensor:
    encoded = vae.encode(pixel_values)
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is None:
        raise RuntimeError("VAE encode() did not return a latent distribution.")
    latents = latent_dist.sample()
    decoded = vae.decode(latents)
    reconstructed = decoded.sample if hasattr(decoded, "sample") else decoded

    recon_loss = F.mse_loss(reconstructed.float(), pixel_values.float(), reduction="mean")
    kl_term = latent_dist.kl().mean() if hasattr(latent_dist, "kl") else torch.tensor(0.0, device=pixel_values.device)
    return recon_loss + (float(kl_weight) * kl_term)

def _sample_flow_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    device: torch.device,
    shift: float,
) -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    timestep = u * float(num_train_timesteps)

    if abs(float(shift) - 1.0) > 1e-6:
        numerator = float(num_train_timesteps) * float(shift) * timestep
        denominator = (float(shift) - 1.0) * timestep + float(num_train_timesteps)
        timestep = numerator / denominator

    return torch.clamp(timestep.long(), 0, num_train_timesteps - 1)


def _flow_add_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = (timesteps.to(dtype=latents.dtype) + 1.0) / float(num_train_timesteps)
    while sigma.dim() < latents.dim():
        sigma = sigma.unsqueeze(-1)
    noisy = noise * sigma + latents * (1.0 - sigma)
    return noisy, sigma


def _calculate_timestep_shift(
    scheduler: Any,
    latent_height: int,
    latent_width: int,
) -> float:
    config = getattr(scheduler, "config", None)
    base_seq_len = int(getattr(config, "base_image_seq_len", 256))
    max_seq_len = int(getattr(config, "max_image_seq_len", 4096))
    base_shift = float(getattr(config, "base_shift", 0.5))
    max_shift = float(getattr(config, "max_shift", 1.15))

    patch_size = 2
    image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
    m = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    b = base_shift - m * float(base_seq_len)
    mu = float(image_seq_len) * m + b
    return float(math.exp(mu))


def _qwen_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    return QwenBaseModel.pack_latents(latents)


def _qwen_unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return QwenBaseModel.unpack_latents(latents, height, width)


def _compute_loss(
    model_impl: Any,
    pipeline: Any,
    family: str,
    batch: _Batch,
    train_module: torch.nn.Module,
    train_dtype: torch.dtype,
    config: dict[str, Any],
) -> torch.Tensor:
    latents = batch.latents
    batch_size = latents.shape[0]
    if batch.prompt_embeds is None:
        raise RuntimeError("Prompt embeddings must be materialized before loss computation.")

    if family in _FLOW_FAMILIES:
        num_train_timesteps = int(getattr(pipeline.scheduler.config, "num_train_timesteps", 1000))
        shift = float(config.get("timestep_shift", 1.0))
        if _as_bool(config.get("dynamic_timestep_shifting", False), False):
            shift = _calculate_timestep_shift(
                pipeline.scheduler,
                int(latents.shape[-2]),
                int(latents.shape[-1]),
            )

        timesteps = _sample_flow_timesteps(batch_size, num_train_timesteps, latents.device, shift)
        noise = torch.randn_like(latents)
        noisy_latents, _ = _flow_add_noise(latents, noise, timesteps, num_train_timesteps)
        flow_target = noise - latents

        if family in {"flux", "flux2"}:
            packed_input, image_ids = model_impl.pack_latents(noisy_latents)
            text_ids = batch.prompt_mask
            if text_ids is None:
                text_ids = model_impl.prepare_text_ids(batch.prompt_embeds)

            # Flux expects [L, 3] (Flux 1) or [B, L, 4] (Flux 2) depending model class.
            if text_ids.dim() == 2 and batch_size > 1:
                text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1)

            timestep_input = timesteps.to(dtype=train_dtype) / 1000.0
            forward_kwargs: dict[str, Any] = {
                "hidden_states": packed_input.to(dtype=train_dtype),
                "encoder_hidden_states": batch.prompt_embeds.to(dtype=train_dtype),
                "timestep": timestep_input,
                "img_ids": image_ids,
                "txt_ids": text_ids,
                "return_dict": True,
            }

            if batch.pooled_prompt_embeds is not None:
                forward_kwargs["pooled_projections"] = batch.pooled_prompt_embeds.to(dtype=train_dtype)

            if bool(getattr(train_module.config, "guidance_embeds", False)):
                guidance_scale = float(config.get("guidance_scale", 1.0))
                forward_kwargs["guidance"] = torch.full(
                    (batch_size,),
                    guidance_scale,
                    device=latents.device,
                    dtype=train_dtype,
                )

            predicted_packed_flow = train_module(**forward_kwargs).sample
            predicted_flow = model_impl.unpack_latents(
                predicted_packed_flow,
                int(latents.shape[-2]),
                int(latents.shape[-1]),
            )
            return F.mse_loss(predicted_flow.float(), flow_target.float(), reduction="mean")

        if family == "sd3":
            forward_kwargs: dict[str, Any] = {
                "hidden_states": noisy_latents.to(dtype=train_dtype),
                "encoder_hidden_states": batch.prompt_embeds.to(dtype=train_dtype),
                "timestep": timesteps,
                "return_dict": True,
            }
            if batch.pooled_prompt_embeds is not None:
                forward_kwargs["pooled_projections"] = batch.pooled_prompt_embeds.to(dtype=train_dtype)
            predicted_flow = train_module(**forward_kwargs).sample
            return F.mse_loss(predicted_flow.float(), flow_target.float(), reduction="mean")

        if family == "ltx2":
            latent_frames = int(latents.shape[-3])
            latent_height = int(latents.shape[-2])
            latent_width = int(latents.shape[-1])
            patch_size, patch_size_t = model_impl.get_patch_sizes(pipeline)
            packed_input = model_impl.pack_latents(
                noisy_latents,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )

            attention_mask = batch.prompt_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, int(batch.prompt_embeds.shape[1])),
                    device=latents.device,
                    dtype=torch.long,
                )

            forward_kwargs: dict[str, Any] = {
                "hidden_states": packed_input.to(dtype=train_dtype),
                "encoder_hidden_states": batch.prompt_embeds.to(dtype=train_dtype),
                "timestep": timesteps,
                "encoder_attention_mask": attention_mask,
                "num_frames": latent_frames,
                "height": latent_height,
                "width": latent_width,
                "return_dict": True,
            }
            rope_scale = model_impl.resolve_rope_interpolation_scale(pipeline)
            if rope_scale is not None:
                if torch.is_tensor(rope_scale):
                    forward_kwargs["rope_interpolation_scale"] = rope_scale.to(latents.device)
                else:
                    forward_kwargs["rope_interpolation_scale"] = rope_scale

            predicted = train_module(**forward_kwargs)
            predicted_packed_flow = predicted.sample if hasattr(predicted, "sample") else predicted
            if isinstance(predicted_packed_flow, tuple | list):
                predicted_packed_flow = predicted_packed_flow[0]
            predicted_flow = model_impl.unpack_latents(
                predicted_packed_flow,
                frames=latent_frames // patch_size_t,
                height=latent_height // patch_size,
                width=latent_width // patch_size,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            return F.mse_loss(predicted_flow.float(), flow_target.float(), reduction="mean")

        if family == "qwen":
            latent_height = int(latents.shape[-2])
            latent_width = int(latents.shape[-1])
            packed_target = model_impl.pack_latents(noisy_latents)
            packed_input = packed_target
            timestep_input = timesteps.to(dtype=train_dtype) / 1000.0
            img_shapes = [[(1, latent_height // 2, latent_width // 2)]] * batch_size

            use_edit_concat = (
                _is_qwen_edit_type(model_impl.model_type)
                and batch.conditioning_latents is not None
                and "editpipeline" in pipeline.__class__.__name__.lower()
            )
            if use_edit_concat:
                packed_conditioning = model_impl.pack_latents(batch.conditioning_latents)
                packed_input = torch.cat([packed_target, packed_conditioning], dim=1)
                img_shapes = [[(1, latent_height // 2, latent_width // 2), (1, latent_height // 2, latent_width // 2)]] * batch_size

            attention_mask = batch.prompt_mask
            if attention_mask is not None:
                txt_seq_lens = attention_mask.sum(dim=1).to(dtype=torch.int64).tolist()
            else:
                txt_seq_lens = [int(batch.prompt_embeds.shape[1])] * batch_size
            if attention_mask is not None and torch.all(attention_mask):
                attention_mask = None

            guidance = None
            if bool(getattr(train_module.config, "guidance_embeds", False)):
                guidance_scale = float(config.get("guidance_scale", 1.0))
                guidance = torch.full((batch_size,), guidance_scale, device=latents.device, dtype=torch.float32)

            predicted_packed_flow = train_module(
                hidden_states=packed_input.to(dtype=train_dtype),
                timestep=timestep_input,
                encoder_hidden_states=batch.prompt_embeds.to(dtype=train_dtype),
                encoder_hidden_states_mask=attention_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=guidance,
                return_dict=True,
            ).sample
            if packed_input.shape[1] != packed_target.shape[1]:
                predicted_packed_flow = predicted_packed_flow[:, : packed_target.shape[1]]
            predicted_flow = model_impl.unpack_latents(predicted_packed_flow, latent_height, latent_width)
            return F.mse_loss(predicted_flow.float(), flow_target.float(), reduction="mean")

        if family == "zimage":
            latent_input = noisy_latents.unsqueeze(2).to(dtype=train_dtype)
            latent_input_list = list(latent_input.unbind(dim=0))
            timestep_input = (1000 - timesteps).to(dtype=train_dtype) / 1000.0
            prompt_list = [pe.to(dtype=train_dtype) for pe in batch.prompt_embeds]

            output_list = train_module(
                latent_input_list,
                timestep_input,
                prompt_list,
                return_dict=True,
            ).sample
            predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)
            return F.mse_loss(predicted_flow.float(), flow_target.float(), reduction="mean")

        raise ValueError(f"Unsupported flow family: {family}")

    noise = torch.randn_like(latents)
    num_train_timesteps = int(getattr(pipeline.scheduler.config, "num_train_timesteps", 1000))
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=latents.device, dtype=torch.long)
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

    if family == "sd15":
        predicted = train_module(
            noisy_latents.to(dtype=train_dtype),
            timesteps,
            batch.prompt_embeds.to(dtype=train_dtype),
        ).sample
    elif family == "sdxl":
        add_time_ids = (
            torch.tensor(
                [batch.height, batch.width, 0, 0, batch.height, batch.width],
                device=latents.device,
                dtype=batch.prompt_embeds.dtype,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        predicted = train_module(
            sample=noisy_latents.to(dtype=train_dtype),
            timestep=timesteps,
            encoder_hidden_states=batch.prompt_embeds.to(dtype=train_dtype),
            added_cond_kwargs={
                "text_embeds": batch.pooled_prompt_embeds.to(dtype=train_dtype),
                "time_ids": add_time_ids,
            },
        ).sample
    else:
        raise ValueError(f"Unsupported diffusion family: {family}")

    prediction_type = str(getattr(pipeline.scheduler.config, "prediction_type", "epsilon"))
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = pipeline.scheduler.get_velocity(latents, noise, timesteps)
    elif prediction_type == "sample":
        target = latents
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    return F.mse_loss(predicted.float(), target.float(), reduction="mean")
