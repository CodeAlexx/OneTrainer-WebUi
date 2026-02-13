"""Data loading, caching, and batching for native diffusion training."""

from __future__ import annotations

import random
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from serenity.cli.utils import (
    as_bool as _as_bool,
    collect_concept_dirs as _collect_concept_dirs,
    is_condlabel_image as _is_condlabel_image,
    load_caption as _load_caption,
    load_image_tensor as _load_image_tensor,
    strip_condlabel_suffix as _strip_condlabel_suffix,
    COND_LABEL_SUFFIXES as _COND_LABEL_SUFFIXES,
)
from serenity.core.interfaces import ModelType

__all__ = [
    "_CachedExample",
    "_Batch",
    "_is_qwen_edit_type",
    "_is_video_file",
    "_load_video_tensor",
    "_expand_image_to_video_tensor",
    "_load_video_frame_tensor",
    "_load_media_tensor",
    "_resolve_conditioning_image_path",
    "_prepare_training_pairs",
    "_build_video_training_pairs",
    "_cache_training_data",
    "_pick_batch",
    "_materialize_batch_prompt_features",
    "_resolve_component_train_flags",
    "_set_module_train_state",
]

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _CachedExample:
    latents: torch.Tensor
    conditioning_latents: torch.Tensor | None
    prompt_embeds: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor | None
    prompt_mask: torch.Tensor | None
    caption: str
    conditioning_image: torch.Tensor | None
    num_frames: int
    height: int
    width: int


@dataclass
class _Batch:
    latents: torch.Tensor
    conditioning_latents: torch.Tensor | None
    prompt_embeds: torch.Tensor | list[torch.Tensor] | None
    pooled_prompt_embeds: torch.Tensor | None
    prompt_mask: torch.Tensor | None
    captions: list[str] | None
    conditioning_images: torch.Tensor | None
    num_frames: int
    height: int
    width: int


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _is_qwen_edit_type(model_type: ModelType) -> bool:
    return model_type == ModelType.QWEN_IMAGE_EDIT


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Media loading
# ---------------------------------------------------------------------------


def _load_video_tensor(
    video_path: Path,
    resolution: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    frame_count: int,
) -> torch.Tensor:
    frame_count = max(1, int(frame_count))
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        f"scale={resolution}:{resolution}:flags=lanczos",
        "-pix_fmt",
        "rgb24",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to decode video from {video_path}: {stderr or 'no output'}")

    frame_size = resolution * resolution * 3
    decoded_frames = len(result.stdout) // frame_size
    if decoded_frames <= 0:
        raise RuntimeError(f"Decoded video buffer for {video_path} contained no complete RGB frames.")

    usable = decoded_frames * frame_size
    array = np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(decoded_frames, resolution, resolution, 3)
    array = array.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()

    if decoded_frames < frame_count:
        pad = tensor[:, -1:, :, :].repeat(1, frame_count - decoded_frames, 1, 1)
        tensor = torch.cat([tensor, pad], dim=1)
    elif decoded_frames > frame_count:
        tensor = tensor[:, :frame_count, :, :]

    return tensor.to(device=device, dtype=dtype)


def _expand_image_to_video_tensor(image: torch.Tensor, frame_count: int) -> torch.Tensor:
    if frame_count <= 1:
        return image
    if image.dim() != 3:
        return image
    return image.unsqueeze(1).repeat(1, frame_count, 1, 1).contiguous()


def _load_video_frame_tensor(video_path: Path, resolution: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Backward-compatible helper for existing tests/imports.
    tensor = _load_video_tensor(video_path, resolution, dtype, device, frame_count=1)
    if tensor.dim() == 4:
        return tensor[:, 0, :, :]
    return tensor.to(device=device, dtype=dtype)


def _load_media_tensor(
    media_path: Path,
    resolution: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    allow_video: bool,
    video_frame_count: int = 1,
) -> torch.Tensor:
    if allow_video:
        if _is_video_file(media_path):
            return _load_video_tensor(
                media_path,
                resolution,
                dtype,
                device,
                frame_count=video_frame_count,
            )
        image = _load_image_tensor(media_path, resolution, dtype, device)
        return _expand_image_to_video_tensor(image, video_frame_count)
    return _load_image_tensor(media_path, resolution, dtype, device)


# ---------------------------------------------------------------------------
# Conditioning & training pairs
# ---------------------------------------------------------------------------


def _resolve_conditioning_image_path(
    image_path: Path,
    *,
    fallback_image_path: Path | None = None,
) -> Path | None:
    base_stem = _strip_condlabel_suffix(image_path.stem)
    parent = image_path.parent
    candidate_paths = [
        parent / f"{base_stem}{suffix}{ext}"
        for suffix in _COND_LABEL_SUFFIXES
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    ]

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate

    if fallback_image_path is not None and fallback_image_path.exists() and fallback_image_path.is_file():
        return fallback_image_path

    return None


def _prepare_training_pairs(
    pairs: list[tuple[Path, str]],
    *,
    model_type: ModelType,
) -> list[tuple[Path, str]]:
    if not _is_qwen_edit_type(model_type):
        return pairs

    filtered: list[tuple[Path, str]] = []
    for image_path, caption in pairs:
        if _is_condlabel_image(image_path):
            continue
        filtered.append((image_path, caption))
    return filtered


def _build_video_training_pairs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for concept_dir, caption_ext in _collect_concept_dirs(config):
        if not concept_dir.exists():
            continue
        for media_path in sorted(concept_dir.rglob("*")):
            if not media_path.is_file() or not _is_video_file(media_path):
                continue
            caption = _load_caption(media_path, caption_ext)
            if caption:
                pairs.append((media_path, caption))
    return pairs


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------


def _cache_training_data(
    model_impl: Any,
    pipeline: Any,
    family: str,
    pairs: list[tuple[Path, str]],
    model_type: ModelType,
    custom_conditioning_image: Path | None,
    resolution: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
    *,
    allow_video: bool = False,
    video_frame_count: int = 1,
    cache_text_embeddings: bool = True,
    keep_text_encoder_on_device: bool = False,
) -> list[_CachedExample]:
    pipeline.vae.to(train_device)
    prompt_device = model_impl.cache_prompt_device(pipeline, train_device)
    if cache_text_embeddings:
        model_impl.move_text_encoders_to_device(pipeline, train_device)

    cached: list[_CachedExample] = []
    total_pairs = len(pairs)
    if total_pairs > 0:
        print(f"[native/diffusion] cache progress 0/{total_pairs} (0.0%)")
    progress_interval = max(1, total_pairs // 20) if total_pairs > 0 else 1
    with torch.no_grad():
        for index, (media_path, caption) in enumerate(pairs, start=1):
            if _is_qwen_edit_type(model_type) and _is_condlabel_image(media_path):
                continue

            pixel_values = _load_media_tensor(
                media_path,
                resolution,
                train_dtype,
                train_device,
                allow_video=allow_video,
                video_frame_count=video_frame_count,
            ).unsqueeze(0)
            latents = model_impl.encode_latents(pipeline, pixel_values).squeeze(0).to(dtype=train_dtype).cpu()

            conditioning_latents: torch.Tensor | None = None
            conditioning_image_cpu: torch.Tensor | None = None
            conditioning_image_prompt: torch.Tensor | None = None
            if _is_qwen_edit_type(model_type):
                conditioning_path = _resolve_conditioning_image_path(
                    media_path,
                    fallback_image_path=custom_conditioning_image,
                )
                if conditioning_path is None:
                    conditioning_path = media_path

                conditioning_pixels = _load_image_tensor(conditioning_path, resolution, train_dtype, train_device).unsqueeze(0)
                conditioning_latents = (
                    model_impl.encode_latents(pipeline, conditioning_pixels).squeeze(0).to(dtype=train_dtype).cpu()
                )
                conditioning_image_cpu = conditioning_pixels.squeeze(0).to(dtype=torch.float32).cpu()
                conditioning_image_prompt = conditioning_pixels.to(device=prompt_device, dtype=torch.float32)

            prompt_embeds_cpu: torch.Tensor | None = None
            pooled_cpu: torch.Tensor | None = None
            mask_cpu: torch.Tensor | None = None
            if cache_text_embeddings:
                if conditioning_image_prompt is not None:
                    prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                        pipeline,
                        caption,
                        prompt_device,
                        conditioning_image=conditioning_image_prompt,
                    )
                else:
                    prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                        pipeline,
                        caption,
                        prompt_device,
                    )
                prompt_embeds_cpu = prompt_embeds.squeeze(0).to(dtype=train_dtype).cpu()
                pooled_cpu = (
                    pooled_prompt_embeds.squeeze(0).to(dtype=train_dtype).cpu()
                    if pooled_prompt_embeds is not None
                    else None
                )
                mask_cpu = prompt_mask.squeeze(0).cpu() if prompt_mask is not None else None

            cached.append(
                _CachedExample(
                    latents=latents,
                    conditioning_latents=conditioning_latents,
                    prompt_embeds=prompt_embeds_cpu,
                    pooled_prompt_embeds=pooled_cpu,
                    prompt_mask=mask_cpu,
                    caption=caption,
                    conditioning_image=conditioning_image_cpu,
                    num_frames=int(latents.shape[-3]) if latents.dim() == 4 else 1,
                    height=resolution,
                    width=resolution,
                )
            )
            if index == 1 or index % progress_interval == 0 or index == total_pairs:
                percent = (float(index) / float(total_pairs) * 100.0) if total_pairs > 0 else 100.0
                print(f"[native/diffusion] cache progress {index}/{total_pairs} ({percent:.1f}%)")

    pipeline.vae.to("cpu")
    if cache_text_embeddings and not keep_text_encoder_on_device:
        model_impl.offload_text_encoders(pipeline)

    return cached


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def _pick_batch(
    cached: list[_CachedExample],
    batch_size: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
    family: str,
) -> _Batch:
    items = random.choices(cached, k=max(1, batch_size))

    latents = torch.stack([item.latents for item in items], dim=0).to(train_device, dtype=train_dtype)
    conditioning_latents = None
    if items[0].conditioning_latents is not None:
        conditioning_latents = torch.stack([item.conditioning_latents for item in items], dim=0).to(
            train_device,
            dtype=train_dtype,
        )

    prompt_embeds: torch.Tensor | list[torch.Tensor] | None = None
    if items[0].prompt_embeds is not None:
        if family == "zimage":
            prompt_embeds = [
                item.prompt_embeds.to(train_device, dtype=train_dtype) for item in items if item.prompt_embeds is not None
            ]
        else:
            prompt_embeds = torch.stack([item.prompt_embeds for item in items if item.prompt_embeds is not None], dim=0).to(
                train_device,
                dtype=train_dtype,
            )

    pooled_prompt_embeds = None
    if items[0].pooled_prompt_embeds is not None:
        pooled_prompt_embeds = torch.stack([item.pooled_prompt_embeds for item in items], dim=0).to(
            train_device,
            dtype=train_dtype,
        )

    prompt_mask = None
    if items[0].prompt_mask is not None:
        prompt_mask = torch.stack([item.prompt_mask for item in items], dim=0).to(train_device)

    conditioning_images = None
    if items[0].conditioning_image is not None:
        conditioning_images = torch.stack([item.conditioning_image for item in items], dim=0).to(
            train_device,
            dtype=torch.float32,
        )
    captions = [item.caption for item in items]

    return _Batch(
        latents=latents,
        conditioning_latents=conditioning_latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_mask=prompt_mask,
        captions=captions,
        conditioning_images=conditioning_images,
        num_frames=items[0].num_frames,
        height=items[0].height,
        width=items[0].width,
    )


# ---------------------------------------------------------------------------
# Training component flags & state
# ---------------------------------------------------------------------------


def _resolve_component_train_flags(
    *,
    config: dict[str, Any],
    model_block: dict[str, Any],
    family: str,
    full_finetune: bool,
    training_method: str,
) -> tuple[bool, dict[str, bool], bool]:
    def _read_flag(
        key: str,
        *,
        block_key: str | None = None,
        default: bool = False,
    ) -> bool:
        for source in (
            config.get("train"),
            config.get("components"),
            model_block,
            config,
        ):
            if not isinstance(source, dict):
                continue
            if key in source:
                return _as_bool(source.get(key), default)
            if block_key and isinstance(source.get(block_key), dict) and "train" in source[block_key]:
                return _as_bool(source[block_key].get("train"), default)
        return default

    primary_key = "unet" if family in {"sd15", "sdxl"} else "transformer"
    primary_flag_key = "train_unet" if primary_key == "unet" else "train_transformer"
    train_primary = _read_flag(primary_flag_key, block_key=primary_key, default=full_finetune)

    if training_method == "fine_tune_vae":
        train_primary = False

    text_flags: dict[str, bool] = {}
    text_encoder_attrs = ("text_encoder", "text_encoder_2", "text_encoder_3")
    for idx, attr in enumerate(text_encoder_attrs, start=1):
        suffix = "" if idx == 1 else f"_{idx}"
        train_key = f"train_text_encoder{suffix}"
        text_flags[attr] = _read_flag(train_key, block_key=attr, default=False) if full_finetune else False

    # Diffusion objective does not provide VAE gradients in this path; keep it explicit.
    train_vae = _read_flag("train_vae", block_key="vae", default=training_method == "fine_tune_vae")
    if train_vae and training_method != "fine_tune_vae":
        print(
            "[native/diffusion] warning: train_vae=true requires training_method='fine_tune_vae'; "
            "ignoring train_vae for diffusion objective."
        )
        train_vae = False

    return train_primary, text_flags, train_vae


def _set_module_train_state(module: torch.nn.Module | None, should_train: bool) -> None:
    if module is None:
        return
    module.train(should_train)
    for param in module.parameters():
        param.requires_grad_(should_train)


# ---------------------------------------------------------------------------
# Prompt materialization
# ---------------------------------------------------------------------------


def _materialize_batch_prompt_features(
    *,
    model_impl: Any,
    pipeline: Any,
    family: str,
    model_type: ModelType,
    batch: _Batch,
    train_device: torch.device,
    prompt_device: torch.device | None = None,
    train_dtype: torch.dtype,
    requires_grad: bool,
) -> _Batch:
    if batch.prompt_embeds is not None:
        return batch
    if not batch.captions:
        raise RuntimeError("Prompt embeddings are missing and no captions are available to regenerate them.")

    prompt_embeds_list: list[torch.Tensor] = []
    pooled_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []

    context = nullcontext() if requires_grad else torch.no_grad()
    resolved_prompt_device = prompt_device or train_device
    with context:
        for idx, caption in enumerate(batch.captions):
            conditioning_image = None
            if _is_qwen_edit_type(model_type) and batch.conditioning_images is not None:
                conditioning_image = batch.conditioning_images[idx : idx + 1]

            if conditioning_image is not None:
                prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                    pipeline,
                    caption,
                    resolved_prompt_device,
                    conditioning_image=conditioning_image,
                )
            else:
                prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                    pipeline,
                    caption,
                    resolved_prompt_device,
                )

            prompt_embeds_list.append(prompt_embeds.squeeze(0))
            if pooled_prompt_embeds is not None:
                pooled_list.append(pooled_prompt_embeds.squeeze(0))
            if prompt_mask is not None:
                mask_list.append(prompt_mask.squeeze(0))

    if family == "zimage":
        resolved_prompt_embeds: torch.Tensor | list[torch.Tensor] = [
            prompt_embed.to(dtype=train_dtype) for prompt_embed in prompt_embeds_list
        ]
    else:
        resolved_prompt_embeds = torch.stack(prompt_embeds_list, dim=0).to(dtype=train_dtype)

    resolved_pooled = (
        torch.stack(pooled_list, dim=0).to(dtype=train_dtype)
        if pooled_list and len(pooled_list) == len(prompt_embeds_list)
        else None
    )
    resolved_mask = torch.stack(mask_list, dim=0) if mask_list and len(mask_list) == len(prompt_embeds_list) else None

    batch.prompt_embeds = resolved_prompt_embeds
    batch.pooled_prompt_embeds = resolved_pooled
    batch.prompt_mask = resolved_mask
    return batch
