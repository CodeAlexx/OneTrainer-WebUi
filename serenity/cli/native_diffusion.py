"""Native non-bridge diffusion training path for Serenity.

Supports native adapter/full training for:
- Flux 1.x (dev/schnell/fill)
- Flux 2.x (dev path)
- Z-Image
- Qwen Image / Qwen Image Edit
- Stable Diffusion 1.5
- Stable Diffusion XL
- Stable Diffusion 3 / 3.5
"""

from __future__ import annotations

import inspect
import math
import os
import random
import subprocess
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from serenity.adapters import create_adapter
from serenity.cli.native_flux2 import (
    _as_bool,
    _build_adapter_kwargs,
    _build_training_pairs,
    _coerce_dtype,
    _collect_concept_dirs,
    _collect_sample_prompts,
    _collect_sample_seeds,
    _create_lr_scheduler,
    _create_optimizer,
    _extract_adapter_config,
    _load_caption,
    _normalize_model_type,
    _optional_float,
    _optional_int,
    _resolve_hf_local_path,
    _resolve_optimizer_steps,
    _resolve_sample_output_extension,
)
from serenity.core.interfaces import ModelType
from serenity.memory.strategy import LayerOffloadStrategy, MemoryConfig
from serenity.models.flux1 import Flux1Model
from serenity.models.flux2 import Flux2Model
from serenity.models.flux_schnell import FluxSchnellModel
from serenity.models.ltx2 import LTX2Model
from serenity.models.qwen import QwenBaseModel, QwenImageEditModel, QwenModel
from serenity.models.sd3 import SD3Model, SD35Model
from serenity.models.sd15 import SD15Model
from serenity.models.sdxl import SDXLModel
from serenity.models.zimage import ZImageModel
from serenity.sampling.sampler import create_sampler
from serenity.training.ema import EMAMode, EMAModel

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import numpy as np
from PIL import Image

_SD15_TYPES = {
    "sd15",
    "sd_15",
}

_FLUX_TYPES = {
    "flux",
    "flux_dev",
    "flux_schnell",
    "flux_fill",
    "flux_fill_dev",
}

_FLUX2_TYPES = {
    "flux_2",
    "flux2",
    "flux_2_dev",
    "flux2_dev",
}

_SDXL_TYPES = {
    "sdxl",
    "sdxl_10_base",
    "sdxl_base",
}

_SD3_TYPES = {
    "sd3",
    "sd_3",
    "sd35",
    "sd_35",
    "sd3.5",
    "stable_diffusion_3",
    "stable_diffusion_35",
    "stable_diffusion_3.5",
}

_ZIMAGE_TYPES = {
    "zimage",
    "z_image",
}

_QWEN_TYPES = {
    "qwen",
    "qwen_image_edit",
}

_LTX2_TYPES = {
    "ltx2",
    "ltx",
    "ltxvideo",
    "ltx_video",
}

_FLOW_FAMILIES = {"sd3", "zimage", "qwen", "flux", "flux2", "ltx2"}
_COND_LABEL_SUFFIXES = ("-condlabel", "_condlabel")
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_FULL_TRAINING_METHODS = {"full", "full_finetune", "full_fine_tune", "fine_tune", "finetune", "fine_tune_vae"}


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


@dataclass
class _DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None
    initialized_here: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _resolve_distributed_context(
    config: dict[str, Any],
    model_block: dict[str, Any],
    requested_train_device: torch.device,
) -> tuple[_DistributedContext, torch.device]:
    distributed_block = config.get("distributed")
    distributed_cfg = distributed_block if isinstance(distributed_block, dict) else {}

    explicit_enabled = _as_bool(
        distributed_cfg.get("enabled", config.get("distributed_enabled", config.get("multi_gpu", False))),
        False,
    )
    env_world_size = _optional_int(os.environ.get("WORLD_SIZE")) or 1
    cfg_world_size = _optional_int(
        distributed_cfg.get("world_size", model_block.get("world_size", config.get("world_size")))
    ) or env_world_size
    world_size = int(max(1, cfg_world_size or env_world_size))
    enabled = bool(explicit_enabled or world_size > 1)
    if not enabled or world_size <= 1:
        return _DistributedContext(enabled=False), requested_train_device

    if not torch.distributed.is_available():
        raise RuntimeError("Distributed training was requested, but torch.distributed is unavailable.")

    rank = _optional_int(os.environ.get("RANK"))
    if rank is None:
        rank = _optional_int(distributed_cfg.get("rank", config.get("rank")))
    if rank is None:
        rank = 0
    local_rank = _optional_int(os.environ.get("LOCAL_RANK"))
    if local_rank is None:
        local_rank = _optional_int(distributed_cfg.get("local_rank", config.get("local_rank")))
    if local_rank is None:
        local_rank = rank
    rank = int(max(0, rank or 0))
    local_rank = int(max(0, local_rank or 0))

    train_device = requested_train_device
    if train_device.type == "cuda":
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("Distributed CUDA training requested but no CUDA devices are visible.")
        if train_device.index is None:
            train_device = torch.device("cuda", local_rank % device_count)

    backend_default = "nccl" if train_device.type == "cuda" else "gloo"
    backend = str(distributed_cfg.get("backend", config.get("distributed_backend", backend_default))).strip().lower()
    if not backend:
        backend = backend_default

    initialized_here = False
    if not torch.distributed.is_initialized():
        master_addr = str(distributed_cfg.get("master_addr", config.get("master_addr", "127.0.0.1")))
        master_port = str(distributed_cfg.get("master_port", config.get("master_port", "29500")))
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    else:
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())

    if train_device.type == "cuda":
        with suppress(Exception):
            torch.cuda.set_device(train_device)

    context = _DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
        initialized_here=initialized_here,
    )
    print(
        f"[native/diffusion] distributed enabled rank={context.rank}/{context.world_size} "
        f"backend={context.backend} device={train_device}"
    )
    return context, train_device


def _distributed_barrier(context: _DistributedContext) -> None:
    if not context.enabled:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    with suppress(Exception):
        torch.distributed.barrier()


def _finalize_distributed_context(context: _DistributedContext) -> None:
    if not context.enabled:
        return
    _distributed_barrier(context)
    if context.initialized_here and torch.distributed.is_available() and torch.distributed.is_initialized():
        with suppress(Exception):
            torch.distributed.destroy_process_group()


def _maybe_wrap_distributed_module(
    module: torch.nn.Module | None,
    context: _DistributedContext,
    train_device: torch.device,
    *,
    find_unused_parameters: bool = False,
) -> torch.nn.Module | None:
    if module is None or not context.enabled:
        return module
    kwargs: dict[str, Any] = {
        "find_unused_parameters": bool(find_unused_parameters),
        "broadcast_buffers": False,
    }
    if train_device.type == "cuda":
        kwargs["device_ids"] = [int(train_device.index or 0)]
        kwargs["output_device"] = int(train_device.index or 0)
    return DistributedDataParallel(module, **kwargs)


def _normalize_training_method(value: Any) -> str:
    normalized = str(value or "lora").strip().lower()
    if normalized in {"fine_tune", "finetune", "full", "full_finetune", "full_fine_tune"}:
        return "fine_tune"
    if normalized in {"fine_tune_vae", "finetune_vae", "vae"}:
        return "fine_tune_vae"
    if normalized in {"embedding", "lora"}:
        return normalized
    return "lora"


def is_native_diffusion_model_type(model_type: str | None) -> bool:
    normalized = _normalize_model_type(model_type)
    return normalized in (
        _FLUX_TYPES
        | _FLUX2_TYPES
        | _SD15_TYPES
        | _SDXL_TYPES
        | _SD3_TYPES
        | _ZIMAGE_TYPES
        | _QWEN_TYPES
        | _LTX2_TYPES
    )


def _resolve_family(model_type: str) -> str:
    if model_type in _FLUX_TYPES:
        return "flux"
    if model_type in _FLUX2_TYPES:
        return "flux2"
    if model_type in _SD15_TYPES:
        return "sd15"
    if model_type in _SDXL_TYPES:
        return "sdxl"
    if model_type in _SD3_TYPES:
        return "sd3"
    if model_type in _ZIMAGE_TYPES:
        return "zimage"
    if model_type in _QWEN_TYPES:
        return "qwen"
    if model_type in _LTX2_TYPES:
        return "ltx2"
    raise ValueError(f"Unsupported native diffusion model type: {model_type}")


def _create_native_model(model_type: str):
    if model_type == "flux_schnell":
        return FluxSchnellModel()
    if model_type in {"flux_fill", "flux_fill_dev"}:
        return Flux1Model(model_type=ModelType.FLUX_FILL_DEV)
    if model_type in {"flux", "flux_dev"}:
        return Flux1Model()
    if model_type in _FLUX2_TYPES:
        return Flux2Model(model_type=ModelType.FLUX_2_DEV)
    if model_type in _SD15_TYPES:
        return SD15Model()
    if model_type in _SDXL_TYPES:
        return SDXLModel()
    if model_type in _SD3_TYPES:
        if "35" in model_type or "3.5" in model_type:
            return SD35Model()
        return SD3Model()
    if model_type in _ZIMAGE_TYPES:
        return ZImageModel()
    if model_type == "qwen_image_edit":
        return QwenImageEditModel()
    if model_type == "qwen":
        return QwenModel()
    if model_type in _LTX2_TYPES:
        return LTX2Model()
    raise ValueError(f"Unsupported native diffusion model type: {model_type}")


def _create_native_model_for_family(family: str):
    if family == "flux":
        return Flux1Model()
    if family == "flux2":
        return Flux2Model(model_type=ModelType.FLUX_2_DEV)
    if family == "sd15":
        return SD15Model()
    if family == "sdxl":
        return SDXLModel()
    if family == "sd3":
        return SD3Model()
    if family == "zimage":
        return ZImageModel()
    if family == "qwen":
        return QwenModel()
    if family == "ltx2":
        return LTX2Model()
    raise ValueError(f"Unsupported family: {family}")


def _resolve_sampler_model_type(model_type: str) -> ModelType:
    if model_type in {"flux", "flux_dev"}:
        return ModelType.FLUX_DEV
    if model_type in {"flux_fill", "flux_fill_dev"}:
        return ModelType.FLUX_FILL_DEV
    if model_type == "flux_schnell":
        return ModelType.FLUX_SCHNELL
    if model_type in _FLUX2_TYPES:
        return ModelType.FLUX_2_DEV if model_type in {"flux_2_dev", "flux2_dev"} else ModelType.FLUX_2
    if model_type in _SD15_TYPES:
        return ModelType.SD15
    if model_type in _SDXL_TYPES:
        return ModelType.SDXL
    if model_type in _SD3_TYPES:
        return ModelType.SD35 if "35" in model_type or "3.5" in model_type else ModelType.SD3
    if model_type in _ZIMAGE_TYPES:
        return ModelType.ZIMAGE
    if model_type == "qwen_image_edit":
        return ModelType.QWEN_IMAGE_EDIT
    if model_type == "qwen":
        return ModelType.QWEN
    if model_type in _LTX2_TYPES:
        return ModelType.LTX2
    raise ValueError(f"Unsupported sampler model type: {model_type}")


def _resolve_snapshot_from_repo_dir(repo_dir: Path) -> str:
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)

    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
    if snapshots:
        return str(snapshots[-1])

    raise FileNotFoundError(f"No snapshots found in {repo_dir}")


def _resolve_model_path(model_path: str) -> str:
    expanded = Path(model_path).expanduser()
    if expanded.exists():
        return str(expanded)

    # If this is an absolute filesystem path and it does not exist, do not reinterpret it as a repo id.
    if expanded.is_absolute():
        # Best-effort recovery for snapshot paths with case mismatches in `models--ORG--NAME`.
        marker = "models--"
        as_posix = expanded.as_posix()
        if marker in as_posix:
            token = as_posix.split(marker, 1)[1].split("/", 1)[0]
            hub_root = Path.home() / ".cache" / "huggingface" / "hub"
            expected = f"{marker}{token}".lower()
            for candidate in hub_root.glob("models--*"):
                if candidate.name.lower() == expected:
                    return _resolve_snapshot_from_repo_dir(candidate)
        raise FileNotFoundError(f"Model path does not exist: {expanded}")

    # Repo id path (org/name) handled by shared helper.
    return _resolve_hf_local_path(str(model_path))


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser())


def _pick_path_override(
    config: dict[str, Any],
    model_block: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = model_block.get(key)
        normalized = _normalize_optional_path(value)
        if normalized:
            return normalized
        value = config.get(key)
        normalized = _normalize_optional_path(value)
        if normalized:
            return normalized
    return None


def _collect_ltx_component_paths(
    config: dict[str, Any],
    model_block: dict[str, Any],
) -> tuple[dict[str, str], str | None]:
    component_paths: dict[str, str] = {}
    key_map: dict[str, tuple[str, ...]] = {
        "transformer": ("transformer_path", "transformer"),
        "vae": ("vae_path", "vae"),
        "text_encoder": ("text_encoder_path", "text_encoder", "clip_path"),
        "tokenizer": ("tokenizer_path", "tokenizer"),
        "scheduler": ("scheduler_path", "scheduler_dir"),
    }

    for component_name, aliases in key_map.items():
        value = _pick_path_override(config, model_block, aliases)
        if value:
            component_paths[component_name] = value

    template_path = _pick_path_override(
        config,
        model_block,
        ("ltx_template_path", "template_path", "template"),
    )
    return component_paths, template_path


def _resolution_multiple_for_family(family: str) -> int:
    if family in {"sd15", "sdxl"}:
        return 8
    if family == "sd3":
        return 16
    if family == "ltx2":
        return 32
    return 64


def _quantize_resolution(resolution: int, multiple: int) -> int:
    quantized = int(round(float(resolution) / float(multiple)) * multiple)
    return max(multiple, quantized)


def _is_qwen_edit_type(model_type: ModelType) -> bool:
    return model_type == ModelType.QWEN_IMAGE_EDIT


def _is_condlabel_image(path: Path) -> bool:
    stem = path.stem.lower()
    return any(stem.endswith(suffix) for suffix in _COND_LABEL_SUFFIXES)


def _strip_condlabel_suffix(stem: str) -> str:
    lowered = stem.lower()
    for suffix in _COND_LABEL_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


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


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTENSIONS


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


def _iter_text_encoders(pipeline: Any) -> list[Any]:
    out: list[Any] = []
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipeline, attr, None)
        if module is not None:
            out.append(module)
    return out


def _normalize_quantization_mode(value: Any) -> str | None:
    return QwenBaseModel.normalize_quantization_mode(value)


def _is_dispatched_module(module: Any) -> bool:
    return QwenBaseModel.is_dispatched_module(module)


def _load_pipeline(
    family: str,
    model_path: str,
    dtype: torch.dtype,
    train_device: torch.device,
    quantization_mode: str | None = None,
    requested_gpu_budget_gib: int | None = None,
):
    model_impl = _create_native_model_for_family(family)
    return model_impl.load_pipeline(
        model_path,
        dtype,
        train_device,
        quantization_mode=quantization_mode,
        requested_gpu_budget_gib=requested_gpu_budget_gib,
    )


def _load_image_tensor(image_path: Path, resolution: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=dtype)


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


def _resolve_ltx_video_frame_count(config: dict[str, Any], data_block: dict[str, Any]) -> int:
    requested = int(
        data_block.get("frames")
        or data_block.get("video_frames")
        or data_block.get("num_frames")
        or config.get("frames")
        or config.get("video_frames")
        or config.get("num_frames")
        or 9
    )
    return max(1, int(LTX2Model.adjust_video_frames(requested)))


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


def _encode_latents(pipeline: Any, family: str, pixel_values: torch.Tensor) -> torch.Tensor:
    model_impl = _create_native_model_for_family(family)
    return model_impl.encode_latents(pipeline, pixel_values)


def _encode_prompt_features(
    pipeline: Any,
    family: str,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    model_impl = _create_native_model_for_family(family)
    return model_impl.encode_prompt_features(pipeline, prompt, device)


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
    with torch.no_grad():
        for media_path, caption in pairs:
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

    pipeline.vae.to("cpu")
    if cache_text_embeddings and not keep_text_encoder_on_device:
        model_impl.offload_text_encoders(pipeline)

    return cached


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


def _materialize_batch_prompt_features(
    *,
    model_impl: Any,
    pipeline: Any,
    family: str,
    model_type: ModelType,
    batch: _Batch,
    train_device: torch.device,
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
    with context:
        for idx, caption in enumerate(batch.captions):
            conditioning_image = None
            if _is_qwen_edit_type(model_type) and batch.conditioning_images is not None:
                conditioning_image = batch.conditioning_images[idx : idx + 1]

            if conditioning_image is not None:
                prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                    pipeline,
                    caption,
                    train_device,
                    conditioning_image=conditioning_image,
                )
            else:
                prompt_embeds, pooled_prompt_embeds, prompt_mask = model_impl.encode_prompt_features(
                    pipeline,
                    caption,
                    train_device,
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


def _save_module_state(
    module: torch.nn.Module,
    output_path: Path,
    *,
    save_dtype: torch.dtype | None = None,
) -> Path:
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().cpu()
        if save_dtype is not None and value.is_floating_point():
            value = value.to(dtype=save_dtype)
        state_dict[name] = value.contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(output_path))
        return output_path
    except (ImportError, OSError):
        fallback_path = output_path.with_suffix(".pt")
        torch.save(state_dict, fallback_path)
        return fallback_path


def _save_training_state(
    output_path: Path,
    *,
    step: int,
    model_checkpoint: Path | None,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any | None,
    ema_model: EMAModel | None,
) -> Path:
    state: dict[str, Any] = {
        "step": int(step),
        "model_checkpoint": str(model_checkpoint) if model_checkpoint is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        with suppress(Exception):
            state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    return output_path


def _resolve_resume_state_path(config: dict[str, Any], checkpoint_block: dict[str, Any], output_dir: Path) -> Path | None:
    resume_raw = (
        checkpoint_block.get("resume_state")
        or checkpoint_block.get("resume_from")
        or config.get("resume_state")
        or config.get("resume_from")
        or config.get("resume_checkpoint")
    )
    if resume_raw:
        candidate = Path(str(resume_raw)).expanduser()
        if candidate.is_dir():
            states = sorted(candidate.glob("state_step_*.pt"))
            return states[-1] if states else None
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Resume state path not found: {candidate}")

    latest = sorted(output_dir.glob("state_step_*.pt"))
    return latest[-1] if latest else None


def _maybe_restore_training_state(
    *,
    resume_state_path: Path | None,
    train_module: torch.nn.Module,
    adapter: Any | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    ema_model: EMAModel | None,
) -> int:
    if resume_state_path is None:
        return 1

    state = torch.load(str(resume_state_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid training state file: {resume_state_path}")

    model_checkpoint_raw = state.get("model_checkpoint")
    if model_checkpoint_raw:
        model_checkpoint = Path(str(model_checkpoint_raw)).expanduser()
        if model_checkpoint.exists():
            if adapter is not None:
                with suppress(Exception):
                    adapter.load(str(model_checkpoint))
            else:
                if model_checkpoint.suffix.lower() == ".safetensors":
                    from safetensors.torch import load_file

                    model_state = load_file(str(model_checkpoint))
                else:
                    model_state = torch.load(str(model_checkpoint), map_location="cpu", weights_only=True)
                train_module.load_state_dict(model_state, strict=False)

    optimizer_state = state.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = state.get("scheduler_state")
    if lr_scheduler is not None and isinstance(scheduler_state, dict):
        with suppress(Exception):
            lr_scheduler.load_state_dict(scheduler_state)

    ema_state = state.get("ema_state")
    if ema_model is not None and isinstance(ema_state, dict):
        with suppress(Exception):
            ema_model.load_state_dict(ema_state)

    py_state = state.get("python_random_state")
    if py_state is not None:
        with suppress(Exception):
            random.setstate(py_state)
    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        with suppress(Exception):
            torch.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda_random_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        with suppress(Exception):
            torch.cuda.set_rng_state_all(cuda_state)

    step = int(state.get("step", 0))
    next_step = max(1, step + 1)
    print(f"[native/diffusion] resumed training from state {resume_state_path} (next step={next_step})")
    return next_step


def _maybe_sample(
    config: dict[str, Any],
    *,
    model_type: ModelType,
    model_path: str,
    output_dir: Path,
    step: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
    live_adapter: Any = None,
    default_resolution: int | None = None,
    default_video_frames: int | None = None,
) -> None:
    sample_block = config.get("sample", {}) if isinstance(config.get("sample"), dict) else {}
    if not sample_block.get("enabled", False):
        return
    use_live_adapter = _as_bool(sample_block.get("use_live_adapter"), True)

    interval = int(sample_block.get("interval", 0) or 0)
    if interval <= 0 or step % interval != 0:
        return

    prompts = _collect_sample_prompts(sample_block)
    if not prompts:
        return

    sample_assistant_lora_path: Path | None = None
    sample_assistant_lora_strength = _optional_float(
        sample_block.get("assistant_lora_inference_strength") or sample_block.get("assistant_lora_strength")
    )
    if use_live_adapter and live_adapter is not None and hasattr(live_adapter, "save"):
        try:
            adapter_cache_dir = output_dir / "samples" / ".adapter_cache"
            adapter_cache_dir.mkdir(parents=True, exist_ok=True)
            sample_assistant_lora_path = adapter_cache_dir / f"step_{step:06d}.safetensors"
            live_adapter.save(str(sample_assistant_lora_path))
            if sample_assistant_lora_strength is None:
                sample_assistant_lora_strength = 1.0
        except (OSError, RuntimeError) as exc:
            print(f"[native/diffusion] warning: failed to snapshot live adapter for sampling: {exc}")
            sample_assistant_lora_path = None

    sampler_model: dict[str, Any] = {"path": model_path}
    if sample_assistant_lora_path is not None:
        sampler_model["assistant_lora_path"] = str(sample_assistant_lora_path)
        if sample_assistant_lora_strength is not None:
            sampler_model["assistant_lora_inference_strength"] = float(sample_assistant_lora_strength)
    elif model_type in {ModelType.ZIMAGE, ModelType.Z_IMAGE}:
        assistant_path = config.get("assistant_lora_path") or config.get("turbo_adapter_path")
        if assistant_path:
            sampler_model["assistant_lora_path"] = str(assistant_path)

        assistant_weight = config.get("assistant_lora_weight_name") or config.get("turbo_adapter_weight_name")
        if assistant_weight:
            sampler_model["assistant_lora_weight_name"] = str(assistant_weight)

        assistant_strength = config.get("assistant_lora_inference_strength")
        if assistant_strength is None:
            assistant_strength = config.get("assistant_lora_strength")
        if assistant_strength is None:
            assistant_strength = config.get("turbo_adapter_strength")
        if assistant_strength is not None:
            sampler_model["assistant_lora_inference_strength"] = float(assistant_strength)

        if "disable_assistant_lora" in config:
            sampler_model["disable_assistant_lora"] = _as_bool(config.get("disable_assistant_lora"), False)

    sampler = create_sampler(model_type, model=sampler_model)
    negative_prompt = str(sample_block.get("negative_prompt", ""))
    seeds = _collect_sample_seeds(sample_block, int(config.get("seed", 42)))
    max_samples = int(sample_block.get("max_samples") or 0)
    prompts_to_run = prompts if max_samples <= 0 else prompts[:max_samples]
    default_ext = ".gif" if model_type == ModelType.LTX2 else ".png"
    output_ext = _resolve_sample_output_extension(sample_block, default_ext)
    sample_height = _optional_int(sample_block.get("height") or sample_block.get("sample_height")) or default_resolution
    sample_width = _optional_int(sample_block.get("width") or sample_block.get("sample_width")) or default_resolution
    sample_steps = _optional_int(
        sample_block.get("num_inference_steps") or sample_block.get("sample_steps") or sample_block.get("steps")
    )
    sample_guidance = _optional_float(sample_block.get("guidance_scale") or sample_block.get("cfg_scale"))

    sample_kwargs: dict[str, Any] = {}
    if model_type == ModelType.QWEN_IMAGE_EDIT:
        image_path = sample_block.get("image_path") or sample_block.get("conditioning_image")
        if image_path:
            sample_kwargs["image_path"] = image_path
    if model_type == ModelType.LTX2:
        sample_num_frames = _optional_int(
            sample_block.get("num_frames")
            or sample_block.get("frames")
            or sample_block.get("video_frames")
            or default_video_frames
        )
        sample_frame_rate = _optional_float(sample_block.get("frame_rate")) or 24.0
        if sample_num_frames is not None:
            sample_kwargs["num_frames"] = sample_num_frames
        sample_kwargs["frame_rate"] = sample_frame_rate

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for prompt_index, prompt in enumerate(prompts_to_run):
        seed = seeds[prompt_index % len(seeds)]
        out_path = samples_dir / f"step_{step:06d}_p{prompt_index:02d}_s{seed}{output_ext}"
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_path": model_path,
            "seed": seed,
            "device": train_device,
            "dtype": train_dtype,
            "output_path": out_path,
            "unload": prompt_index == len(prompts_to_run) - 1,
            **sample_kwargs,
        }
        if sample_height is not None:
            call_kwargs["height"] = sample_height
        if sample_width is not None:
            call_kwargs["width"] = sample_width
        if sample_steps is not None:
            call_kwargs["num_inference_steps"] = sample_steps
        if sample_guidance is not None:
            call_kwargs["guidance_scale"] = sample_guidance

        sampler.sample(
            **call_kwargs,
        )


def _get_train_module(pipeline: Any, family: str) -> torch.nn.Module:
    model_impl = _create_native_model_for_family(family)
    return model_impl.get_train_module(pipeline)


def _setup_memory_strategy(
    train_module: torch.nn.Module,
    family: str,
    memory_block: dict[str, Any],
    config: dict[str, Any],
) -> LayerOffloadStrategy | None:
    activation_offloading = _as_bool(
        memory_block.get("enable_activation_offloading", config.get("enable_activation_offloading", False))
    )
    gradient_checkpointing = str(
        memory_block.get("gradient_checkpointing") or config.get("gradient_checkpointing") or "off"
    )
    layer_offload_fraction = float(
        memory_block.get("layer_offload_fraction", config.get("layer_offload_fraction", 0.0))
    )
    blocks_to_swap = int(memory_block.get("blocks_to_swap", config.get("blocks_to_swap", 0)) or 0)

    if blocks_to_swap > 0:
        total_blocks = 0
        for attr in ("transformer_blocks", "single_transformer_blocks", "blocks", "layers"):
            maybe_layers = getattr(train_module, attr, None)
            if isinstance(maybe_layers, torch.nn.ModuleList | list):
                total_blocks += len(maybe_layers)
        if total_blocks > 0:
            layer_offload_fraction = max(layer_offload_fraction, min(float(blocks_to_swap) / float(total_blocks), 1.0))

    if activation_offloading and gradient_checkpointing.strip().lower() == "on":
        gradient_checkpointing = "cpu_offloaded"

    # Qwen can exceed 24GB quickly without quantization; default to an aggressive offload floor.
    if family == "qwen" and layer_offload_fraction < 0.97:
        layer_offload_fraction = 0.97
    # LTX follows video-scale block swapping patterns; keep near-full offload by default.
    if family == "ltx2" and layer_offload_fraction < 0.995:
        layer_offload_fraction = 0.995

    if gradient_checkpointing.lower() != "off" and hasattr(train_module, "enable_gradient_checkpointing"):
        with suppress(Exception):
            train_module.enable_gradient_checkpointing()

    if family not in _FLOW_FAMILIES:
        return None

    strategy = LayerOffloadStrategy(
        MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=layer_offload_fraction,
            gradient_checkpointing=gradient_checkpointing,
            enable_activation_offloading=activation_offloading,
            enable_async_offloading=_as_bool(
                memory_block.get("enable_async_offloading", config.get("enable_async_offloading", False))
            ),
            train_device=str(config.get("train_device", "cuda")),
            temp_device=str(config.get("temp_device", "cpu")),
        )
    )

    try:
        strategy.setup(SimpleNamespace(transformer=train_module))
        return strategy
    except (ImportError, RuntimeError) as exc:
        print(f"[native/diffusion] warning: memory strategy setup skipped ({exc})")
        return None


def _move_non_offloaded_tensors_to_device(
    train_module: torch.nn.Module,
    *,
    conductor: Any,
    train_device: torch.device,
) -> None:
    offloaded_tensor_ids: set[int] = set()
    for state in getattr(conductor, "_layers", []):
        for param in state.layer.parameters(recurse=True):
            offloaded_tensor_ids.add(id(param))
        for buffer in state.layer.buffers(recurse=True):
            offloaded_tensor_ids.add(id(buffer))

    with torch.no_grad():
        for param in train_module.parameters():
            if id(param) in offloaded_tensor_ids:
                continue
            if param.device != train_device:
                param.data = param.data.to(device=train_device, non_blocking=True)

        for buffer_name, buffer in train_module.named_buffers():
            if id(buffer) in offloaded_tensor_ids:
                continue
            if buffer.device == train_device:
                continue

            parent_name, _, local_name = buffer_name.rpartition(".")
            parent_module = train_module.get_submodule(parent_name) if parent_name else train_module
            parent_module._buffers[local_name] = buffer.to(device=train_device, non_blocking=True)


def _call_with_filtered_kwargs(fn, *args: Any, **kwargs: Any):
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(*args, **kwargs)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return fn(*args, **filtered_kwargs)


def _is_lora_adapter_state_dict(state_dict: dict[str, Any]) -> bool:
    if not state_dict:
        return False
    return any("lora_" in key.lower() for key in state_dict)


def _freeze_named_adapter_params(module: torch.nn.Module, adapter_name: str) -> None:
    marker = f".{adapter_name}"
    for name, param in module.named_parameters():
        if marker in name and "lora_" in name:
            param.requires_grad_(False)


def _maybe_load_transformer_override(
    train_module: torch.nn.Module,
    config: dict[str, Any],
    *,
    pipeline: Any | None = None,
    family: str | None = None,
) -> None:
    override_path = config.get("turbo_adapter_path") or config.get("transformer_weights")
    if not override_path:
        return

    path = Path(str(override_path)).expanduser()
    if not path.exists():
        print(f"[native/diffusion] warning: override weights not found: {path}")
        return

    try:
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

        if family == "zimage" and pipeline is not None and _is_lora_adapter_state_dict(state_dict):
            pipeline_cls = pipeline.__class__
            load_fn = getattr(pipeline_cls, "load_lora_into_transformer", None)
            if load_fn is None:
                print(
                    "[native/diffusion] warning: Z-Image pipeline does not expose "
                    "`load_lora_into_transformer`; turbo adapter was skipped."
                )
                return

            load_kwargs: dict[str, Any] = {
                "transformer": train_module,
                "adapter_name": "assistant",
                "_pipeline": None,
                "low_cpu_mem_usage": False,
            }
            _call_with_filtered_kwargs(load_fn, state_dict, **load_kwargs)
            _freeze_named_adapter_params(train_module, "assistant")

            assistant_strength_raw = config.get("assistant_lora_strength")
            if assistant_strength_raw is None:
                assistant_strength_raw = config.get("turbo_adapter_strength")
            assistant_strength = _optional_float(assistant_strength_raw)
            with suppress(Exception):
                if hasattr(train_module, "set_adapters"):
                    if assistant_strength is None:
                        train_module.set_adapters(["assistant"])
                    else:
                        train_module.set_adapters(["assistant"], [float(assistant_strength)])
                elif hasattr(train_module, "set_adapter"):
                    train_module.set_adapter("assistant")

            print(f"[native/diffusion] loaded Z-Image assistant LoRA from {path}")
            return

        missing, unexpected = train_module.load_state_dict(state_dict, strict=False)
        print(
            "[native/diffusion] loaded transformer override "
            f"from {path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    except (OSError, RuntimeError) as exc:
        print(f"[native/diffusion] warning: failed to load override weights {path}: {exc}")


def _run_sd15_vae_finetune(
    *,
    config: dict[str, Any],
    checkpoint_block: dict[str, Any],
    optimizer_block: dict[str, Any],
    scheduler_block: dict[str, Any],
    pipeline: Any,
    model_type_enum: ModelType,
    model_path: str,
    output_dir: Path,
    pairs: list[tuple[Path, str]],
    resolution: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
    save_dtype: torch.dtype,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
) -> int:
    vae = pipeline.vae
    vae.to(train_device)
    _set_module_train_state(vae, True)

    params = [param for param in vae.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable VAE parameters found for fine_tune_vae mode")

    optimizer, optimizer_name = _create_optimizer(
        params,
        config=config,
        optimizer_block=optimizer_block,
        learning_rate=learning_rate,
    )
    total_optimizer_steps = _resolve_optimizer_steps(max_steps, grad_accum)
    lr_scheduler, scheduler_name = _create_lr_scheduler(
        optimizer,
        config=config,
        scheduler_block=scheduler_block,
        total_optimizer_steps=total_optimizer_steps,
    )
    print(f"[native/diffusion/vae] optimizer={optimizer_name} lr_scheduler={scheduler_name}")

    ema_mode = str(config.get("ema_mode", "off")).strip().lower()
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_model: EMAModel | None = None
    if ema_mode != EMAMode.OFF.value:
        ema_model = EMAModel(modules=[vae], decay=ema_decay)

    resume_state_path = _resolve_resume_state_path(config, checkpoint_block, output_dir)
    start_step = _maybe_restore_training_state(
        resume_state_path=resume_state_path,
        train_module=vae,
        adapter=None,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
    )

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(
        checkpoint_block.get("save_every") or config.get("save_every") or config.get("save_every_n_steps") or 0
    )
    kl_weight = float(config.get("vae_kl_weight", 1e-6))

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}

    for step in range(start_step, max_steps + 1):
        selected = random.choices(pairs, k=max(1, batch_size))
        pixel_values = torch.stack(
            [_load_image_tensor(image_path, resolution, train_dtype, train_device) for image_path, _ in selected],
            dim=0,
        )

        with torch.autocast(device_type=train_device.type, dtype=train_dtype, enabled=autocast_enabled):
            loss = _compute_vae_loss(
                vae,
                pixel_values,
                kl_weight=kl_weight,
            )

        scaled_loss = loss / float(max(grad_accum, 1))
        scaled_loss.backward()

        should_step = (step % max(grad_accum, 1) == 0) or (step == max_steps)
        if should_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema_model is not None:
                ema_model.update()
            optimizer.zero_grad(set_to_none=True)

        if step == start_step or step % 10 == 0 or step == max_steps:
            print(f"[native/diffusion/vae] step {step}/{max_steps} loss={float(loss.detach().cpu()):.6f}")

        checkpoint_path: Path | None = None
        if save_every > 0 and step % save_every == 0:
            checkpoint_path = output_dir / f"vae_step_{step:06d}.safetensors"
            saved_path = _save_module_state(vae, checkpoint_path, save_dtype=save_dtype)
            print(f"[native/diffusion/vae] saved VAE checkpoint to {saved_path}")
            state_path = output_dir / f"state_step_{step:06d}.pt"
            _save_training_state(
                state_path,
                step=step,
                model_checkpoint=saved_path,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema_model=ema_model,
            )

    final_path = output_dir / "vae_last.safetensors"
    saved_path = _save_module_state(vae, final_path, save_dtype=save_dtype)
    final_state_path = output_dir / f"state_step_{max_steps:06d}.pt"
    _save_training_state(
        final_state_path,
        step=max_steps,
        model_checkpoint=saved_path,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
    )
    print(f"[native/diffusion/vae] training complete, VAE saved to {saved_path}")
    print(
        "[native/diffusion/vae] note: validation sampling in this mode is skipped because sampler reloads from "
        f"base model path ({model_path}) without injecting fine-tuned VAE weights."
    )
    return 0


def run_native_diffusion_training(
    config: dict[str, Any],
    *,
    source_path: Path,
    steps_override: int | None = None,
) -> int:
    model_block = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    data_block = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    memory_block = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    checkpoint_block = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}
    optimizer_block = config.get("optimizer", {}) if isinstance(config.get("optimizer"), dict) else {}
    scheduler_block = config.get("scheduler", {}) if isinstance(config.get("scheduler"), dict) else {}

    normalized_model_type = _normalize_model_type(config.get("model_type") or model_block.get("type"))
    if not is_native_diffusion_model_type(normalized_model_type):
        raise ValueError(f"Unsupported native diffusion model type: {normalized_model_type}")
    training_method = _normalize_training_method(config.get("training_method"))
    if training_method == "embedding":
        raise ValueError(
            "Native embedding training is not implemented yet for diffusion backends. "
            "Use adapter/fine_tune mode or run bridge mode explicitly."
        )

    family = _resolve_family(normalized_model_type)
    native_model = _create_native_model(normalized_model_type)
    model_type_enum = _resolve_sampler_model_type(normalized_model_type)

    model_path_raw = (
        model_block.get("path")
        or config.get("base_model")
        or config.get("base_model_name")
        or config.get("model_path")
        or config.get("transformer_path")
    )
    if not model_path_raw:
        raise ValueError("Missing model path. Expected model.path/base_model/base_model_name/model_path.")

    resolved_model_path = _resolve_model_path(str(model_path_raw))
    quantization_mode = _normalize_quantization_mode(memory_block.get("quantization") or config.get("quantization"))
    requested_gpu_budget_gib = memory_block.get("quantized_gpu_memory_gib") or config.get("quantized_gpu_memory_gib")
    if requested_gpu_budget_gib is not None:
        requested_gpu_budget_gib = int(requested_gpu_budget_gib)

    train_dtype = _coerce_dtype(config.get("train_dtype") or model_block.get("dtype") or "bfloat16")
    save_dtype = _coerce_dtype(config.get("output_dtype"), default=train_dtype)
    train_device = torch.device(str(config.get("train_device", "cuda")))
    distributed_context, train_device = _resolve_distributed_context(
        config=config,
        model_block=model_block,
        requested_train_device=train_device,
    )

    seed = int(config.get("seed", 42))
    if distributed_context.enabled:
        seed += int(distributed_context.rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    learning_rate = float(config.get("learning_rate", 1e-4))
    batch_size = int(data_block.get("batch_size") or config.get("batch_size") or 1)
    grad_accum = int(config.get("gradient_accumulation_steps") or config.get("gradient_accumulation") or 1)
    max_steps = int(
        steps_override
        if steps_override is not None and steps_override > 0
        else (config.get("max_steps") or config.get("max_train_steps") or 100)
    )

    raw_resolution = int(data_block.get("resolution") or config.get("resolution") or 1024)
    resolution_multiple = int(getattr(native_model, "resolution_multiple", _resolution_multiple_for_family(family)))
    resolution = _quantize_resolution(raw_resolution, resolution_multiple)

    output_dir = Path(
        checkpoint_block.get("output_dir") or config.get("output_dir") or (Path("output") / source_path.stem)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ltx_component_paths: dict[str, str] = {}
    ltx_template_path: str | None = None
    if family == "ltx2":
        ltx_component_paths, ltx_template_path = _collect_ltx_component_paths(config, model_block)

    adapter_type, adapter_block = _extract_adapter_config(config)
    if training_method in _FULL_TRAINING_METHODS:
        adapter_type, adapter_block = "full", {}
    full_finetune = adapter_type == "full"
    is_vae_finetune = training_method == "fine_tune_vae"
    if is_vae_finetune and family != "sd15":
        raise ValueError(
            f"Native fine_tune_vae is currently only supported for sd15 family, got '{normalized_model_type}'."
        )
    if full_finetune and quantization_mode in {"int8", "fp8"} and family == "qwen":
        raise ValueError(
            "Quantized Qwen modes (INT8/FP8) are not supported for native full-finetune mode. "
            "Use an adapter mode."
        )

    pairs = _build_training_pairs(config)
    allow_video_dataset = family == "ltx2"
    ltx_video_frame_count = 1
    if allow_video_dataset:
        ltx_video_frame_count = _resolve_ltx_video_frame_count(config, data_block)
        requested_frames = int(
            data_block.get("frames")
            or data_block.get("video_frames")
            or data_block.get("num_frames")
            or config.get("frames")
            or config.get("video_frames")
            or config.get("num_frames")
            or ltx_video_frame_count
        )
        if ltx_video_frame_count != requested_frames:
            print(
                "[native/diffusion] adjusted LTX2 frame count "
                f"from {requested_frames} to {ltx_video_frame_count} (must satisfy frames % 8 == 1)"
            )
        else:
            print(f"[native/diffusion] using LTX2 frame count {ltx_video_frame_count}")

        pairs.extend(_build_video_training_pairs(config))
        deduped: dict[Path, str] = dict(pairs)
        pairs = list(deduped.items())

    if not pairs:
        if allow_video_dataset:
            raise ValueError("No training media/captions found in configured concepts (expected images or videos).")
        raise ValueError("No training images/captions found in configured concepts.")
    pairs = _prepare_training_pairs(pairs, model_type=model_type_enum)
    if not pairs:
        raise ValueError("No training pairs remain after model-specific preprocessing.")

    if steps_override is not None and steps_override > 0:
        max_required = max(1, max_steps * max(1, batch_size) * max(1, grad_accum))
        pairs = pairs[:max_required]

    custom_conditioning_image_raw = config.get("custom_conditioning_image") or data_block.get("custom_conditioning_image")
    custom_conditioning_image = None
    if custom_conditioning_image_raw:
        candidate = Path(str(custom_conditioning_image_raw)).expanduser()
        if candidate.exists() and candidate.is_file():
            custom_conditioning_image = candidate
        else:
            print(f"[native/diffusion] warning: custom conditioning image not found: {candidate}")

    print(f"[native/diffusion] loading {family} model from {resolved_model_path}")
    pipeline = native_model.load_pipeline(
        resolved_model_path,
        train_dtype,
        train_device=train_device,
        quantization_mode=quantization_mode,
        requested_gpu_budget_gib=requested_gpu_budget_gib,
        ltx_component_paths=ltx_component_paths,
        ltx_template_path=ltx_template_path,
    )
    if is_vae_finetune:
        if distributed_context.enabled:
            _finalize_distributed_context(distributed_context)
            raise ValueError("fine_tune_vae mode is not supported with distributed native training.")
        try:
            return _run_sd15_vae_finetune(
                config=config,
                checkpoint_block=checkpoint_block,
                optimizer_block=optimizer_block,
                scheduler_block=scheduler_block,
                pipeline=pipeline,
                model_type_enum=model_type_enum,
                model_path=resolved_model_path,
                output_dir=output_dir,
                pairs=pairs,
                resolution=resolution,
                train_device=train_device,
                train_dtype=train_dtype,
                save_dtype=save_dtype,
                max_steps=max_steps,
                batch_size=batch_size,
                grad_accum=grad_accum,
                learning_rate=learning_rate,
            )
        finally:
            _finalize_distributed_context(distributed_context)

    train_module = native_model.get_train_module(pipeline)
    dispatched_train_module = _is_dispatched_module(train_module)

    train_primary, text_train_flags, train_vae = _resolve_component_train_flags(
        config=config,
        model_block=model_block,
        family=family,
        full_finetune=full_finetune,
        training_method=training_method,
    )
    if not train_primary and adapter_type == "full":
        print("[native/diffusion] warning: primary train module disabled; enabling it to keep objective valid.")
        train_primary = True

    cache_text_embeddings = _as_bool(
        data_block.get("cache_text_embeddings", config.get("cache_text_embeddings", True)),
        True,
    )
    text_encoder_training_active = any(text_train_flags.values())
    if text_encoder_training_active and cache_text_embeddings:
        cache_text_embeddings = False
        print("[native/diffusion] disabled cache_text_embeddings because text encoder training is enabled")
    if distributed_context.enabled and text_encoder_training_active:
        _finalize_distributed_context(distributed_context)
        raise ValueError(
            "Distributed native diffusion currently supports training only the primary module/adapters; "
            "text encoder training must be disabled."
        )

    _maybe_load_transformer_override(train_module, config, pipeline=pipeline, family=family)

    _set_module_train_state(train_module, full_finetune and train_primary)
    if train_vae:
        print(
            "[native/diffusion] warning: train_vae requested outside fine_tune_vae mode; "
            "VAE parameters remain frozen in diffusion objective mode."
        )

    text_encoders = {attr: getattr(pipeline, attr, None) for attr in ("text_encoder", "text_encoder_2", "text_encoder_3")}
    for attr, module in text_encoders.items():
        should_train = bool(text_train_flags.get(attr, False))
        _set_module_train_state(module, should_train)
        if should_train and module is not None:
            module.to(train_device)

    print(f"[native/diffusion] caching latents+text embeddings for {len(pairs)} samples")
    cached = _cache_training_data(
        native_model,
        pipeline,
        family,
        pairs,
        model_type_enum,
        custom_conditioning_image,
        resolution,
        train_device,
        train_dtype,
        allow_video=allow_video_dataset,
        video_frame_count=ltx_video_frame_count,
        cache_text_embeddings=cache_text_embeddings,
        keep_text_encoder_on_device=text_encoder_training_active,
    )
    if not cached:
        raise ValueError("No cached samples were produced. Verify dataset paths and conditioning images.")

    memory_strategy = (
        None if dispatched_train_module else _setup_memory_strategy(train_module, family, memory_block, config)
    )
    if dispatched_train_module:
        print("[native/diffusion] detected dispatched transformer; skipping manual layer-offload placement")
        gradient_checkpointing = (
            str(memory_block.get("gradient_checkpointing") or config.get("gradient_checkpointing") or "off")
            .strip()
            .lower()
        )
        if gradient_checkpointing != "off" and hasattr(train_module, "enable_gradient_checkpointing"):
            with suppress(Exception):
                train_module.enable_gradient_checkpointing()
                print("[native/diffusion] enabled native gradient checkpointing on dispatched transformer")

    placed_on_device = False
    if memory_strategy is not None and memory_strategy.conductor is not None:
        conductor = memory_strategy.conductor
        if conductor.offload_activated():
            temp_device = torch.device(str(config.get("temp_device", "cpu")))
            conductor.to(temp_device)
            _move_non_offloaded_tensors_to_device(
                train_module,
                conductor=conductor,
                train_device=train_device,
            )
            placed_on_device = True
            print(
                "[native/diffusion] keeping module on temp device for layer/activation offload "
                f"(temp_device={temp_device}, train_device={train_device})"
            )

    if not placed_on_device and not dispatched_train_module:
        with suppress(Exception):
            train_module.to(train_device)

    if full_finetune and train_primary:
        train_module.train()
    else:
        train_module.eval()

    adapter = None
    if not full_finetune:
        rank = int(adapter_block.get("rank") or adapter_block.get("network_dim") or config.get("lora_rank") or 16)
        alpha = float(
            adapter_block.get("alpha") or adapter_block.get("network_alpha") or config.get("lora_alpha") or rank
        )
        adapter_backend: str | None = None
        if adapter_type == "lora":
            if isinstance(config.get("lycoris"), dict):
                adapter_backend = "lycoris"
            else:
                adapter_backend = str(
                    adapter_block.get("backend")
                    or adapter_block.get("implementation")
                    or config.get("lora_backend")
                    or config.get("adapter_backend")
                    or "native"
                ).strip().lower()
        adapter = create_adapter(
            adapter_type=adapter_type,
            rank=rank,
            alpha=alpha,
            model_type=model_type_enum.value,
            dropout=float(adapter_block.get("dropout", 0.0)),
            backend=adapter_backend,
            **_build_adapter_kwargs(adapter_block),
        )
        adapter.inject(train_module)

    offload_active = bool(
        memory_strategy is not None
        and memory_strategy.conductor is not None
        and memory_strategy.conductor.offload_activated()
    )
    if adapter is not None and hasattr(adapter, "to") and not dispatched_train_module and not offload_active:
        adapter.to(train_device, dtype=train_dtype)

    if distributed_context.enabled:
        if dispatched_train_module:
            _finalize_distributed_context(distributed_context)
            raise ValueError("Distributed native diffusion does not support dispatched train modules yet.")
        if offload_active:
            _finalize_distributed_context(distributed_context)
            raise ValueError("Distributed native diffusion is incompatible with layer/activation offloading.")

    train_forward_module = _maybe_wrap_distributed_module(
        train_module,
        distributed_context,
        train_device,
        find_unused_parameters=bool(not train_primary),
    )

    trainable_modules: list[torch.nn.Module] = []
    if full_finetune and train_primary:
        trainable_modules.append(train_module)
    if full_finetune:
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = text_encoders.get(attr)
            if module is not None and text_train_flags.get(attr, False):
                trainable_modules.append(module)

    if adapter is not None:
        params = [param for param in adapter.get_trainable_params() if param.requires_grad]
    else:
        params = []
        for module in trainable_modules:
            params.extend(param for param in module.parameters() if param.requires_grad)

    if not params:
        mode = "full-finetune" if full_finetune else f"adapter ({adapter_type})"
        raise RuntimeError(f"No trainable parameters found for {mode} mode")

    optimizer, optimizer_name = _create_optimizer(
        params,
        config=config,
        optimizer_block=optimizer_block,
        learning_rate=learning_rate,
    )
    total_optimizer_steps = _resolve_optimizer_steps(max_steps, grad_accum)
    lr_scheduler, scheduler_name = _create_lr_scheduler(
        optimizer,
        config=config,
        scheduler_block=scheduler_block,
        total_optimizer_steps=total_optimizer_steps,
    )
    print(f"[native/diffusion] optimizer={optimizer_name} lr_scheduler={scheduler_name}")

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(
        checkpoint_block.get("save_every") or config.get("save_every") or config.get("save_every_n_steps") or 0
    )
    save_full_model = _as_bool(checkpoint_block.get("save_full_model", config.get("save_full_model", True)), True)
    ema_mode = str(config.get("ema_mode", "off")).strip().lower()
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_model: EMAModel | None = None
    if ema_mode != EMAMode.OFF.value:
        if adapter is not None:
            print("[native/diffusion] warning: EMA is currently applied only to full-finetune modules; skipping for adapter mode")
        elif trainable_modules:
            ema_model = EMAModel(modules=trainable_modules, decay=ema_decay)

    resume_state_path = _resolve_resume_state_path(config, checkpoint_block, output_dir)
    start_step = _maybe_restore_training_state(
        resume_state_path=resume_state_path,
        train_module=train_module,
        adapter=adapter,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
    )
    if start_step > max_steps:
        print(
            f"[native/diffusion] resume state step exceeds max_steps ({start_step}>{max_steps}); nothing to do."
        )
        _finalize_distributed_context(distributed_context)
        return 0

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}

    for step in range(start_step, max_steps + 1):
        batch = _pick_batch(cached, batch_size, train_device, train_dtype, family)
        if batch.prompt_embeds is None:
            batch = _materialize_batch_prompt_features(
                model_impl=native_model,
                pipeline=pipeline,
                family=family,
                model_type=model_type_enum,
                batch=batch,
                train_device=train_device,
                train_dtype=train_dtype,
                requires_grad=text_encoder_training_active,
            )

        forward_ctx = memory_strategy.forward_context() if memory_strategy is not None else nullcontext()
        with forward_ctx:
            with torch.autocast(device_type=train_device.type, dtype=train_dtype, enabled=autocast_enabled):
                loss = _compute_loss(
                    native_model,
                    pipeline,
                    family,
                    batch,
                    train_forward_module,
                    train_dtype,
                    config,
                )

            scaled_loss = loss / float(max(grad_accum, 1))
            scaled_loss.backward()

        should_step = (step % max(grad_accum, 1) == 0) or (step == max_steps)
        if should_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema_model is not None:
                ema_model.update()
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"[native/diffusion] step {step}/{max_steps} loss={float(loss.detach().cpu()):.6f}")

        if distributed_context.is_main_process:
            _maybe_sample(
                config,
                model_type=model_type_enum,
                model_path=resolved_model_path,
                output_dir=output_dir,
                step=step,
                train_device=train_device,
                train_dtype=train_dtype,
                live_adapter=adapter,
                default_resolution=resolution,
                default_video_frames=ltx_video_frame_count if allow_video_dataset else None,
            )

        if distributed_context.is_main_process and save_every > 0 and step % save_every == 0:
            saved_path: Path | None = None
            if adapter is not None:
                ckpt_path = output_dir / f"{adapter_type}_step_{step:06d}.safetensors"
                adapter.save(str(ckpt_path))
                saved_path = ckpt_path
            elif save_full_model:
                stem = "unet" if family in {"sd15", "sdxl"} else "transformer"
                ckpt_path = output_dir / f"{stem}_step_{step:06d}.safetensors"
                saved_path = _save_module_state(train_module, ckpt_path, save_dtype=save_dtype)
                print(f"[native/diffusion] saved full checkpoint to {saved_path}")
            if saved_path is not None:
                state_path = output_dir / f"state_step_{step:06d}.pt"
                _save_training_state(
                    state_path,
                    step=step,
                    model_checkpoint=saved_path,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema_model=ema_model,
                )

    _distributed_barrier(distributed_context)

    if memory_strategy is not None:
        memory_strategy.cleanup()

    final_saved_path: Path | None = None
    if distributed_context.is_main_process and adapter is not None:
        final_path = output_dir / f"{adapter_type}_last.safetensors"
        adapter.save(str(final_path))
        print(f"[native/diffusion] training complete, adapter saved to {final_path}")
        final_saved_path = final_path
    elif distributed_context.is_main_process and save_full_model:
        stem = "unet" if family in {"sd15", "sdxl"} else "transformer"
        final_path = output_dir / f"{stem}_last.safetensors"
        saved_path = _save_module_state(train_module, final_path, save_dtype=save_dtype)
        print(f"[native/diffusion] training complete, full module saved to {saved_path}")
        final_saved_path = saved_path
    elif distributed_context.is_main_process:
        print("[native/diffusion] training complete (save_full_model=false, no full checkpoint written)")

    if distributed_context.is_main_process and final_saved_path is not None:
        final_state_path = output_dir / f"state_step_{max_steps:06d}.pt"
        _save_training_state(
            final_state_path,
            step=max_steps,
            model_checkpoint=final_saved_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
        )

    with suppress(Exception):
        native_model.offload_text_encoders(pipeline)

    _finalize_distributed_context(distributed_context)
    return 0
