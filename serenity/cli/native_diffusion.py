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
import os
import random
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from serenity.adapters import create_adapter
from serenity.cli.utils import (
    as_bool as _as_bool,
    build_training_pairs as _build_training_pairs,
    coerce_dtype as _coerce_dtype,
    collect_concept_dirs as _collect_concept_dirs,
    load_caption as _load_caption,
    normalize_model_type as _normalize_model_type,
    optional_float as _optional_float,
    optional_int as _optional_int,
    resolve_hf_local_path as _resolve_hf_local_path,
)
from serenity.cli.native_flux2 import (
    _build_adapter_kwargs,
    _collect_sample_prompts,
    _collect_sample_seeds,
    _extract_adapter_config,
    _resolve_sample_output_extension,
)
from serenity.cli.diffusion_data import (
    _CachedExample,
    _Batch,
    _build_video_training_pairs,
    _cache_training_data,
    _is_qwen_edit_type,
    _is_video_file,
    _load_media_tensor,
    _load_video_tensor,
    _materialize_batch_prompt_features,
    _pick_batch,
    _prepare_training_pairs,
    _resolve_component_train_flags,
    _resolve_conditioning_image_path,
    _set_module_train_state,
)
from serenity.cli.diffusion_losses import (
    _FLOW_FAMILIES,
    _compute_loss,
    _compute_vae_loss,
    _qwen_pack_latents,
    _qwen_unpack_latents,
    _sample_flow_timesteps,
)
from serenity.cli.diffusion_checkpoint import (
    _maybe_restore_training_state,
    _resolve_resume_state_path,
    _save_module_state,
    _save_training_state,
)
from serenity.cli.flux2_optimizer import (
    _create_lr_scheduler,
    _create_optimizer,
    _resolve_optimizer_steps,
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
from torch.nn.parallel import DistributedDataParallel

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

_FULL_TRAINING_METHODS = {"full", "full_finetune", "full_fine_tune", "fine_tune", "finetune", "fine_tune_vae"}


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


from serenity.cli.utils import (
    load_image_tensor as _load_image_tensor,
)


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


# _load_image_tensor — imported from serenity.cli.utils above


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
