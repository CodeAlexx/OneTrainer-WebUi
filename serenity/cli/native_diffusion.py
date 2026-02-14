"""Native non-bridge diffusion training path for Serenity.

Supports native adapter/full training for:
- Flux 1.x (dev/schnell/fill)
- Flux 2.x (dev path)
- Z-Image
- Qwen Image / Qwen Image Edit
- Stable Diffusion 1.5
- Stable Diffusion XL
- Stable Diffusion 3 / 3.5
- WAN 2.1 / 2.2 (14B)
"""

from __future__ import annotations

import inspect
import os
import random
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
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
    _sample_flow_timesteps_bounded,
)
from serenity.cli.diffusion_checkpoint import (
    _load_dual_stage_state,
    _maybe_restore_training_state,
    _resolve_resume_state_path,
    _save_dual_stage_state,
    _save_module_state,
    _save_training_state,
)
from serenity.cli.flux2_optimizer import (
    _create_lr_scheduler,
    _create_optimizer,
    _resolve_optimizer_steps,
)
from serenity.core.interfaces import ModelType
from serenity.memory.stagehand_strategy import StagehandStrategy, StagehandStrategyConfig
from serenity.memory.strategy import LayerOffloadStrategy, MemoryConfig, MemoryStrategy
from serenity.models.flux1 import Flux1Model
from serenity.models.flux2 import Flux2Model
from serenity.models.flux_schnell import FluxSchnellModel
from serenity.models.ltx2 import LTX2Model
from serenity.models.qwen import QwenBaseModel, QwenImageEditModel, QwenModel
from serenity.models.sd3 import SD3Model, SD35Model
from serenity.models.sd15 import SD15Model
from serenity.models.sdxl import SDXLModel
from serenity.models.wan import WanModel
from serenity.models.zimage import ZImageModel
from serenity.sampling.sampler import create_sampler
from serenity.training.ema import EMAMode, EMAModel

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

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

_WAN_TYPES = {
    "wan",
    "wan21",
    "wan22",
    "wan_2_1",
    "wan_2_2",
    "wan22_14b",
    "wan22_t2v_high",
    "wan22_t2v_low",
    "wan22_i2v_high",
    "wan22_i2v_low",
    "wan22_high",
    "wan22_low",
    "wan22_dual",
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


def _format_eta(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        with suppress(Exception):
            return float(value.detach().float().cpu().item())
        return None
    with suppress(Exception):
        return float(value)
    return None


def _resolve_progress_desc(config: dict[str, Any], default: str) -> str:
    candidates = (
        config.get("tracker_run_name"),
        config.get("run_name"),
        config.get("job_name"),
        config.get("name"),
        Path(str(config.get("output_dir", "") or "")).name,
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return default


def _create_train_progress(
    config: dict[str, Any],
    *,
    start_step: int,
    max_steps: int,
    default_desc: str,
) -> tqdm | None:
    if not _as_bool(config.get("progress_bar"), True):
        return None
    desc = _resolve_progress_desc(config, default_desc)
    initial_step = max(0, min(max_steps, int(start_step) - 1))
    return tqdm(total=max_steps, initial=initial_step, desc=desc, dynamic_ncols=True, leave=True)


def _progress_write(progress: tqdm | None, message: str) -> None:
    if progress is not None:
        progress.write(message)
        return
    print(message)


def _current_vram_mb(device: torch.device | None = None) -> tuple[float, float] | None:
    if not torch.cuda.is_available():
        return None
    device_index: int | None = None
    if device is not None and device.type == "cuda":
        device_index = int(device.index) if device.index is not None else None
    if device_index is None:
        with suppress(Exception):
            device_index = int(torch.cuda.current_device())
    if device_index is None:
        return None
    with suppress(Exception):
        allocated_mb = float(torch.cuda.memory_allocated(device_index) / 1024**2)
        reserved_mb = float(torch.cuda.memory_reserved(device_index) / 1024**2)
        return allocated_mb, reserved_mb
    return None


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
        | _WAN_TYPES
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
    if model_type in _WAN_TYPES:
        return "wan"
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
    if model_type in _WAN_TYPES:
        return WanModel()
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
    if family == "wan":
        return WanModel()
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
    if model_type in _WAN_TYPES:
        return ModelType.WAN
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
    if family in {"sd3", "wan"}:
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


def _resolve_wan_video_frame_count(config: dict[str, Any], data_block: dict[str, Any]) -> int:
    """Resolve WAN video frame count satisfying (frames - 1) % 4 == 0."""
    requested = int(
        data_block.get("frames")
        or data_block.get("video_frames")
        or data_block.get("num_frames")
        or config.get("frames")
        or config.get("video_frames")
        or config.get("num_frames")
        or 17
    )
    if (requested - 1) % 4 != 0:
        adjusted = ((requested - 1) // 4) * 4 + 1
        return max(1, adjusted)
    return max(1, requested)


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
    sample_device_raw = sample_block.get("sample_device") or sample_block.get("device")
    if sample_device_raw:
        sample_device = torch.device(str(sample_device_raw))
    else:
        sample_device = train_device
    sample_dtype = train_dtype if sample_device.type != "cpu" else torch.float32

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
        sample_steps_value = int(sample_steps) if sample_steps is not None else int(sampler.default_steps)
        print(
            "[native/diffusion] sampling start "
            f"step={step} prompt={prompt_index + 1}/{len(prompts_to_run)} "
            f"device={sample_device.type} size={sample_width}x{sample_height} "
            f"steps={sample_steps_value} output={out_path}",
            flush=True,
        )
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_path": model_path,
            "seed": seed,
            "device": sample_device,
            "dtype": sample_dtype,
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
        print(
            "[native/diffusion] sampling complete "
            f"step={step} prompt={prompt_index + 1}/{len(prompts_to_run)} output={out_path}",
            flush=True,
        )


def _get_train_module(pipeline: Any, family: str) -> torch.nn.Module:
    model_impl = _create_native_model_for_family(family)
    return model_impl.get_train_module(pipeline)


def _move_non_block_submodules_to_device(
    train_module: torch.nn.Module,
    train_device: torch.device,
) -> None:
    """Move non-block submodules to GPU for Stagehand training.

    Stagehand manages the transformer blocks via hooks, so they stay on CPU.
    Everything else (patch embedding, condition embedder, norms, projections)
    must live on GPU for the forward pass to work.
    """
    block_attrs = {"transformer_blocks", "single_transformer_blocks", "blocks", "layers"}
    block_module_ids: set[int] = set()

    for attr in block_attrs:
        block_container = getattr(train_module, attr, None)
        if block_container is None:
            continue
        for block in block_container:
            block_module_ids.add(id(block))
            for param in block.parameters(recurse=True):
                block_module_ids.add(id(param))
            for buf in block.buffers(recurse=True):
                block_module_ids.add(id(buf))

    with torch.no_grad():
        for name, child in train_module.named_children():
            if name in block_attrs:
                continue
            child.to(train_device)

        # Move any top-level parameters/buffers that aren't in blocks.
        for param in train_module.parameters(recurse=False):
            if id(param) not in block_module_ids and param.device != train_device:
                param.data = param.data.to(device=train_device, non_blocking=True)
        for buf_name, buf in train_module.named_buffers(recurse=False):
            if id(buf) not in block_module_ids and buf.device != train_device:
                train_module._buffers[buf_name] = buf.to(device=train_device, non_blocking=True)


def _setup_memory_strategy(
    train_module: torch.nn.Module,
    family: str,
    memory_block: dict[str, Any],
    config: dict[str, Any],
    *,
    checkpoint_path: str | None = None,
) -> MemoryStrategy | None:
    # Check for Stagehand strategy first.
    strategy_name = str(memory_block.get("strategy", config.get("memory_strategy", ""))).strip().lower()
    stagehand_enabled = _as_bool(config.get("stagehand_enabled", False), False)
    stagehand_block = memory_block.get("stagehand", {}) if isinstance(memory_block.get("stagehand"), dict) else {}

    if strategy_name == "stagehand" or stagehand_enabled or stagehand_block:
        gradient_checkpointing = str(
            memory_block.get("gradient_checkpointing") or config.get("gradient_checkpointing") or "off"
        )
        if gradient_checkpointing.lower() != "off" and hasattr(train_module, "enable_gradient_checkpointing"):
            with suppress(Exception):
                train_module.enable_gradient_checkpointing()
                print("[native/diffusion] enabled gradient checkpointing before Stagehand setup")

        stagehand_source_suffixes = (".safetensors", ".fpk", ".slab")
        stagehand_checkpoint_path = (
            stagehand_block.get("source_path")
            or stagehand_block.get("checkpoint_path")
            or config.get("stagehand_checkpoint_path")
            or checkpoint_path
            or config.get("model_path")
        )
        file_backed_default = family == "wan"
        file_backed_weights = _as_bool(
            stagehand_block.get(
                "file_backed_weights",
                config.get("stagehand_file_backed", file_backed_default),
            ),
            file_backed_default,
        )
        if stagehand_checkpoint_path and not str(stagehand_checkpoint_path).lower().endswith(stagehand_source_suffixes):
            file_backed_weights = False

        stagehand_config = StagehandStrategyConfig(
            family=family,
            block_pattern=stagehand_block.get("block_pattern"),
            pinned_pool_mb=int(stagehand_block.get("pinned_pool_mb", 8192)),
            pinned_slab_mb=int(stagehand_block.get("pinned_slab_mb", 512)),
            vram_high_watermark_mb=int(stagehand_block.get("vram_high_watermark_mb", 20000)),
            vram_low_watermark_mb=int(stagehand_block.get("vram_low_watermark_mb", 16000)),
            prefetch_window_blocks=int(stagehand_block.get("prefetch_window_blocks", 2)),
            max_inflight_transfers=int(stagehand_block.get("max_inflight_transfers", 2)),
            telemetry_enabled=_as_bool(stagehand_block.get("telemetry_enabled", True), True),
            telemetry_file=str(stagehand_block.get("telemetry_file", "stagehand_telemetry.jsonl")),
            gradient_checkpointing=gradient_checkpointing,
            dtype=str(config.get("train_dtype") or "bfloat16"),
            file_backed_weights=file_backed_weights,
            checkpoint_path=str(stagehand_checkpoint_path) if stagehand_checkpoint_path else None,
        )

        strategy = StagehandStrategy(stagehand_config)
        try:
            strategy.setup(SimpleNamespace(transformer=train_module))
            print(
                f"[native/diffusion] stagehand strategy active "
                f"(pool={stagehand_config.pinned_pool_mb}MB, "
                f"vram={stagehand_config.vram_low_watermark_mb}-{stagehand_config.vram_high_watermark_mb}MB)"
            )
            if stagehand_config.file_backed_weights and stagehand_config.checkpoint_path:
                converted = int(getattr(strategy, "file_backed_converted_params", 0))
                if converted > 0:
                    print(
                        "[native/diffusion] stagehand file-backed mode enabled "
                        f"(source={stagehand_config.checkpoint_path}, converted_params={converted})"
                    )
                else:
                    print(
                        "[native/diffusion] warning: stagehand file-backed requested "
                        f"but converted_params=0 (source={stagehand_config.checkpoint_path}); "
                        "continuing module-backed"
                    )
            return strategy
        except (ImportError, RuntimeError) as exc:
            print(f"[native/diffusion] warning: stagehand setup failed, falling back to layer offload ({exc})")

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


def _audit_gpu_memory(train_module: torch.nn.Module, pipeline: Any) -> None:
    """Diagnostic: walk model tree and report all CUDA-resident tensors."""
    if not torch.cuda.is_available():
        return

    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2
    print(f"\n[OOM-DIAG] === GPU Memory Audit ===")
    print(f"[OOM-DIAG] allocated={alloc_mb:.1f}MB  reserved={reserved_mb:.1f}MB  free={free_mb:.1f}MB")

    # Walk train_module (transformer) and group CUDA params by top-level child
    child_gpu: dict[str, float] = {}
    child_cpu: dict[str, float] = {}
    for name, child in train_module.named_children():
        gpu_bytes = 0
        cpu_bytes = 0
        for param in child.parameters():
            nbytes = param.numel() * param.element_size()
            if param.device.type == "cuda":
                gpu_bytes += nbytes
            else:
                cpu_bytes += nbytes
        for buf in child.buffers():
            nbytes = buf.numel() * buf.element_size()
            if buf.device.type == "cuda":
                gpu_bytes += nbytes
            else:
                cpu_bytes += nbytes
        if gpu_bytes > 0:
            child_gpu[name] = gpu_bytes / 1024**2
        if cpu_bytes > 0:
            child_cpu[name] = cpu_bytes / 1024**2

    if child_gpu:
        print(f"[OOM-DIAG] Transformer children ON GPU:")
        for name, mb in sorted(child_gpu.items(), key=lambda x: -x[1]):
            print(f"[OOM-DIAG]   {name}: {mb:.1f} MB")
        print(f"[OOM-DIAG]   TOTAL: {sum(child_gpu.values()):.1f} MB")
    else:
        print(f"[OOM-DIAG] No transformer children on GPU (good)")

    if child_cpu:
        total_cpu = sum(child_cpu.values())
        print(f"[OOM-DIAG] Transformer children ON CPU: {total_cpu:.1f} MB total")

    # Check pipeline components (VAE, text encoder)
    for comp_name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        comp = getattr(pipeline, comp_name, None)
        if comp is None:
            continue
        gpu_bytes = sum(
            p.numel() * p.element_size() for p in comp.parameters() if p.device.type == "cuda"
        )
        if gpu_bytes > 0:
            print(f"[OOM-DIAG] WARNING: pipeline.{comp_name} has {gpu_bytes / 1024**2:.1f} MB ON GPU!")

    # Check for mystery CUDA tensors (allocated but not in model tree)
    model_gpu_total = sum(child_gpu.values()) if child_gpu else 0.0
    pipeline_gpu = 0.0
    for comp_name in ("vae", "text_encoder", "text_encoder_2"):
        comp = getattr(pipeline, comp_name, None)
        if comp is not None:
            pipeline_gpu += sum(
                p.numel() * p.element_size() for p in comp.parameters() if p.device.type == "cuda"
            ) / 1024**2
    accounted = model_gpu_total + pipeline_gpu
    unaccounted = alloc_mb - accounted
    if unaccounted > 100:
        print(f"[OOM-DIAG] WARNING: {unaccounted:.1f} MB allocated but NOT in model/pipeline params!")
        print(f"[OOM-DIAG]   (could be: cached activations, optimizer state, CUDA context, unreleased tensors)")

    # Force empty cache and re-check
    torch.cuda.empty_cache()
    alloc_after = torch.cuda.memory_allocated() / 1024**2
    reserved_after = torch.cuda.memory_reserved() / 1024**2
    free_after = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2
    print(f"[OOM-DIAG] After empty_cache: alloc={alloc_after:.1f}MB  reserved={reserved_after:.1f}MB  free={free_after:.1f}MB")
    print(f"[OOM-DIAG] === End Audit ===\n")


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or "cuda error: out of memory" in text


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
        optimizer_state_device=train_device,
        restore_optimizer_state=True,
    )

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(
        checkpoint_block.get("save_every") or config.get("save_every") or config.get("save_every_n_steps") or 0
    )
    kl_weight = float(config.get("vae_kl_weight", 1e-6))

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}
    log_every_steps = max(1, int(config.get("log_every_steps", 1) or 1))
    loss_window = deque(maxlen=max(1, int(config.get("loss_avg_window", 20) or 20)))
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = max(start_step - 1, 0)
    last_grad_norm: float | None = None
    progress = _create_train_progress(
        config,
        start_step=start_step,
        max_steps=max_steps,
        default_desc="native_vae",
    )

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
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                last_grad_norm = _to_float(grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema_model is not None:
                ema_model.update()
            optimizer.zero_grad(set_to_none=True)

        if progress is not None:
            progress.update(1)

        loss_value = float(loss.detach().cpu())
        loss_window.append(loss_value)
        should_log = step == start_step or step % log_every_steps == 0 or step == max_steps
        if should_log:
            now = time.perf_counter()
            steps_since_log = max(step - last_log_step, 1)
            elapsed_since_log = max(now - last_log_time, 1e-6)
            steps_per_second = steps_since_log / elapsed_since_log
            sec_per_step = 1.0 / max(steps_per_second, 1e-6)
            steps_remaining = max(max_steps - step, 0)
            elapsed_total = max(now - train_start_time, 1e-6)
            steps_done = max(step - start_step + 1, 1)
            avg_steps_per_second = steps_done / elapsed_total
            eta = _format_eta(float(steps_remaining) / max(avg_steps_per_second, 1e-6))
            elapsed = _format_eta(elapsed_total)
            avg_loss = sum(loss_window) / max(len(loss_window), 1)
            current_lr = (
                float(optimizer.param_groups[0].get("lr", learning_rate))
                if getattr(optimizer, "param_groups", None)
                else float(learning_rate)
            )
            grad_norm_text = f"{last_grad_norm:.3f}" if last_grad_norm is not None else "n/a"
            vram_stats = _current_vram_mb(train_device)
            vram_text = (
                f"{vram_stats[0] / 1024.0:.2f}/{vram_stats[1] / 1024.0:.2f}G"
                if vram_stats is not None
                else "n/a"
            )
            if progress is not None:
                postfix = {
                    "loss": f"{loss_value:.6f}",
                    "avg_loss": f"{avg_loss:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "s/it": f"{sec_per_step:.2f}",
                    "eta": eta,
                    "vram": vram_text,
                    "gn": grad_norm_text,
                }
                progress.set_postfix(postfix, refresh=True)
            else:
                print(
                    f"[native/diffusion/vae] step {step}/{max_steps} "
                    f"loss={loss_value:.6f} avg_loss={avg_loss:.6f} "
                    f"lr={current_lr:.2e} speed={steps_per_second:.2f} step/s ({sec_per_step:.2f}s/step) "
                    f"elapsed={elapsed} eta={eta} vram={vram_text} grad_norm={grad_norm_text}"
                )
            last_log_time = now
            last_log_step = step

        checkpoint_path: Path | None = None
        if save_every > 0 and step % save_every == 0:
            checkpoint_path = output_dir / f"vae_step_{step:06d}.safetensors"
            saved_path = _save_module_state(vae, checkpoint_path, save_dtype=save_dtype)
            _progress_write(progress, f"[native/diffusion/vae] saved VAE checkpoint to {saved_path}")
            state_path = output_dir / f"state_step_{step:06d}.pt"
            _save_training_state(
                state_path,
                step=step,
                model_checkpoint=saved_path,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema_model=ema_model,
            )

    if progress is not None:
        progress.close()

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


def _run_wan_dual_stage_training(
    config: dict[str, Any],
    *,
    source_path: Path,
    steps_override: int | None = None,
) -> int:
    """Train WAN 2.2 dual-stage (high-noise + low-noise) with alternating model swaps.

    Only one 14B transformer is loaded at a time; they swap every N steps,
    reusing the same pinned memory pool.
    """
    import gc

    model_block = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    data_block = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    memory_block = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    checkpoint_block = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}
    optimizer_block = config.get("optimizer", {}) if isinstance(config.get("optimizer"), dict) else {}
    scheduler_block = config.get("scheduler", {}) if isinstance(config.get("scheduler"), dict) else {}

    dual_block = config.get("dual_stage", {}) if isinstance(config.get("dual_stage"), dict) else {}

    high_model_path = str(
        dual_block.get("high_model_path")
        or model_block.get("high_model_path")
        or config.get("high_model_path")
    )
    low_model_path = str(
        dual_block.get("low_model_path")
        or model_block.get("low_model_path")
        or config.get("low_model_path")
    )
    if not high_model_path or high_model_path == "None":
        raise ValueError("Missing dual_stage.high_model_path for WAN 2.2 dual-stage training")
    if not low_model_path or low_model_path == "None":
        raise ValueError("Missing dual_stage.low_model_path for WAN 2.2 dual-stage training")
    high_model_path = str(Path(high_model_path).expanduser())
    low_model_path = str(Path(low_model_path).expanduser())

    boundary_ratio = float(dual_block.get("boundary_ratio", 0.90))
    swap_every_steps = int(dual_block.get("swap_every_steps", 250))
    start_stage = str(dual_block.get("start_stage", "high")).strip().lower()
    if start_stage not in ("high", "low"):
        start_stage = "high"

    num_train_timesteps = 1000
    boundary_timestep = int(boundary_ratio * num_train_timesteps)

    stage_config = {
        "high": {"model_path": high_model_path, "t_min": boundary_timestep, "t_max": num_train_timesteps},
        "low": {"model_path": low_model_path, "t_min": 0, "t_max": boundary_timestep},
    }

    print(
        f"[native/diffusion/dual] WAN 2.2 dual-stage training\n"
        f"  high model: {high_model_path}\n"
        f"  low model:  {low_model_path}\n"
        f"  boundary:   t={boundary_timestep} (ratio={boundary_ratio})\n"
        f"  swap every: {swap_every_steps} steps\n"
        f"  start:      {start_stage}"
    )

    # --- Common training params ---
    train_dtype = _coerce_dtype(config.get("train_dtype") or model_block.get("dtype") or "bfloat16")
    save_dtype = _coerce_dtype(config.get("output_dtype"), default=train_dtype)
    train_device = torch.device(str(config.get("train_device", "cuda")))

    seed = int(config.get("seed", 42))
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
        else (config.get("max_steps") or config.get("max_train_steps") or 4000)
    )

    native_model = WanModel()
    raw_resolution = int(data_block.get("resolution") or config.get("resolution") or 384)
    resolution_multiple = int(getattr(native_model, "resolution_multiple", 16))
    resolution = _quantize_resolution(raw_resolution, resolution_multiple)

    output_dir = Path(
        checkpoint_block.get("output_dir") or config.get("output_dir") or (Path("output") / source_path.stem)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    high_output_dir = output_dir / "high"
    low_output_dir = output_dir / "low"
    high_output_dir.mkdir(parents=True, exist_ok=True)
    low_output_dir.mkdir(parents=True, exist_ok=True)

    adapter_type, adapter_block = _extract_adapter_config(config)
    rank = int(adapter_block.get("rank") or adapter_block.get("network_dim") or config.get("lora_rank") or 16)
    alpha = float(adapter_block.get("alpha") or adapter_block.get("network_alpha") or config.get("lora_alpha") or rank)

    # --- Build training pairs & cache data using first-stage model ---
    pairs = _build_training_pairs(config)
    wan_video_frame_count = _resolve_wan_video_frame_count(config, data_block)
    pairs.extend(_build_video_training_pairs(config))
    deduped: dict[Path, str] = dict(pairs)
    pairs = list(deduped.items())
    if not pairs:
        raise ValueError("No training media/captions found for dual-stage training.")
    pairs = _prepare_training_pairs(pairs, model_type=ModelType.WAN)
    if not pairs:
        raise ValueError("No training pairs remain after preprocessing.")

    print(f"[native/diffusion/dual] loading initial model for data caching ({start_stage} stage)")
    initial_model_path = stage_config[start_stage]["model_path"]
    pipeline = native_model.load_pipeline(
        initial_model_path,
        train_dtype,
        train_device=train_device,
        model_block=model_block,
    )

    print(f"[native/diffusion/dual] caching latents+text for {len(pairs)} samples")
    cached = _cache_training_data(
        native_model,
        pipeline,
        "wan",
        pairs,
        ModelType.WAN,
        None,
        resolution,
        train_device,
        train_dtype,
        allow_video=True,
        video_frame_count=wan_video_frame_count,
        cache_text_embeddings=True,
        keep_text_encoder_on_device=False,
    )
    if not cached:
        raise ValueError("No cached samples produced.")

    # Offload VAE and text encoder to CPU — shared across both stages.
    pipeline.vae.to("cpu")
    native_model.offload_text_encoders(pipeline)
    from serenity.memory.sync import torch_gc
    torch_gc()

    # --- Stagehand config (shared) ---
    stagehand_block = memory_block.get("stagehand", {}) if isinstance(memory_block.get("stagehand"), dict) else {}
    stagehand_file_backed = _as_bool(
        stagehand_block.get("file_backed_weights", config.get("stagehand_file_backed", True)),
        True,
    )
    stagehand_cfg = StagehandStrategyConfig(
        family="wan",
        block_pattern=stagehand_block.get("block_pattern"),
        pinned_pool_mb=int(stagehand_block.get("pinned_pool_mb", 8192)),
        pinned_slab_mb=int(stagehand_block.get("pinned_slab_mb", 512)),
        vram_high_watermark_mb=int(stagehand_block.get("vram_high_watermark_mb", 20000)),
        vram_low_watermark_mb=int(stagehand_block.get("vram_low_watermark_mb", 16000)),
        prefetch_window_blocks=int(stagehand_block.get("prefetch_window_blocks", 2)),
        max_inflight_transfers=int(stagehand_block.get("max_inflight_transfers", 2)),
        telemetry_enabled=_as_bool(stagehand_block.get("telemetry_enabled", True), True),
        telemetry_file=str(stagehand_block.get("telemetry_file", "stagehand_telemetry.jsonl")),
        gradient_checkpointing=str(
            memory_block.get("gradient_checkpointing") or config.get("gradient_checkpointing") or "on"
        ),
        dtype=str(config.get("train_dtype") or "bfloat16"),
        file_backed_weights=stagehand_file_backed,
        checkpoint_path=None,
    )
    stagehand_source_suffixes = (".safetensors", ".fpk", ".slab")

    def _resolve_stagehand_source_path(stage_name: str, model_path: str) -> str:
        """Resolve stage-specific file-backed source path.

        Falls back to the stage's model_path when no explicit source is provided.
        """
        direct = (
            stagehand_block.get(f"{stage_name}_source_path")
            or stagehand_block.get(f"{stage_name}_checkpoint_path")
        )
        if direct:
            return str(direct)

        source_paths = stagehand_block.get("source_paths")
        if isinstance(source_paths, dict):
            by_stage = source_paths.get(stage_name)
            if by_stage:
                return str(by_stage)

        checkpoint_paths = stagehand_block.get("checkpoint_paths")
        if isinstance(checkpoint_paths, dict):
            by_stage = checkpoint_paths.get(stage_name)
            if by_stage:
                return str(by_stage)

        shared = (
            stagehand_block.get("source_path")
            or stagehand_block.get("checkpoint_path")
            or config.get("stagehand_checkpoint_path")
        )
        if shared:
            return str(shared)
        return str(model_path)

    # --- Helper: set up one stage ---
    def _setup_stage(
        stage_name: str,
        model_path: str,
        existing_pool=None,
    ):
        """Load transformer, inject LoRA, set up Stagehand, create optimizer.

        Returns (train_module, adapter, strategy, optimizer, lr_scheduler, params).
        """
        from diffusers import WanTransformer3DModel
        print(f"[native/diffusion/dual] loading {stage_name}-noise transformer from {model_path}")
        transformer = WanTransformer3DModel.from_single_file(
            model_path, torch_dtype=train_dtype,
        )
        transformer.to("cpu")
        # Strip accelerate hooks
        try:
            from accelerate.hooks import remove_hook_from_module
            for module in transformer.modules():
                remove_hook_from_module(module)
        except ImportError:
            pass
        if hasattr(transformer, "hf_device_map"):
            del transformer.hf_device_map

        pipeline.transformer = transformer

        # Adapter training mode: freeze backbone parameters so Stagehand can
        # convert frozen base weights to file-backed sources.
        _set_module_train_state(transformer, False)

        # Enable gradient checkpointing
        if hasattr(transformer, "enable_gradient_checkpointing"):
            with suppress(Exception):
                transformer.enable_gradient_checkpointing()

        # Inject LoRA adapter
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
            model_type=ModelType.WAN.value,
            dropout=float(adapter_block.get("dropout", 0.0)),
            backend=adapter_backend,
            **_build_adapter_kwargs(adapter_block),
        )
        adapter.inject(transformer)

        # Set up Stagehand
        stage_source_path = _resolve_stagehand_source_path(stage_name, model_path)
        stage_file_backed = stagehand_cfg.file_backed_weights
        if stage_source_path and not str(stage_source_path).lower().endswith(stagehand_source_suffixes):
            stage_file_backed = False
        stage_cfg = replace(
            stagehand_cfg,
            file_backed_weights=stage_file_backed,
            checkpoint_path=stage_source_path,
        )
        strategy = StagehandStrategy(stage_cfg)
        if existing_pool is not None:
            strategy.setup_with_pool(SimpleNamespace(transformer=transformer), existing_pool)
        else:
            strategy.setup(SimpleNamespace(transformer=transformer))
        if stage_cfg.file_backed_weights and stage_cfg.checkpoint_path:
            converted = int(getattr(strategy, "file_backed_converted_params", 0))
            if converted > 0:
                print(
                    f"[native/diffusion/dual] stagehand file-backed enabled for {stage_name} "
                    f"(source={stage_cfg.checkpoint_path}, converted_params={converted})"
                )
            else:
                print(
                    f"[native/diffusion/dual] warning: file-backed requested for {stage_name} "
                    f"but converted_params=0 (source={stage_cfg.checkpoint_path}); "
                    "continuing module-backed"
                )

        # Move non-block submodules to GPU
        _move_non_block_submodules_to_device(transformer, train_device)

        # Create optimizer
        params = [p for p in adapter.get_trainable_params() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters found for {stage_name} stage adapter")

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
        optimizer.zero_grad(set_to_none=True)

        print(
            f"[native/diffusion/dual] {stage_name} stage ready "
            f"optimizer={optimizer_name} lr_scheduler={scheduler_name} "
            f"params={len(params)}"
        )
        return transformer, adapter, strategy, optimizer, lr_scheduler, params

    # --- Helper: tear down current stage, return pool ---
    def _teardown_stage(
        stage_name: str,
        transformer,
        adapter,
        strategy: StagehandStrategy,
        optimizer,
        stage_dir: Path,
        step: int,
    ):
        """Save LoRA + optimizer, shut down Stagehand (keep pool), free transformer."""
        # Save LoRA weights
        lora_path = stage_dir / f"{adapter_type}_step_{step:06d}.safetensors"
        adapter.save(str(lora_path))

        # Save optimizer state
        opt_path = stage_dir / f"optimizer_step_{step:06d}.pt"
        torch.save(optimizer.state_dict(), opt_path)

        print(f"[native/diffusion/dual] saved {stage_name} stage: lora={lora_path} optimizer={opt_path}")

        # Shutdown Stagehand but keep the pool
        pool = strategy.shutdown_keep_pool()

        # Move non-block submodules to CPU before deleting
        with suppress(Exception):
            _move_non_block_submodules_to_device(transformer, torch.device("cpu"))

        # Free transformer
        pipeline.transformer = None
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pool, lora_path, opt_path

    # --- Check for resume state ---
    dual_state_path = output_dir / "dual_state.pt"
    resume_state = _load_dual_stage_state(dual_state_path)
    start_step = 1
    active_stage = start_stage
    saved_lora_paths: dict[str, Path | None] = {"high": None, "low": None}
    saved_opt_paths: dict[str, Path | None] = {"high": None, "low": None}

    if resume_state is not None:
        start_step = int(resume_state.get("step", 0)) + 1
        active_stage = str(resume_state.get("active_stage", start_stage))
        for stage_key in ("high", "low"):
            lp = resume_state.get(f"{stage_key}_lora_path")
            if lp and Path(lp).exists():
                saved_lora_paths[stage_key] = Path(lp)
            op = resume_state.get(f"{stage_key}_optimizer_path")
            if op and Path(op).exists():
                saved_opt_paths[stage_key] = Path(op)
        print(
            f"[native/diffusion/dual] resuming from step {start_step}, "
            f"active_stage={active_stage}"
        )

    if start_step > max_steps:
        print(f"[native/diffusion/dual] resume step exceeds max_steps ({start_step}>{max_steps}); nothing to do.")
        return 0

    # --- Initialize active stage ---
    cur_stage = active_stage
    cur_cfg = stage_config[cur_stage]
    cur_dir = high_output_dir if cur_stage == "high" else low_output_dir

    transformer, adapter, strategy, optimizer, lr_scheduler, params = _setup_stage(
        cur_stage, cur_cfg["model_path"],
    )

    # Load saved LoRA weights if resuming
    if saved_lora_paths[cur_stage] is not None:
        print(f"[native/diffusion/dual] restoring {cur_stage} LoRA from {saved_lora_paths[cur_stage]}")
        adapter.load(str(saved_lora_paths[cur_stage]))
    if saved_opt_paths[cur_stage] is not None:
        print(f"[native/diffusion/dual] restoring {cur_stage} optimizer from {saved_opt_paths[cur_stage]}")
        try:
            opt_state = torch.load(str(saved_opt_paths[cur_stage]), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_state)
        except Exception as exc:
            print(f"[native/diffusion/dual] warning: failed to restore optimizer ({exc})")

    # --- Training loop ---
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(
        checkpoint_block.get("save_every") or config.get("save_every") or config.get("save_every_n_steps") or 0
    )
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}
    log_every_steps = max(1, int(config.get("log_every_steps", 1) or 1))
    loss_window = deque(maxlen=max(1, int(config.get("loss_avg_window", 20) or 20)))
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = max(start_step - 1, 0)
    last_grad_norm: float | None = None
    progress = _create_train_progress(
        config,
        start_step=start_step,
        max_steps=max_steps,
        default_desc="wan_dual_stage",
    )

    print(
        f"[native/diffusion/dual] training loop start step={start_step}/{max_steps} "
        f"active_stage={cur_stage} "
        f"timestep_range=[{cur_cfg['t_min']}, {cur_cfg['t_max']})"
    )

    for step in range(start_step, max_steps + 1):
        # --- Check if we need to swap stages ---
        steps_into_current = step - start_step
        if steps_into_current > 0 and steps_into_current % swap_every_steps == 0:
            next_stage = "low" if cur_stage == "high" else "high"
            next_cfg = stage_config[next_stage]
            next_dir = high_output_dir if next_stage == "high" else low_output_dir

            _progress_write(
                progress,
                f"[native/diffusion/dual] === SWAPPING {cur_stage} → {next_stage} at step {step} ===",
            )
            swap_t0 = time.perf_counter()

            # Tear down current stage
            pool, lora_path, opt_path = _teardown_stage(
                cur_stage, transformer, adapter, strategy, optimizer, cur_dir, step,
            )
            saved_lora_paths[cur_stage] = lora_path
            saved_opt_paths[cur_stage] = opt_path

            # Set up new stage with reused pool
            transformer, adapter, strategy, optimizer, lr_scheduler, params = _setup_stage(
                next_stage, next_cfg["model_path"], existing_pool=pool,
            )

            # Restore saved weights if this stage was trained before
            if saved_lora_paths[next_stage] is not None:
                _progress_write(progress, f"[native/diffusion/dual] restoring {next_stage} LoRA from {saved_lora_paths[next_stage]}")
                adapter.load(str(saved_lora_paths[next_stage]))
            if saved_opt_paths[next_stage] is not None:
                _progress_write(
                    progress,
                    f"[native/diffusion/dual] restoring {next_stage} optimizer from {saved_opt_paths[next_stage]}",
                )
                try:
                    opt_state = torch.load(str(saved_opt_paths[next_stage]), map_location="cpu", weights_only=False)
                    optimizer.load_state_dict(opt_state)
                except Exception as exc:
                    _progress_write(progress, f"[native/diffusion/dual] warning: failed to restore optimizer ({exc})")

            cur_stage = next_stage
            cur_cfg = next_cfg
            cur_dir = next_dir

            swap_elapsed = time.perf_counter() - swap_t0
            _progress_write(
                progress,
                f"[native/diffusion/dual] swap complete in {swap_elapsed:.1f}s, "
                f"now training {cur_stage} stage "
                f"timestep_range=[{cur_cfg['t_min']}, {cur_cfg['t_max']})"
            )

        # --- Forward/backward ---
        batch = _pick_batch(cached, batch_size, train_device, train_dtype, "wan")

        # Set timestep bounds for this stage
        config["_timestep_bounds"] = (cur_cfg["t_min"], cur_cfg["t_max"])

        forward_ctx = strategy.forward_context()
        with forward_ctx:
            with torch.autocast(device_type=train_device.type, dtype=train_dtype, enabled=autocast_enabled):
                loss = _compute_loss(
                    native_model,
                    pipeline,
                    "wan",
                    batch,
                    transformer,
                    train_dtype,
                    config,
                )

            scaled_loss = loss / float(max(grad_accum, 1))
            scaled_loss.backward()

        should_step = (step % max(grad_accum, 1) == 0) or (step == max_steps)
        if should_step:
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                last_grad_norm = _to_float(grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if progress is not None:
            progress.update(1)

        loss_value = float(loss.detach().cpu())
        loss_window.append(loss_value)
        # --- Logging ---
        should_log = step == start_step or step % log_every_steps == 0 or step == max_steps
        if should_log:
            now = time.perf_counter()
            steps_since_log = max(step - last_log_step, 1)
            elapsed_since_log = max(now - last_log_time, 1e-6)
            steps_per_second = steps_since_log / elapsed_since_log
            sec_per_step = 1.0 / max(steps_per_second, 1e-6)
            steps_remaining = max(max_steps - step, 0)
            elapsed_total = max(now - train_start_time, 1e-6)
            steps_done = max(step - start_step + 1, 1)
            avg_steps_per_second = steps_done / elapsed_total
            eta = _format_eta(float(steps_remaining) / max(avg_steps_per_second, 1e-6))
            elapsed = _format_eta(elapsed_total)
            avg_loss = sum(loss_window) / max(len(loss_window), 1)
            current_lr = (
                float(optimizer.param_groups[0].get("lr", learning_rate))
                if getattr(optimizer, "param_groups", None)
                else float(learning_rate)
            )
            grad_norm_text = f"{last_grad_norm:.3f}" if last_grad_norm is not None else "n/a"
            vram_stats = _current_vram_mb(train_device)
            vram_text = (
                f"{vram_stats[0] / 1024.0:.2f}/{vram_stats[1] / 1024.0:.2f}G"
                if vram_stats is not None
                else "n/a"
            )
            if progress is not None:
                postfix = {
                    "stage": cur_stage,
                    "loss": f"{loss_value:.6f}",
                    "avg_loss": f"{avg_loss:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "s/it": f"{sec_per_step:.2f}",
                    "eta": eta,
                    "vram": vram_text,
                    "gn": grad_norm_text,
                }
                progress.set_postfix(postfix, refresh=True)
            else:
                print(
                    f"[native/diffusion/dual] step {step}/{max_steps} "
                    f"stage={cur_stage} "
                    f"loss={loss_value:.6f} avg_loss={avg_loss:.6f} "
                    f"lr={current_lr:.2e} speed={steps_per_second:.2f} step/s ({sec_per_step:.2f}s/step) "
                    f"elapsed={elapsed} eta={eta} vram={vram_text} grad_norm={grad_norm_text}"
                )
            last_log_time = now
            last_log_step = step

        # --- Periodic save ---
        if save_every > 0 and step % save_every == 0:
            ckpt_path = cur_dir / f"{adapter_type}_step_{step:06d}.safetensors"
            adapter.save(str(ckpt_path))
            saved_lora_paths[cur_stage] = ckpt_path

            opt_path = cur_dir / f"optimizer_step_{step:06d}.pt"
            torch.save(optimizer.state_dict(), opt_path)
            saved_opt_paths[cur_stage] = opt_path

            _save_dual_stage_state(
                dual_state_path,
                step=step,
                active_stage=cur_stage,
                high_lora_path=saved_lora_paths["high"],
                low_lora_path=saved_lora_paths["low"],
                high_optimizer_path=saved_opt_paths["high"],
                low_optimizer_path=saved_opt_paths["low"],
            )
            _progress_write(progress, f"[native/diffusion/dual] checkpoint saved at step {step} (stage={cur_stage})")

    if progress is not None:
        progress.close()

    # --- Final save ---
    # Clean up timestep bounds
    config.pop("_timestep_bounds", None)

    # Save current (final) stage
    final_lora_path = cur_dir / f"{adapter_type}_last.safetensors"
    adapter.save(str(final_lora_path))
    saved_lora_paths[cur_stage] = final_lora_path
    print(f"[native/diffusion/dual] saved final {cur_stage} LoRA to {final_lora_path}")

    # Save dual state
    _save_dual_stage_state(
        dual_state_path,
        step=max_steps,
        active_stage=cur_stage,
        high_lora_path=saved_lora_paths["high"],
        low_lora_path=saved_lora_paths["low"],
        high_optimizer_path=saved_opt_paths["high"],
        low_optimizer_path=saved_opt_paths["low"],
    )

    # Cleanup Stagehand
    strategy.cleanup()

    print(
        f"[native/diffusion/dual] training complete\n"
        f"  high LoRA: {saved_lora_paths['high']}\n"
        f"  low LoRA:  {saved_lora_paths['low']}\n"
        f"  output:    {output_dir}"
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

    if family == "wan":
        wan_stage = config.get("wan_stage") or model_block.get("stage")
        if wan_stage is None:
            if "high" in normalized_model_type:
                wan_stage = "high"
            elif "low" in normalized_model_type:
                wan_stage = "low"
        if wan_stage:
            print(f"[native/diffusion] WAN 2.2 stage: {wan_stage}")

        # Dual-stage routing: if model_type is wan22_dual or dual_stage block is present
        dual_block = config.get("dual_stage", {}) if isinstance(config.get("dual_stage"), dict) else {}
        is_dual = normalized_model_type == "wan22_dual" or _as_bool(dual_block.get("enabled", False), False)
        if is_dual:
            return _run_wan_dual_stage_training(config, source_path=source_path, steps_override=steps_override)

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
    allow_video_dataset = family in {"ltx2", "wan"}
    video_frame_count = 1
    ltx_video_frame_count = 1
    if family == "wan" and allow_video_dataset:
        wan_video_frame_count = _resolve_wan_video_frame_count(config, data_block)
        requested_frames = int(
            data_block.get("frames")
            or data_block.get("video_frames")
            or data_block.get("num_frames")
            or config.get("frames")
            or config.get("video_frames")
            or config.get("num_frames")
            or wan_video_frame_count
        )
        if wan_video_frame_count != requested_frames:
            print(
                "[native/diffusion] adjusted WAN frame count "
                f"from {requested_frames} to {wan_video_frame_count} (must satisfy (frames-1) % 4 == 0)"
            )
        else:
            print(f"[native/diffusion] using WAN frame count {wan_video_frame_count}")

        pairs.extend(_build_video_training_pairs(config))
        deduped: dict[Path, str] = dict(pairs)
        pairs = list(deduped.items())
        video_frame_count = wan_video_frame_count
    elif family == "ltx2" and allow_video_dataset:
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
        video_frame_count = ltx_video_frame_count

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
        model_block=model_block,
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
        video_frame_count=video_frame_count,
        cache_text_embeddings=cache_text_embeddings,
        keep_text_encoder_on_device=text_encoder_training_active,
    )
    if not cached:
        raise ValueError("No cached samples were produced. Verify dataset paths and conditioning images.")

    # Offload VAE and text encoders to CPU after caching to free VRAM for training.
    pipeline.vae.to("cpu")
    if cache_text_embeddings and not text_encoder_training_active:
        native_model.offload_text_encoders(pipeline)
    from serenity.memory.sync import torch_gc
    torch_gc()
    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
        print(f"[native/diffusion] GPU after offload: alloc={alloc_mb:.0f}MB reserved={reserved_mb:.0f}MB")

    runtime_prompt_device = native_model.cache_prompt_device(pipeline, train_device)

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
        if torch.cuda.is_available():
            print(f"[native/diffusion] GPU after adapter.inject: alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB")

    memory_strategy = (
        None
        if dispatched_train_module
        else _setup_memory_strategy(
            train_module,
            family,
            memory_block,
            config,
            checkpoint_path=resolved_model_path,
        )
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

    if torch.cuda.is_available():
        print(f"[native/diffusion] GPU after memory_strategy setup: alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB")

    stagehand_active = isinstance(memory_strategy, StagehandStrategy)
    placed_on_device = False
    if stagehand_active:
        # Stagehand manages block placement via hooks.  Move only non-block
        # submodules (embeddings, norms, projections) to GPU.
        _move_non_block_submodules_to_device(train_module, train_device)
        placed_on_device = True
        if torch.cuda.is_available():
            print(f"[native/diffusion] GPU after non-block move: alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB")
        print(
            "[native/diffusion] stagehand active — non-block submodules moved to GPU, "
            "blocks managed by StagehandRuntime"
        )
    elif memory_strategy is not None and memory_strategy.conductor is not None:
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

    offload_active = bool(
        stagehand_active
        or (
            memory_strategy is not None
            and memory_strategy.conductor is not None
            and memory_strategy.conductor.offload_activated()
        )
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
    if torch.cuda.is_available():
        print(f"[native/diffusion] GPU after optimizer creation: alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB")

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
        optimizer_state_device=train_device,
        restore_optimizer_state=not offload_active,
    )
    if start_step > max_steps:
        print(
            f"[native/diffusion] resume state step exceeds max_steps ({start_step}>{max_steps}); nothing to do."
        )
        _finalize_distributed_context(distributed_context)
        return 0

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}
    log_every_steps = max(1, int(config.get("log_every_steps", 1) or 1))
    loss_window = deque(maxlen=max(1, int(config.get("loss_avg_window", 20) or 20)))
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = max(start_step - 1, 0)
    last_grad_norm: float | None = None
    progress = _create_train_progress(
        config,
        start_step=start_step,
        max_steps=max_steps,
        default_desc="native_diffusion",
    )
    sample_block = config.get("sample", {}) if isinstance(config.get("sample"), dict) else {}
    sample_enabled = _as_bool(sample_block.get("enabled"), False)
    sample_interval = int(sample_block.get("interval", 0) or 0)
    print(
        f"[native/diffusion] training loop start step={start_step}/{max_steps} "
        f"log_every={log_every_steps} save_every={save_every}"
    )

    if torch.cuda.is_available() and stagehand_active and _as_bool(config.get("debug_memory", False)):
        _audit_gpu_memory(train_module, pipeline)

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
                prompt_device=runtime_prompt_device,
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
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                last_grad_norm = _to_float(grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema_model is not None:
                ema_model.update()
            optimizer.zero_grad(set_to_none=True)

        if progress is not None:
            progress.update(1)

        loss_value = float(loss.detach().cpu())
        loss_window.append(loss_value)
        should_log = step == start_step or step % log_every_steps == 0 or step == max_steps
        if should_log:
            now = time.perf_counter()
            steps_since_log = max(step - last_log_step, 1)
            elapsed_since_log = max(now - last_log_time, 1e-6)
            steps_per_second = steps_since_log / elapsed_since_log
            sec_per_step = 1.0 / max(steps_per_second, 1e-6)
            steps_remaining = max(max_steps - step, 0)
            elapsed_total = max(now - train_start_time, 1e-6)
            steps_done = max(step - start_step + 1, 1)
            avg_steps_per_second = steps_done / elapsed_total
            eta = _format_eta(float(steps_remaining) / max(avg_steps_per_second, 1e-6))
            elapsed = _format_eta(elapsed_total)
            avg_loss = sum(loss_window) / max(len(loss_window), 1)
            current_lr = (
                float(optimizer.param_groups[0].get("lr", learning_rate))
                if getattr(optimizer, "param_groups", None)
                else float(learning_rate)
            )
            grad_norm_text = f"{last_grad_norm:.3f}" if last_grad_norm is not None else "n/a"
            vram_stats = _current_vram_mb(train_device)
            vram_text = (
                f"{vram_stats[0] / 1024.0:.2f}/{vram_stats[1] / 1024.0:.2f}G"
                if vram_stats is not None
                else "n/a"
            )
            if progress is not None:
                postfix = {
                    "loss": f"{loss_value:.6f}",
                    "avg_loss": f"{avg_loss:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "s/it": f"{sec_per_step:.2f}",
                    "eta": eta,
                    "vram": vram_text,
                    "gn": grad_norm_text,
                }
                progress.set_postfix(postfix, refresh=True)
            else:
                print(
                    f"[native/diffusion] step {step}/{max_steps} "
                    f"loss={loss_value:.6f} avg_loss={avg_loss:.6f} "
                    f"lr={current_lr:.2e} speed={steps_per_second:.2f} step/s ({sec_per_step:.2f}s/step) "
                    f"elapsed={elapsed} eta={eta} vram={vram_text} grad_norm={grad_norm_text}"
                )
            last_log_time = now
            last_log_step = step

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
                _progress_write(progress, f"[native/diffusion] saved full checkpoint to {saved_path}")
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

        should_sample_now = bool(sample_enabled and sample_interval > 0 and step % sample_interval == 0)
        if distributed_context.is_main_process:
            try:
                moved_for_sample = False
                conductor = memory_strategy.conductor if memory_strategy is not None else None
                if should_sample_now and train_device.type == "cuda":
                    # Free as much CUDA memory as possible before building a sample pipeline.
                    with suppress(Exception):
                        del batch
                    with suppress(Exception):
                        del loss
                    with suppress(Exception):
                        del scaled_loss

                    if offload_active and conductor is not None:
                        cpu_device = torch.device(str(config.get("temp_device", "cpu")))
                        with suppress(Exception):
                            conductor.to(cpu_device)
                            _move_non_offloaded_tensors_to_device(
                                train_module,
                                conductor=conductor,
                                train_device=cpu_device,
                            )
                            moved_for_sample = True
                    elif not dispatched_train_module:
                        with suppress(Exception):
                            train_module.to("cpu")
                            moved_for_sample = True

                    with suppress(Exception):
                        torch.cuda.synchronize()
                    with suppress(Exception):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

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
                    default_video_frames=video_frame_count if allow_video_dataset else None,
                )
                if should_sample_now and moved_for_sample:
                    if offload_active and conductor is not None:
                        cpu_device = torch.device(str(config.get("temp_device", "cpu")))
                        with suppress(Exception):
                            conductor.to(cpu_device)
                            _move_non_offloaded_tensors_to_device(
                                train_module,
                                conductor=conductor,
                                train_device=train_device,
                            )
                    elif not dispatched_train_module:
                        with suppress(Exception):
                            train_module.to(train_device)
                    with suppress(Exception):
                        torch.cuda.empty_cache()
            except Exception as exc:
                retried = False
                retry_error: Exception | None = None
                if train_device.type == "cuda" and _is_cuda_oom_error(exc):
                    retried = True
                    print(
                        f"[native/diffusion] warning: sampling OOM at step {step}; "
                        "retrying after releasing trainer CUDA memory"
                    )
                    moved_to_cpu = False
                    conductor = memory_strategy.conductor if memory_strategy is not None else None
                    try:
                        if offload_active and conductor is not None:
                            cpu_device = torch.device(str(config.get("temp_device", "cpu")))
                            with suppress(Exception):
                                conductor.to(cpu_device)
                                _move_non_offloaded_tensors_to_device(
                                    train_module,
                                    conductor=conductor,
                                    train_device=cpu_device,
                                )
                                moved_to_cpu = True
                        elif not dispatched_train_module:
                            with suppress(Exception):
                                train_module.to("cpu")
                                moved_to_cpu = True

                        with suppress(Exception):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()

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
                            default_video_frames=video_frame_count if allow_video_dataset else None,
                        )
                        print(f"[native/diffusion] sampling retry succeeded at step {step}")
                    except Exception as retry_exc:
                        retry_error = retry_exc
                    finally:
                        if moved_to_cpu:
                            if offload_active and conductor is not None:
                                cpu_device = torch.device(str(config.get("temp_device", "cpu")))
                                with suppress(Exception):
                                    conductor.to(cpu_device)
                                    _move_non_offloaded_tensors_to_device(
                                        train_module,
                                        conductor=conductor,
                                        train_device=train_device,
                                    )
                            elif not dispatched_train_module:
                                with suppress(Exception):
                                    train_module.to(train_device)
                        with suppress(Exception):
                            torch.cuda.empty_cache()

                if retry_error is not None:
                    _progress_write(progress, f"[native/diffusion] warning: sampling retry failed at step {step}: {retry_error}")
                elif not retried:
                    _progress_write(progress, f"[native/diffusion] warning: sampling failed at step {step}: {exc}")

    if progress is not None:
        progress.close()

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
