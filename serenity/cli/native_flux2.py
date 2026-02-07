"""Native FLUX.2/Klein training path for Serenity.

Native Flux 2 training pipeline without bridge mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serenity.adapters import create_adapter
from serenity.core.interfaces import ModelType
from serenity.models.flux2_klein import Flux2KleinModelLoader
from serenity.sampling.sampler import create_sampler
from serenity.training.flux2.edit_trainer import Flux2EditTrainer, Flux2EditTrainerConfig
from serenity.training.flux2.image_trainer import Flux2ImageTrainer, Flux2ImageTrainerConfig

import torch

import numpy as np
from PIL import Image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_COND_LABEL_SUFFIXES = ("-condlabel", "_condlabel")
_FULL_TRAINING_METHODS = {"full", "full_finetune", "full_fine_tune", "fine_tune", "finetune"}
_UNSUPPORTED_NATIVE_FLUX2_TRAINING_METHODS = {"embedding", "fine_tune_vae", "finetune_vae"}
_ADAPTER_ALIASES = {
    "diag-oft": "diag_oft",
    "diagoft": "diag_oft",
    "oft_2": "oft",
    "full_finetune": "full",
    "full_fine_tune": "full",
    "fine_tune": "full",
    "finetune": "full",
}
_KNOWN_ADAPTER_KEYS = (
    "lora",
    "lokr",
    "loha",
    "locon",
    "dora",
    "ia3",
    "oft",
    "boft",
    "diag_oft",
    "glora",
    "dylora",
    "full",
)

_CONSTANT_SCHEDULERS = {"", "none", "off", "constant", "constant_with_warmup", "adafactor"}
_LINEAR_SCHEDULERS = {"linear"}
_COSINE_SCHEDULERS = {"cosine"}
_COSINE_RESTART_SCHEDULERS = {"cosine_with_restarts", "cosine_with_hard_restarts", "cosine_restarts"}
_SCHEDULER_ALIASES = {
    "": "constant",
    "none": "constant",
    "off": "constant",
    "constant_with_warmup": "constant",
    "cosine_with_hard_restart": "cosine_with_hard_restarts",
    "cosine_restart": "cosine_with_restarts",
    "learning_rate_scheduler": "constant",
}
_OPTIMIZER_ALIASES = {
    "adamw": "adamw",
    "adamw_8bit": "adamw",
    "adamw8bit": "adamw",
    "paged_adamw_8bit": "adamw",
    "paged_adamw8bit": "adamw",
    "schedule_free_adamw": "adamw",
    "schedulefree_adamw": "adamw",
    "adam": "adam",
    "adam_8bit": "adam",
    "adam8bit": "adam",
    "paged_adam_8bit": "adam",
    "paged_adam8bit": "adam",
    "sgd": "sgd",
    "adafactor": "adafactor",
    "lion": "lion",
}


@dataclass
class _CachedSample:
    latents: torch.Tensor
    text_embeddings: torch.Tensor


@dataclass
class _CachedEditSample:
    target_latents: torch.Tensor
    reference_latents: torch.Tensor
    text_embeddings: torch.Tensor


def _resolve_hf_local_path(model_path: str) -> str:
    expanded = Path(model_path).expanduser()
    if expanded.exists():
        return str(expanded)

    if "/" not in model_path:
        raise FileNotFoundError(
            f"Model path not found locally: {model_path}. "
            "Expected a local path or a cached HF repo id."
        )

    org, name = model_path.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{org}--{name}"
    if not repo_dir.exists():
        raise FileNotFoundError(f"HF cache not found for {model_path}: {repo_dir}")

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

    raise FileNotFoundError(f"No HF snapshots found in cache for {model_path}")


def _coerce_dtype(value: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default

    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    return default


def _normalize_model_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", "null"}:
        return False
    return default


def _first_config_value(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in primary and primary.get(key) is not None:
            return primary.get(key)
        if key in secondary and secondary.get(key) is not None:
            return secondary.get(key)
    return default


def _normalize_optimizer_name(value: Any, default: str = "adamw") -> str:
    normalized = _normalize_model_type(value or default)
    normalized = _OPTIMIZER_ALIASES.get(normalized, normalized)

    if normalized in _OPTIMIZER_ALIASES:
        return _OPTIMIZER_ALIASES[normalized]
    if "adafactor" in normalized:
        return "adafactor"
    if "lion" in normalized:
        return "lion"
    if normalized.startswith("sgd"):
        return "sgd"
    if normalized.startswith("adamw"):
        return "adamw"
    if normalized.startswith("adam"):
        return "adam"
    return default


def _normalize_scheduler_name(value: Any, default: str = "constant") -> str:
    normalized = _normalize_model_type(value or default)
    normalized = _SCHEDULER_ALIASES.get(normalized, normalized)
    if normalized in _CONSTANT_SCHEDULERS | _LINEAR_SCHEDULERS | _COSINE_SCHEDULERS | _COSINE_RESTART_SCHEDULERS:
        return normalized
    return default


def _resolve_optimizer_steps(max_steps: int, grad_accum: int) -> int:
    return max(1, math.ceil(float(max_steps) / float(max(1, grad_accum))))


def _resolve_warmup_steps(
    config: dict[str, Any],
    scheduler_block: dict[str, Any],
    total_optimizer_steps: int,
) -> int:
    warmup_raw = _first_config_value(
        scheduler_block,
        config,
        ("warmup_steps", "lr_warmup_steps", "learning_rate_warmup_steps"),
        0,
    )
    warmup_steps = int(float(warmup_raw or 0))
    return max(0, min(warmup_steps, total_optimizer_steps))


def _resolve_scheduler_min_factor(config: dict[str, Any], scheduler_block: dict[str, Any]) -> float:
    min_factor_raw = _first_config_value(
        scheduler_block,
        config,
        ("min_factor", "min_lr_factor", "lr_min_factor", "eta_min_ratio", "min_lr_ratio"),
        0.0,
    )
    min_factor = float(min_factor_raw or 0.0)
    return max(0.0, min(min_factor, 1.0))


def _resolve_scheduler_cycles(config: dict[str, Any], scheduler_block: dict[str, Any]) -> float:
    num_cycles_raw = _first_config_value(
        scheduler_block,
        config,
        ("num_cycles", "lr_num_cycles", "cosine_num_cycles"),
        1.0,
    )
    num_cycles = float(num_cycles_raw or 1.0)
    return max(1.0, num_cycles)


def _create_optimizer(
    params: list[torch.nn.Parameter],
    *,
    config: dict[str, Any],
    optimizer_block: dict[str, Any],
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name_raw = _first_config_value(
        optimizer_block,
        config,
        ("optimizer", "optimizer_type"),
        "adamw",
    )
    optimizer_name = _normalize_optimizer_name(optimizer_name_raw)

    weight_decay = float(_first_config_value(optimizer_block, config, ("weight_decay",), 0.0))
    beta1 = float(_first_config_value(optimizer_block, config, ("beta1",), 0.9))
    beta2 = float(_first_config_value(optimizer_block, config, ("beta2",), 0.999))
    eps = float(_first_config_value(optimizer_block, config, ("eps", "epsilon"), 1e-8))
    amsgrad = _as_bool(_first_config_value(optimizer_block, config, ("amsgrad",), False))

    if optimizer_name == "adafactor":
        from transformers import Adafactor

        clip_threshold_raw = _first_config_value(optimizer_block, config, ("clip_threshold",), None)
        adafactor_kwargs: dict[str, Any] = {
            "lr": learning_rate,
            "scale_parameter": _as_bool(_first_config_value(optimizer_block, config, ("scale_parameter",), False)),
            "relative_step": _as_bool(_first_config_value(optimizer_block, config, ("relative_step",), False)),
            "warmup_init": _as_bool(_first_config_value(optimizer_block, config, ("warmup_init",), False)),
            "weight_decay": weight_decay,
            "eps": (eps, 1e-3),
        }
        if clip_threshold_raw is not None:
            adafactor_kwargs["clip_threshold"] = float(clip_threshold_raw)
        return Adafactor(params, **adafactor_kwargs), optimizer_name

    if optimizer_name == "adam":
        return (
            torch.optim.Adam(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            ),
            optimizer_name,
        )

    if optimizer_name == "sgd":
        momentum = float(_first_config_value(optimizer_block, config, ("momentum",), 0.0))
        dampening = float(_first_config_value(optimizer_block, config, ("dampening",), 0.0))
        nesterov = _as_bool(_first_config_value(optimizer_block, config, ("nesterov",), False))
        if momentum <= 0.0:
            nesterov = False
        return (
            torch.optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
            ),
            optimizer_name,
        )

    if optimizer_name == "lion":
        lion_class = getattr(torch.optim, "Lion", None)
        if lion_class is None:
            try:
                from lion_pytorch import Lion as lion_class
            except Exception:
                print("[native/flux2] warning: Lion optimizer unavailable, falling back to AdamW.")
                optimizer_name = "adamw"
                lion_class = None
        if lion_class is not None:
            return (
                lion_class(
                    params,
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                ),
                optimizer_name,
            )

    return (
        torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        ),
        "adamw",
    )


def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, Any],
    scheduler_block: dict[str, Any],
    total_optimizer_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, str]:
    scheduler_name_raw = _first_config_value(
        scheduler_block,
        config,
        ("scheduler", "lr_scheduler", "learning_rate_scheduler"),
        "constant",
    )
    scheduler_name = _normalize_scheduler_name(scheduler_name_raw)
    warmup_steps = _resolve_warmup_steps(config, scheduler_block, total_optimizer_steps)
    min_factor = _resolve_scheduler_min_factor(config, scheduler_block)
    num_cycles = _resolve_scheduler_cycles(config, scheduler_block)

    use_constant_schedule = scheduler_name in _CONSTANT_SCHEDULERS
    if use_constant_schedule and warmup_steps <= 0 and min_factor <= 0.0:
        return None, scheduler_name

    def _lr_lambda(last_epoch: int) -> float:
        step = max(0, int(last_epoch) + 1)

        if warmup_steps > 0 and step <= warmup_steps:
            warmup_progress = float(step) / float(max(1, warmup_steps))
            return max(min_factor, min(1.0, warmup_progress))

        if total_optimizer_steps <= warmup_steps:
            progress = 1.0
        else:
            progress = (float(step) - float(warmup_steps)) / float(max(1, total_optimizer_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_name in _LINEAR_SCHEDULERS:
            base_factor = 1.0 - progress
        elif scheduler_name in _COSINE_SCHEDULERS:
            base_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler_name in _COSINE_RESTART_SCHEDULERS:
            if progress >= 1.0:
                base_factor = 0.0
            else:
                cycle_position = (num_cycles * progress) % 1.0
                base_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_position))
        else:
            base_factor = 1.0

        factor = min_factor + (1.0 - min_factor) * base_factor
        return min(max(float(factor), min_factor), 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return scheduler, scheduler_name


def _collect_sample_prompts(sample_block: dict[str, Any]) -> list[str]:
    prompt_values = sample_block.get("prompts")
    if prompt_values is None:
        prompt_values = sample_block.get("prompt")

    if prompt_values is None:
        return []
    if isinstance(prompt_values, str):
        text = prompt_values.strip()
        return [text] if text else []

    if isinstance(prompt_values, dict):
        prompt_text = str(prompt_values.get("prompt") or prompt_values.get("text") or "").strip()
        return [prompt_text] if prompt_text else []

    prompts: list[str] = []
    if isinstance(prompt_values, list | tuple | set):
        for item in prompt_values:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    prompts.append(text)
                continue
            if isinstance(item, dict):
                text = str(item.get("prompt") or item.get("text") or "").strip()
                if text:
                    prompts.append(text)
    return prompts


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _collect_sample_seeds(sample_block: dict[str, Any], default_seed: int) -> list[int]:
    seed_values = sample_block.get("seeds")
    if seed_values is None:
        seed_values = sample_block.get("seed")

    if seed_values is None:
        return [default_seed]

    if isinstance(seed_values, list | tuple | set):
        seeds: list[int] = []
        for value in seed_values:
            candidate = _coerce_int_or_none(value)
            if candidate is not None:
                seeds.append(candidate)
        return seeds or [default_seed]

    candidate = _coerce_int_or_none(seed_values)
    return [candidate] if candidate is not None else [default_seed]


def _resolve_sample_output_extension(sample_block: dict[str, Any], default: str) -> str:
    ext = str(sample_block.get("output_ext") or sample_block.get("file_ext") or default).strip().lower()
    if not ext:
        ext = default
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _optional_int(value: Any) -> int | None:
    return _coerce_int_or_none(value)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_adapter_type(value: Any, default: str = "lora") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _ADAPTER_ALIASES.get(normalized, normalized)
    return normalized or default


def _resolve_flux2_model_type(value: str, model_path: str) -> ModelType:
    normalized = _normalize_model_type(value)
    path_lower = str(model_path).lower()

    if normalized in {"flux_2_klein_4b", "flux2_klein_4b"}:
        return ModelType.FLUX_2_KLEIN_4B
    if normalized in {"flux_2_klein_9b", "flux2_klein_9b"}:
        return ModelType.FLUX_2_KLEIN_9B
    if normalized in {"flux_2_klein_4b_base", "flux2_klein_4b_base"}:
        return ModelType.FLUX_2_KLEIN_4B_BASE
    if normalized in {"flux_2_klein_9b_base", "flux2_klein_9b_base"}:
        return ModelType.FLUX_2_KLEIN_9B_BASE

    # Heuristic fallback for generic flux2/flux values.
    is_4b = "4b" in path_lower
    is_base = "base" in path_lower
    if is_4b and is_base:
        return ModelType.FLUX_2_KLEIN_4B_BASE
    if is_4b:
        return ModelType.FLUX_2_KLEIN_4B
    if is_base:
        return ModelType.FLUX_2_KLEIN_9B_BASE
    return ModelType.FLUX_2_KLEIN_9B


def _extract_adapter_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    training_method = _normalize_model_type(config.get("training_method"))
    if training_method in _FULL_TRAINING_METHODS:
        return "full", {}

    if "adapter" in config and isinstance(config["adapter"], dict):
        adapter_block = config["adapter"]
        return _normalize_adapter_type(adapter_block.get("type", config.get("peft_type", "lora"))), adapter_block

    peft_type = _normalize_adapter_type(config.get("peft_type", "lora"))
    if peft_type in _FULL_TRAINING_METHODS:
        return "full", {}
    if peft_type in config and isinstance(config[peft_type], dict):
        return peft_type, config[peft_type]

    for key in _KNOWN_ADAPTER_KEYS:
        if key in config and isinstance(config[key], dict):
            return _normalize_adapter_type(key), config[key]

    lycoris_block = config.get("lycoris")
    if isinstance(lycoris_block, dict):
        lycoris_type = lycoris_block.get("type") or lycoris_block.get("algorithm") or peft_type
        return _normalize_adapter_type(lycoris_type), lycoris_block

    return peft_type, {}


def _as_target_modules(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple | set):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _build_adapter_kwargs(adapter_block: dict[str, Any]) -> dict[str, Any]:
    conv_rank_raw = adapter_block.get("conv_rank")
    conv_alpha_raw = adapter_block.get("conv_alpha")

    return {
        "target_modules": _as_target_modules(adapter_block.get("target_modules")),
        "conv_rank": int(conv_rank_raw) if conv_rank_raw is not None else None,
        "conv_alpha": float(conv_alpha_raw) if conv_alpha_raw is not None else None,
        "rank_dropout": float(adapter_block.get("rank_dropout", 0.0)),
        "module_dropout": float(adapter_block.get("module_dropout", 0.0)),
        "factor": int(adapter_block.get("factor", 2)),
        "decompose_both": _as_bool(adapter_block.get("decompose_both", False)),
        "use_tucker": _as_bool(adapter_block.get("use_tucker", False)),
        "full_matrix": _as_bool(adapter_block.get("full_matrix", False)),
        "weight_decompose": _as_bool(
            adapter_block.get("weight_decompose", adapter_block.get("dora_wd", False))
        ),
        "dora_on_output": _as_bool(adapter_block.get("dora_on_output", adapter_block.get("wd_on_output", True))),
        "rs_lora": _as_bool(adapter_block.get("rs_lora", False)),
        "block_size": int(adapter_block.get("block_size", adapter_block.get("boft_block_size", 4))),
        "constraint": float(adapter_block.get("constraint", 0.0)),
        "rescaled": _as_bool(adapter_block.get("rescaled", False)),
        "multiplier": float(adapter_block.get("multiplier", 1.0)),
    }


def _collect_concept_dirs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    data_block = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    concepts = data_block.get("concepts") or config.get("concepts") or []

    out: list[tuple[Path, str]] = []
    for concept in concepts:
        if isinstance(concept, str):
            out.append((Path(concept).expanduser(), ".txt"))
            continue

        if not isinstance(concept, dict):
            continue

        concept_path = concept.get("path")
        if not concept_path:
            continue

        caption_ext = str(concept.get("caption_file_ext") or ".txt")
        if not caption_ext.startswith("."):
            caption_ext = f".{caption_ext}"

        out.append((Path(concept_path).expanduser(), caption_ext))

    return out


def _load_caption(image_path: Path, caption_ext: str) -> str:
    caption_path = image_path.with_suffix(caption_ext)
    if caption_path.exists():
        text = caption_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return image_path.stem.replace("_", " ").strip()


def _is_condlabel_image(path: Path) -> bool:
    stem = path.stem.lower()
    return any(stem.endswith(suffix) for suffix in _COND_LABEL_SUFFIXES)


def _strip_condlabel_suffix(stem: str) -> str:
    lowered = stem.lower()
    for suffix in _COND_LABEL_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_reference_image_path(image_path: Path) -> Path | None:
    base_stem = _strip_condlabel_suffix(image_path.stem)
    parent = image_path.parent
    for suffix in _COND_LABEL_SUFFIXES:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            candidate = parent / f"{base_stem}{suffix}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _load_image_tensor(image_path: Path, resolution: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=dtype)


def _build_training_pairs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for concept_dir, caption_ext in _collect_concept_dirs(config):
        if not concept_dir.exists():
            continue
        for image_path in sorted(concept_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            if _is_condlabel_image(image_path):
                continue
            caption = _load_caption(image_path, caption_ext)
            if caption:
                pairs.append((image_path, caption))
    return pairs


def _build_edit_training_pairs(config: dict[str, Any]) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for concept_dir, caption_ext in _collect_concept_dirs(config):
        if not concept_dir.exists():
            continue
        for image_path in sorted(concept_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            if _is_condlabel_image(image_path):
                continue

            reference_path = _resolve_reference_image_path(image_path)
            if reference_path is None:
                continue

            caption = _load_caption(image_path, caption_ext)
            if caption:
                pairs.append((image_path, reference_path, caption))
    return pairs


def _cache_training_data(
    model,
    pairs: list[tuple[Path, str]],
    resolution: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
) -> list[_CachedSample]:
    model.vae.to(train_device)
    model.text_encoder.to(train_device)

    cached: list[_CachedSample] = []
    with torch.no_grad():
        for image_path, caption in pairs:
            pixel_values = _load_image_tensor(image_path, resolution, train_dtype, train_device).unsqueeze(0)
            latent_32 = model.encode_image(pixel_values)
            latents = model.normalize_latents(model.patchify_latents(latent_32)).squeeze(0).cpu()

            text_emb = model.encode_prompt(
                caption,
                device=train_device,
                max_sequence_length=model.max_sequence_length,
            ).squeeze(0).cpu()

            cached.append(_CachedSample(latents=latents, text_embeddings=text_emb))

    model.vae.to("cpu")
    model.text_encoder.to("cpu")
    return cached


def _cache_edit_training_data(
    model,
    pairs: list[tuple[Path, Path, str]],
    resolution: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
) -> list[_CachedEditSample]:
    model.vae.to(train_device)
    model.text_encoder.to(train_device)

    cached: list[_CachedEditSample] = []
    with torch.no_grad():
        for target_path, reference_path, caption in pairs:
            target_pixels = _load_image_tensor(target_path, resolution, train_dtype, train_device).unsqueeze(0)
            reference_pixels = _load_image_tensor(reference_path, resolution, train_dtype, train_device).unsqueeze(0)

            target_latent_32 = model.encode_image(target_pixels)
            target_latents = model.normalize_latents(model.patchify_latents(target_latent_32)).squeeze(0).cpu()

            reference_latent_32 = model.encode_image(reference_pixels)
            reference_latents = model.normalize_latents(model.patchify_latents(reference_latent_32)).squeeze(0).cpu()

            text_emb = model.encode_prompt(
                caption,
                device=train_device,
                max_sequence_length=model.max_sequence_length,
            ).squeeze(0).cpu()

            cached.append(
                _CachedEditSample(
                    target_latents=target_latents,
                    reference_latents=reference_latents,
                    text_embeddings=text_emb,
                )
            )

    model.vae.to("cpu")
    model.text_encoder.to("cpu")
    return cached


def _save_transformer_state(
    transformer: torch.nn.Module,
    output_path: Path,
    *,
    save_dtype: torch.dtype | None = None,
) -> Path:
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in transformer.state_dict().items():
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
    except Exception:
        fallback_path = output_path.with_suffix(".pt")
        torch.save(state_dict, fallback_path)
        return fallback_path


def _pick_batch(
    cached: list[_CachedSample],
    batch_size: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    items = random.choices(cached, k=max(1, batch_size))
    latents = torch.stack([item.latents for item in items], dim=0).to(train_device, dtype=train_dtype)
    text_embeddings = torch.stack([item.text_embeddings for item in items], dim=0).to(train_device, dtype=train_dtype)
    return {
        "latents": latents,
        "text_embeddings": text_embeddings,
    }


def _pick_edit_batch(
    cached: list[_CachedEditSample],
    batch_size: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    items = random.choices(cached, k=max(1, batch_size))
    target_latents = torch.stack([item.target_latents for item in items], dim=0).to(train_device, dtype=train_dtype)
    reference_latents = (
        torch.stack([item.reference_latents for item in items], dim=0)
        .unsqueeze(1)
        .to(train_device, dtype=train_dtype)
    )
    text_embeddings = torch.stack([item.text_embeddings for item in items], dim=0).to(train_device, dtype=train_dtype)
    return {
        "target_latents": target_latents,
        "reference_latents": reference_latents,
        "text_embeddings": text_embeddings,
    }


def _maybe_sample(
    config: dict[str, Any],
    *,
    model_type: ModelType,
    model_path: str,
    output_dir: Path,
    step: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
    default_resolution: int | None = None,
) -> None:
    sample_block = config.get("sample", {}) if isinstance(config.get("sample"), dict) else {}
    if not sample_block.get("enabled", False):
        return

    interval = int(sample_block.get("interval", 0) or 0)
    if interval <= 0 or step % interval != 0:
        return

    prompts = _collect_sample_prompts(sample_block)
    if not prompts:
        return

    negative_prompt = str(sample_block.get("negative_prompt", ""))
    seeds = _collect_sample_seeds(sample_block, int(config.get("seed", 42)))
    max_samples = int(sample_block.get("max_samples") or 0)
    prompts_to_run = prompts if max_samples <= 0 else prompts[:max_samples]
    output_ext = _resolve_sample_output_extension(sample_block, ".png")

    sample_height = _optional_int(sample_block.get("height") or sample_block.get("sample_height")) or default_resolution
    sample_width = _optional_int(sample_block.get("width") or sample_block.get("sample_width")) or default_resolution
    sample_steps = _optional_int(
        sample_block.get("num_inference_steps") or sample_block.get("sample_steps") or sample_block.get("steps")
    )
    sample_guidance = _optional_float(sample_block.get("guidance_scale") or sample_block.get("cfg_scale"))

    sampler = create_sampler(model_type, model={"path": model_path})
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for prompt_index, prompt in enumerate(prompts_to_run):
        seed = seeds[prompt_index % len(seeds)]
        out_path = samples_dir / f"step_{step:06d}_p{prompt_index:02d}_s{seed}{output_ext}"
        sample_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_path": model_path,
            "seed": seed,
            "device": train_device,
            "dtype": train_dtype,
            "output_path": out_path,
            "unload": prompt_index == len(prompts_to_run) - 1,
        }
        if sample_height is not None:
            sample_kwargs["height"] = sample_height
        if sample_width is not None:
            sample_kwargs["width"] = sample_width
        if sample_steps is not None:
            sample_kwargs["num_inference_steps"] = sample_steps
        if sample_guidance is not None:
            sample_kwargs["guidance_scale"] = sample_guidance

        sampler.sample(
            **sample_kwargs,
        )


def run_native_flux2_training(
    config: dict[str, Any],
    *,
    source_path: Path,
    steps_override: int | None = None,
) -> int:
    training_method = _normalize_model_type(config.get("training_method") or "lora")
    if training_method in _UNSUPPORTED_NATIVE_FLUX2_TRAINING_METHODS:
        raise ValueError(
            f"Unsupported native FLUX.2 training_method '{training_method}'. "
            "Use lora/fine_tune for FLUX.2 native training."
        )

    model_block = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    memory_block = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    checkpoint_block = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}
    optimizer_block = config.get("optimizer", {}) if isinstance(config.get("optimizer"), dict) else {}
    scheduler_block = config.get("scheduler", {}) if isinstance(config.get("scheduler"), dict) else {}
    adapter_type, adapter_block = _extract_adapter_config(config)
    full_finetune = adapter_type == "full"

    model_path_raw = (
        model_block.get("path")
        or config.get("base_model")
        or config.get("base_model_name")
        or config.get("transformer_path")
    )
    if not model_path_raw:
        raise ValueError("Missing model path. Expected `model.path` or `base_model`.")

    resolved_model_path = _resolve_hf_local_path(str(model_path_raw))

    normalized_model_type = _normalize_model_type(config.get("model_type"))
    flux_model_type = _resolve_flux2_model_type(normalized_model_type, resolved_model_path)
    edit_mode = (
        normalized_model_type.endswith("_edit")
        or _as_bool(config.get("edit_mode"), False)
        or _as_bool(config.get("data", {}).get("edit_mode") if isinstance(config.get("data"), dict) else None, False)
    )

    train_dtype = _coerce_dtype(config.get("train_dtype") or model_block.get("dtype") or "bfloat16")
    train_device = torch.device(str(config.get("train_device", "cuda")))
    save_dtype = _coerce_dtype(config.get("output_dtype"), default=train_dtype)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    learning_rate = float(config.get("learning_rate", 1e-4))
    batch_size = int(config.get("batch_size") or config.get("data", {}).get("batch_size", 1) or 1)
    grad_accum = int(config.get("gradient_accumulation_steps") or config.get("gradient_accumulation") or 1)
    max_steps = int(
        steps_override
        if steps_override is not None and steps_override > 0
        else (config.get("max_steps") or config.get("max_train_steps") or 100)
    )
    resolution = int(config.get("resolution") or config.get("data", {}).get("resolution") or 1024)
    gradient_checkpointing = str(
        memory_block.get("gradient_checkpointing") or config.get("gradient_checkpointing") or "on"
    )

    output_dir = Path(
        checkpoint_block.get("output_dir")
        or config.get("output_dir")
        or (Path("output") / source_path.stem)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if edit_mode:
        edit_pairs = _build_edit_training_pairs(config)
        if not edit_pairs:
            raise ValueError(
                "No edit training pairs found in configured concepts. "
                "Expected target images plus matching '*-condlabel.*' reference images."
            )
    else:
        pairs = _build_training_pairs(config)
        if not pairs:
            raise ValueError("No training images/captions found in configured concepts.")
    if steps_override is not None and steps_override > 0:
        max_required = max(1, max_steps * max(1, batch_size) * max(1, grad_accum))
        if edit_mode:
            edit_pairs = edit_pairs[:max_required]
        else:
            pairs = pairs[:max_required]

    print(f"[native/flux2] loading model from {resolved_model_path}")
    model = Flux2KleinModelLoader.load(
        resolved_model_path,
        model_type=flux_model_type,
        dtype=train_dtype,
        device="cpu",
        use_flux2_transformer=True,
    )

    if edit_mode:
        print(f"[native/flux2] caching edit latents+text embeddings for {len(edit_pairs)} samples")
        cached_edit = _cache_edit_training_data(model, edit_pairs, resolution, train_device, train_dtype)
    else:
        print(f"[native/flux2] caching latents+text embeddings for {len(pairs)} samples")
        cached = _cache_training_data(model, pairs, resolution, train_device, train_dtype)

    base_trainer_kwargs: dict[str, Any] = {
        "model_path": resolved_model_path,
        "model_variant": flux_model_type.value.replace("flux_2_", "").replace("_", "-"),
        "train_dtype": train_dtype,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": gradient_checkpointing,
        "enable_activation_offloading": _as_bool(
            memory_block.get("enable_activation_offloading", config.get("enable_activation_offloading", False))
        ),
        "enable_async_offloading": _as_bool(
            memory_block.get("enable_async_offloading", config.get("enable_async_offloading", False))
        ),
        "layer_offload_fraction": float(
            memory_block.get("layer_offload_fraction", config.get("layer_offload_fraction", 0.0))
        ),
        "train_device": str(train_device),
        "temp_device": str(config.get("temp_device", "cpu")),
        "resolution": resolution,
    }

    if edit_mode:
        trainer_cfg = Flux2EditTrainerConfig(
            **base_trainer_kwargs,
            max_references=int(config.get("max_references", 1)),
            use_masks=_as_bool(config.get("use_masks", False), False),
            reference_blend_mode=str(config.get("reference_blend_mode", "average")),
            conditioning_injection=str(config.get("conditioning_injection", "add")),
            reference_dropout=float(config.get("reference_dropout", 0.0)),
        )
        trainer = Flux2EditTrainer(trainer_cfg, model=model)
    else:
        trainer_cfg = Flux2ImageTrainerConfig(**base_trainer_kwargs)
        trainer = Flux2ImageTrainer(trainer_cfg, model=model)

    adapter = None
    if not full_finetune:
        rank = int(adapter_block.get("rank") or adapter_block.get("network_dim") or 16)
        alpha = float(adapter_block.get("alpha") or adapter_block.get("network_alpha") or rank)
        adapter = create_adapter(
            adapter_type=adapter_type,
            rank=rank,
            alpha=alpha,
            model_type=flux_model_type.value,
            dropout=float(adapter_block.get("dropout", 0.0)),
            **_build_adapter_kwargs(adapter_block),
        )
        trainer.inject_adapter(adapter)
    trainer.to_train_mode()
    if adapter is not None and hasattr(adapter, "to"):
        adapter.to(train_device, dtype=train_dtype)

    params = [param for param in trainer.get_trainable_params() if param.requires_grad]
    if not params:
        if full_finetune:
            raise RuntimeError("No trainable transformer parameters found for full-finetune mode.")
        raise RuntimeError("No trainable adapter parameters were found after injection.")

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
    print(f"[native/flux2] optimizer={optimizer_name} lr_scheduler={scheduler_name}")

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(checkpoint_block.get("save_every") or config.get("save_every") or 0)
    save_full_model = _as_bool(checkpoint_block.get("save_full_model", config.get("save_full_model", True)), True)

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}

    for step in range(1, max_steps + 1):
        if edit_mode:
            batch = _pick_edit_batch(cached_edit, batch_size, train_device, train_dtype)
        else:
            batch = _pick_batch(cached, batch_size, train_device, train_dtype)
        with torch.autocast(device_type=train_device.type, dtype=train_dtype, enabled=autocast_enabled):
            step_result = trainer.training_step(batch)
            loss = step_result["loss"]
        scaled_loss = loss / float(max(grad_accum, 1))
        scaled_loss.backward()

        should_step = (step % max(grad_accum, 1) == 0) or (step == max_steps)
        if should_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"[native/flux2] step {step}/{max_steps} loss={float(loss.detach().cpu()):.6f}")

        _maybe_sample(
            config,
            model_type=flux_model_type,
            model_path=resolved_model_path,
            output_dir=output_dir,
            step=step,
            train_device=train_device,
            train_dtype=train_dtype,
            default_resolution=resolution,
        )

        if save_every > 0 and step % save_every == 0:
            if adapter is not None:
                ckpt_path = output_dir / f"{adapter_type}_step_{step:06d}.safetensors"
                adapter.save(str(ckpt_path))
            elif save_full_model:
                ckpt_path = output_dir / f"transformer_step_{step:06d}.safetensors"
                saved_path = _save_transformer_state(
                    trainer.model.transformer,
                    ckpt_path,
                    save_dtype=save_dtype,
                )
                print(f"[native/flux2] saved full transformer checkpoint to {saved_path}")

    if adapter is not None:
        final_path = output_dir / f"{adapter_type}_last.safetensors"
        adapter.save(str(final_path))
        print(f"[native/flux2] training complete, adapter saved to {final_path}")
    elif save_full_model:
        final_path = output_dir / "transformer_last.safetensors"
        saved_path = _save_transformer_state(
            trainer.model.transformer,
            final_path,
            save_dtype=save_dtype,
        )
        print(f"[native/flux2] training complete, full transformer saved to {saved_path}")
    else:
        print("[native/flux2] training complete (save_full_model=false, no full checkpoint written)")

    return 0
