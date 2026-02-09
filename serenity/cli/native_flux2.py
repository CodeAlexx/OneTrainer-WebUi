"""Native FLUX.2/Klein training path for Serenity.

Native Flux 2 training pipeline without bridge mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serenity.adapters import create_adapter
from serenity.cli.utils import (
    IMAGE_EXTENSIONS as _IMAGE_EXTENSIONS,
    COND_LABEL_SUFFIXES as _COND_LABEL_SUFFIXES,
    coerce_dtype as _coerce_dtype,
    as_bool as _as_bool,
    coerce_int_or_none as _coerce_int_or_none,
    optional_int as _optional_int,
    optional_float as _optional_float,
    normalize_model_type as _normalize_model_type,
    first_config_value as _first_config_value,
    resolve_hf_local_path as _resolve_hf_local_path,
    is_condlabel_image as _is_condlabel_image,
    strip_condlabel_suffix as _strip_condlabel_suffix,
    load_image_tensor as _load_image_tensor,
    load_caption as _load_caption,
    collect_concept_dirs as _collect_concept_dirs,
    build_training_pairs as _build_training_pairs,
    resolve_reference_image_path as _resolve_reference_image_path,
    build_edit_training_pairs as _build_edit_training_pairs,
)
from serenity.core.interfaces import ModelType
from serenity.models.flux2_klein import Flux2KleinModelLoader
from serenity.sampling.sampler import create_sampler
from serenity.training.flux2.edit_trainer import Flux2EditTrainer, Flux2EditTrainerConfig
from serenity.training.flux2.image_trainer import Flux2ImageTrainer, Flux2ImageTrainerConfig

import torch

from serenity.cli.flux2_optimizer import (
    _create_lr_scheduler,
    _create_optimizer,
    _normalize_optimizer_name,
    _normalize_scheduler_name,
    _resolve_optimizer_steps,
    _resolve_warmup_steps,
    _resolve_scheduler_min_factor,
    _resolve_scheduler_cycles,
)

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


@dataclass
class _CachedSample:
    latents: torch.Tensor
    text_embeddings: torch.Tensor


@dataclass
class _CachedEditSample:
    target_latents: torch.Tensor
    reference_latents: torch.Tensor
    text_embeddings: torch.Tensor


# _resolve_hf_local_path, _coerce_dtype, _normalize_model_type, _as_bool,
# _first_config_value — imported from serenity.cli.utils above


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


# _coerce_int_or_none, _optional_int, _optional_float — imported from serenity.cli.utils above


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


# _collect_concept_dirs, _load_caption, _is_condlabel_image, _strip_condlabel_suffix,
# _resolve_reference_image_path, _load_image_tensor, _build_training_pairs,
# _build_edit_training_pairs — imported from serenity.cli.utils above


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
    except (ImportError, OSError):
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
    live_adapter: Any = None,
    default_resolution: int | None = None,
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
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_assistant_lora_path: Path | None = None
    sample_assistant_lora_strength = _optional_float(
        sample_block.get("assistant_lora_inference_strength") or sample_block.get("assistant_lora_strength")
    )
    if use_live_adapter and live_adapter is not None and hasattr(live_adapter, "save"):
        try:
            adapter_cache_dir = samples_dir / ".adapter_cache"
            adapter_cache_dir.mkdir(parents=True, exist_ok=True)
            sample_assistant_lora_path = adapter_cache_dir / f"step_{step:06d}.safetensors"
            live_adapter.save(str(sample_assistant_lora_path))
            if sample_assistant_lora_strength is None:
                sample_assistant_lora_strength = 1.0
        except (OSError, RuntimeError) as exc:
            print(f"[native/flux2] warning: failed to snapshot live adapter for sampling: {exc}")
            sample_assistant_lora_path = None

    sample_device_raw = sample_block.get("device") or sample_block.get("sample_device") or config.get("sample_device")
    sample_device = torch.device(str(sample_device_raw)) if sample_device_raw else train_device
    sample_dtype = _coerce_dtype(sample_block.get("dtype") or sample_block.get("sample_dtype"), default=train_dtype)
    allow_cpu_fallback = _as_bool(sample_block.get("oom_fallback_cpu"), False)
    if sample_device.type == "cpu":
        sample_dtype = torch.float32

    def _run_sampling(device: torch.device, dtype: torch.dtype) -> None:
        sampler_model: dict[str, Any] = {"path": model_path}
        if sample_assistant_lora_path is not None:
            sampler_model["assistant_lora_path"] = str(sample_assistant_lora_path)
            if sample_assistant_lora_strength is not None:
                sampler_model["assistant_lora_inference_strength"] = float(sample_assistant_lora_strength)
        sampler = create_sampler(model_type, model=sampler_model)
        for prompt_index, prompt in enumerate(prompts_to_run):
            seed = seeds[prompt_index % len(seeds)]
            out_path = samples_dir / f"step_{step:06d}_p{prompt_index:02d}_s{seed}{output_ext}"
            sample_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_path": model_path,
                "seed": seed,
                "device": device,
                "dtype": dtype,
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

            sampler.sample(**sample_kwargs)

    try:
        _run_sampling(sample_device, sample_dtype)
    except RuntimeError as exc:
        # FLUX.2 9B sampling may OOM on 24GB cards while training.
        message = str(exc).lower()
        if sample_device.type == "cuda" and "out of memory" in message:
            print("[native/flux2] warning: sampling OOM on CUDA")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not allow_cpu_fallback:
                print("[native/flux2] warning: skipping sample (set sample.oom_fallback_cpu=true to retry on CPU)")
                return
            print("[native/flux2] warning: retrying sample on CPU")
            _run_sampling(torch.device("cpu"), torch.float32)
            return
        raise


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
            model_type=flux_model_type.value,
            dropout=float(adapter_block.get("dropout", 0.0)),
            backend=adapter_backend,
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
    sample_block = config.get("sample", {}) if isinstance(config.get("sample"), dict) else {}
    sample_at_start = _as_bool(sample_block.get("sample_at_start"), False)

    if sample_at_start:
        _maybe_sample(
            config,
            model_type=flux_model_type,
            model_path=resolved_model_path,
            output_dir=output_dir,
            step=0,
            train_device=train_device,
            train_dtype=train_dtype,
            live_adapter=adapter,
            default_resolution=resolution,
        )

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
            live_adapter=adapter,
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
