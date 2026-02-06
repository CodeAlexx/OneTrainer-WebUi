"""Native FLUX.2/Klein training path for EriTrainer.

This module intentionally avoids the OneTrainer bridge and runs a minimal
native training loop for FLUX.2/Klein variants using EriTrainer components.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from eritrainer.adapters import create_adapter
from eritrainer.core.interfaces import ModelType
from eritrainer.models.flux2_klein import Flux2KleinModelLoader
from eritrainer.sampling.sampler import create_sampler
from eritrainer.training.flux2.image_trainer import Flux2ImageTrainer, Flux2ImageTrainerConfig


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_FULL_TRAINING_METHODS = {"full", "full_finetune", "full_fine_tune", "fine_tune", "finetune"}
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
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", "null"}:
        return False
    return default


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
    if isinstance(value, (list, tuple, set)):
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
            caption = _load_caption(image_path, caption_ext)
            if caption:
                pairs.append((image_path, caption))
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


def _maybe_sample(
    config: dict[str, Any],
    *,
    model_type: ModelType,
    model_path: str,
    output_dir: Path,
    step: int,
    train_device: torch.device,
    train_dtype: torch.dtype,
) -> None:
    sample_block = config.get("sample", {}) if isinstance(config.get("sample"), dict) else {}
    if not sample_block.get("enabled", False):
        return

    interval = int(sample_block.get("interval", 0) or 0)
    if interval <= 0 or step % interval != 0:
        return

    prompts = sample_block.get("prompts") or []
    if not prompts:
        return

    prompt = str(prompts[0])
    negative_prompt = str(sample_block.get("negative_prompt", ""))
    seed = int((sample_block.get("seeds") or [42])[0])

    sampler = create_sampler(model_type, model={"path": model_path})
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"step_{step:06d}.png"

    sampler.sample(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_path=model_path,
        height=int(sample_block.get("height", 1024)),
        width=int(sample_block.get("width", 1024)),
        num_inference_steps=int(sample_block.get("num_inference_steps", 20)),
        guidance_scale=float(sample_block.get("guidance_scale", 4.0)),
        seed=seed,
        device=train_device,
        dtype=train_dtype,
        output_path=out_path,
        unload=True,
    )


def run_native_flux2_training(
    config: dict[str, Any],
    *,
    source_path: Path,
    steps_override: int | None = None,
) -> int:
    model_block = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    memory_block = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    checkpoint_block = config.get("checkpoint", {}) if isinstance(config.get("checkpoint"), dict) else {}
    optimizer_block = config.get("optimizer", {}) if isinstance(config.get("optimizer"), dict) else {}
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

    train_dtype = _coerce_dtype(config.get("train_dtype") or model_block.get("dtype") or "bfloat16")
    train_device = torch.device(str(config.get("train_device", "cuda")))
    save_dtype = _coerce_dtype(config.get("output_dtype"), default=train_dtype)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
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

    pairs = _build_training_pairs(config)
    if not pairs:
        raise ValueError("No training images/captions found in configured concepts.")
    if steps_override is not None and steps_override > 0:
        max_required = max(1, max_steps * max(1, batch_size) * max(1, grad_accum))
        pairs = pairs[:max_required]

    print(f"[native/flux2] loading model from {resolved_model_path}")
    model = Flux2KleinModelLoader.load(
        resolved_model_path,
        model_type=flux_model_type,
        dtype=train_dtype,
        device="cpu",
        use_flux2_transformer=True,
    )

    print(f"[native/flux2] caching latents+text embeddings for {len(pairs)} samples")
    cached = _cache_training_data(model, pairs, resolution, train_device, train_dtype)

    trainer_cfg = Flux2ImageTrainerConfig(
        model_path=resolved_model_path,
        model_variant=flux_model_type.value.replace("flux_2_", "").replace("_", "-"),
        train_dtype=train_dtype,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        enable_activation_offloading=_as_bool(
            memory_block.get("enable_activation_offloading", config.get("enable_activation_offloading", False))
        ),
        enable_async_offloading=_as_bool(
            memory_block.get("enable_async_offloading", config.get("enable_async_offloading", False))
        ),
        layer_offload_fraction=float(
            memory_block.get("layer_offload_fraction", config.get("layer_offload_fraction", 0.0))
        ),
        train_device=str(train_device),
        temp_device=str(config.get("temp_device", "cpu")),
        resolution=resolution,
    )

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

    optimizer_name = str(optimizer_block.get("optimizer") or config.get("optimizer") or "adamw").lower()
    if optimizer_name == "adafactor":
        from transformers import Adafactor

        optimizer = Adafactor(
            params,
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=float(optimizer_block.get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=float(optimizer_block.get("weight_decay", 0.0)),
        )

    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    save_every = int(checkpoint_block.get("save_every") or config.get("save_every") or 0)
    save_full_model = _as_bool(checkpoint_block.get("save_full_model", config.get("save_full_model", True)), True)

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}

    for step in range(1, max_steps + 1):
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
