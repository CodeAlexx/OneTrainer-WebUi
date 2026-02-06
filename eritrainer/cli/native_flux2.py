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
    if "adapter" in config and isinstance(config["adapter"], dict):
        adapter_block = config["adapter"]
        return str(adapter_block.get("type", "lora")).lower(), adapter_block

    peft_type = str(config.get("peft_type", "lora")).lower()
    if peft_type in config and isinstance(config[peft_type], dict):
        return peft_type, config[peft_type]

    if "lora" in config and isinstance(config["lora"], dict):
        return "lora", config["lora"]
    if "lokr" in config and isinstance(config["lokr"], dict):
        return "lokr", config["lokr"]
    if "loha" in config and isinstance(config["loha"], dict):
        return "loha", config["loha"]
    if "locon" in config and isinstance(config["locon"], dict):
        return "locon", config["locon"]

    return peft_type, {}


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
        enable_activation_offloading=bool(
            memory_block.get("enable_activation_offloading", config.get("enable_activation_offloading", False))
        ),
        enable_async_offloading=bool(
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

    adapter_type, adapter_block = _extract_adapter_config(config)
    adapter = create_adapter(
        adapter_type=adapter_type,
        rank=int(adapter_block.get("rank", 16)),
        alpha=float(adapter_block.get("alpha", 16.0)),
        model_type=flux_model_type.value,
        dropout=float(adapter_block.get("dropout", 0.0)),
    )
    trainer.inject_adapter(adapter)
    trainer.to_train_mode()
    if hasattr(adapter, "to"):
        adapter.to(train_device, dtype=train_dtype)

    params = list(trainer.get_trainable_params())
    if not params:
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

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = train_device.type == "cuda" and train_dtype in {torch.bfloat16, torch.float16}

    for step in range(1, max_steps + 1):
        batch = _pick_batch(cached, batch_size, train_device, train_dtype)
        with torch.autocast(device_type=train_device.type, dtype=train_dtype, enabled=autocast_enabled):
            step_result = trainer.training_step(batch)
            loss = step_result["loss"]
        scaled_loss = loss / float(max(grad_accum, 1))
        scaled_loss.backward()

        if step % max(grad_accum, 1) == 0:
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
            ckpt_path = output_dir / f"{adapter_type}_step_{step:06d}.safetensors"
            adapter.save(str(ckpt_path))

    final_path = output_dir / f"{adapter_type}_last.safetensors"
    adapter.save(str(final_path))
    print(f"[native/flux2] training complete, adapter saved to {final_path}")

    return 0
