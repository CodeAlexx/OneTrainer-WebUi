"""CLI command implementations for EriTrainer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs") from exc
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _is_onetrainer_config(config: dict[str, Any]) -> bool:
    return "base_model_name" in config or "output_model_destination" in config


def _to_enum_name(value: str, mapping: dict[str, str], default: str) -> str:
    if value is None:
        return default
    return mapping.get(str(value).strip().lower(), default)


MODEL_TYPE_MAP: dict[str, str] = {
    "flux_dev": "FLUX_DEV_1",
    "flux_dev_1": "FLUX_DEV_1",
    "flux_fill_dev": "FLUX_FILL_DEV_1",
    "flux_fill_dev_1": "FLUX_FILL_DEV_1",
    "flux_fill": "FLUX_FILL_DEV_1",
    "flux_schnell": "FLUX_DEV_1",
    "zimage": "Z_IMAGE",
    "z_image": "Z_IMAGE",
    "qwen": "QWEN",
    "qwen_image_edit": "QWEN",
    "sdxl": "STABLE_DIFFUSION_XL_10_BASE",
    "sdxl_10_base": "STABLE_DIFFUSION_XL_10_BASE",
    "sdxl_base": "STABLE_DIFFUSION_XL_10_BASE",
    "sdxl_inpainting": "STABLE_DIFFUSION_XL_10_BASE_INPAINTING",
    "sdxl_inpaint": "STABLE_DIFFUSION_XL_10_BASE_INPAINTING",
    "sd15": "STABLE_DIFFUSION_15",
    "sd_15": "STABLE_DIFFUSION_15",
    "sd15_inpainting": "STABLE_DIFFUSION_15_INPAINTING",
    "sd_15_inpainting": "STABLE_DIFFUSION_15_INPAINTING",
    "sd15_inpaint": "STABLE_DIFFUSION_15_INPAINTING",
    "sd20": "STABLE_DIFFUSION_20",
    "sd_20": "STABLE_DIFFUSION_20",
    "sd20_base": "STABLE_DIFFUSION_20_BASE",
    "sd_20_base": "STABLE_DIFFUSION_20_BASE",
    "sd20_inpainting": "STABLE_DIFFUSION_20_INPAINTING",
    "sd_20_inpainting": "STABLE_DIFFUSION_20_INPAINTING",
    "sd20_depth": "STABLE_DIFFUSION_20_DEPTH",
    "sd_20_depth": "STABLE_DIFFUSION_20_DEPTH",
    "sd21": "STABLE_DIFFUSION_21",
    "sd_21": "STABLE_DIFFUSION_21",
    "sd21_base": "STABLE_DIFFUSION_21_BASE",
    "sd_21_base": "STABLE_DIFFUSION_21_BASE",
    "sd3": "STABLE_DIFFUSION_3",
    "sd_3": "STABLE_DIFFUSION_3",
    "sd35": "STABLE_DIFFUSION_35",
    "sd_35": "STABLE_DIFFUSION_35",
    "sd3.5": "STABLE_DIFFUSION_35",
    "stable_diffusion_3": "STABLE_DIFFUSION_3",
    "stable_diffusion_35": "STABLE_DIFFUSION_35",
    "stable_diffusion_3.5": "STABLE_DIFFUSION_35",
    "wuerstchen": "WUERSTCHEN_2",
    "wuerstchen_2": "WUERSTCHEN_2",
    "stable_cascade": "STABLE_CASCADE_1",
    "stable_cascade_1": "STABLE_CASCADE_1",
    "pixart": "PIXART_ALPHA",
    "pixart_alpha": "PIXART_ALPHA",
    "pixart_sigma": "PIXART_SIGMA",
    "sana": "SANA",
    "hunyuan_video": "HUNYUAN_VIDEO",
    "hidream": "HI_DREAM_FULL",
    "hi_dream_full": "HI_DREAM_FULL",
    "chroma": "CHROMA_1",
    "chroma_1": "CHROMA_1",
    "flux_2": "FLUX_2",
    "flux2": "FLUX_2",
    "flux": "FLUX_2",
    "flux_2_klein": "FLUX_2",
    "flux2_klein": "FLUX_2",
    "flux_2_klein_4b": "FLUX_2",
    "flux_2_klein_9b": "FLUX_2",
    "flux2_klein_4b": "FLUX_2",
    "flux2_klein_9b": "FLUX_2",
}


def _normalize_model_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _map_model_type(normalized: str) -> str:
    if normalized in MODEL_TYPE_MAP:
        return MODEL_TYPE_MAP[normalized]
    if normalized.startswith("flux_fill"):
        return "FLUX_FILL_DEV_1"
    if normalized.startswith("flux_dev") or normalized == "flux_schnell":
        return "FLUX_DEV_1"
    if "flux" in normalized:
        return "FLUX_2"
    if normalized.startswith("sd3"):
        return "STABLE_DIFFUSION_35" if "5" in normalized else "STABLE_DIFFUSION_3"
    raise ValueError(f"Unsupported model_type for this path: {normalized}")


def _is_native_flux2_type(normalized: str) -> bool:
    if not normalized:
        return False
    if normalized in {
        "flux_2",
        "flux2",
        "flux",
        "flux_2_dev",
        "flux2_dev",
        "flux_2_klein",
        "flux2_klein",
        "flux_2_klein_4b",
        "flux_2_klein_9b",
        "flux2_klein_4b",
        "flux2_klein_9b",
        "flux_2_klein_4b_base",
        "flux_2_klein_9b_base",
    }:
        return True
    return normalized.startswith("flux_2") or normalized.startswith("flux2")


def _is_structured_config(config: dict[str, Any]) -> bool:
    for key in ("model", "data", "adapter", "memory", "sample", "checkpoint", "logging", "optimizer", "scheduler"):
        if isinstance(config.get(key), dict):
            return True
    return False


def _map_quantization_dtype(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "off", "false", "null"}:
        return None
    if normalized in {"int8", "int_8", "int-8", "int"}:
        return "INT_8"
    if normalized in {"float8", "fp8", "float_8", "fp-8"}:
        return "FLOAT_8"
    if normalized in {"nf4", "nfloat4", "nfloat_4"}:
        return "NFLOAT_4"
    if normalized in {"int_w8a8", "intw8a8", "w8a8_int"}:
        return "INT_W8A8"
    if normalized in {"float_w8a8", "fpw8a8", "w8a8_float"}:
        return "FLOAT_W8A8"
    if normalized in {"gguf"}:
        return "GGUF"
    if normalized in {"gguf_a8_float"}:
        return "GGUF_A8_FLOAT"
    if normalized in {"gguf_a8_int"}:
        return "GGUF_A8_INT"
    return None


def _resolve_hf_local_path(model_path: str) -> str:
    # Keep explicit local filesystem paths untouched.
    expanded = os.path.expanduser(model_path)
    if os.path.exists(expanded):
        return expanded

    # Resolve HF repo id to local cache snapshot path.
    if "/" not in model_path:
        raise FileNotFoundError(
            f"Model path not found locally: {model_path}. "
            "Expected a local path or a cached HF repo id."
        )

    org, name = model_path.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{org}--{name}"
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"HF cache not found for {model_path}: {repo_dir}"
        )

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


def _extract_adapter_block(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "adapter" in config and isinstance(config["adapter"], dict):
        adapter = config["adapter"]
        adapter_type = str(adapter.get("type", "lora")).lower()
        return adapter_type, adapter

    peft_type = str(config.get("peft_type", "lora")).lower()
    if peft_type in config and isinstance(config[peft_type], dict):
        return peft_type, config[peft_type]

    if "lora" in config and isinstance(config["lora"], dict):
        return "lora", config["lora"]
    if "lokr" in config and isinstance(config["lokr"], dict):
        return "lokr", config["lokr"]
    if "loha" in config and isinstance(config["loha"], dict):
        return "loha", config["loha"]

    return peft_type, {}


def _convert_eritrainer_to_onetrainer(
    config: dict[str, Any],
    source_path: Path,
    steps_override: int | None = None,
) -> dict[str, Any]:
    normalized_model_type = _normalize_model_type(config.get("model_type"))
    if not normalized_model_type:
        raise ValueError("Missing model_type in config.")

    onetrainer_model_type = _map_model_type(normalized_model_type)
    structured = _is_structured_config(config)

    model_block = config.get("model", {}) if structured and isinstance(config.get("model"), dict) else {}
    base_model = (
        model_block.get("path")
        or config.get("base_model")
        or config.get("base_model_name")
        or config.get("transformer_path")
    )
    if not base_model:
        raise ValueError("Missing model path. Expected `model.path` or `base_model`.")

    resolved_model = _resolve_hf_local_path(str(base_model))

    data_block = config.get("data", {}) if structured and isinstance(config.get("data"), dict) else {}
    memory_block = config.get("memory", {}) if structured and isinstance(config.get("memory"), dict) else {}
    optimizer_block = config.get("optimizer", {}) if structured and isinstance(config.get("optimizer"), dict) else {}
    scheduler_block = config.get("scheduler", {}) if structured and isinstance(config.get("scheduler"), dict) else {}
    sample_block = config.get("sample", {}) if structured and isinstance(config.get("sample"), dict) else {}
    checkpoint_block = config.get("checkpoint", {}) if structured and isinstance(config.get("checkpoint"), dict) else {}
    logging_block = config.get("logging", {}) if structured and isinstance(config.get("logging"), dict) else {}

    adapter_type, adapter_block = _extract_adapter_block(config)

    output_dir = (
        checkpoint_block.get("output_dir")
        or config.get("output_dir")
        or str(Path("output") / source_path.stem)
    )
    output_dir = str(Path(output_dir).expanduser())
    workspace_dir = str(Path(output_dir) / "workspace")

    dtype_map = {
        "float16": "FLOAT_16",
        "fp16": "FLOAT_16",
        "bfloat16": "BFLOAT_16",
        "bf16": "BFLOAT_16",
        "float32": "FLOAT_32",
        "fp32": "FLOAT_32",
    }
    train_dtype = _to_enum_name(
        str(
            config.get("train_dtype")
            or model_block.get("dtype")
            or config.get("dtype")
            or "bfloat16"
        ),
        dtype_map,
        "BFLOAT_16",
    )
    output_dtype = _to_enum_name(
        str(config.get("output_dtype") or "float32"),
        dtype_map,
        "FLOAT_32",
    )

    checkpointing_map = {
        "on": "ON",
        "true": "ON",
        "off": "OFF",
        "false": "OFF",
        "cpu_offloaded": "CPU_OFFLOADED",
    }
    gradient_checkpointing = _to_enum_name(
        str(
            memory_block.get("gradient_checkpointing")
            or config.get("gradient_checkpointing")
            or "on"
        ),
        checkpointing_map,
        "ON",
    )

    scheduler_name = _to_enum_name(
        str(
            scheduler_block.get("scheduler")
            or config.get("lr_scheduler")
            or "constant"
        ),
        {
            "constant": "CONSTANT",
            "linear": "LINEAR",
            "cosine": "COSINE",
            "cosine_with_restarts": "COSINE_WITH_RESTARTS",
            "cosine_with_hard_restarts": "COSINE_WITH_HARD_RESTARTS",
            "rex": "REX",
            "adafactor": "ADAFACTOR",
        },
        "CONSTANT",
    )

    optimizer_name = _to_enum_name(
        str(optimizer_block.get("optimizer") or "adamw"),
        {
            "adamw": "ADAMW",
            "adam": "ADAM",
            "adafactor": "ADAFACTOR",
            "sgd": "SGD",
            "lion": "LION",
        },
        "ADAMW",
    )

    adapter_type_normalized = adapter_type.lower()
    peft_type = {
        "lora": "LORA",
        "loha": "LOHA",
        "lokr": "LOKR",
        "oft": "OFT_2",
        "oft_2": "OFT_2",
    }.get(adapter_type_normalized, "LORA")

    concepts_in = data_block.get("concepts") or config.get("concepts") or []
    concepts_out: list[dict[str, Any]] = []
    for idx, concept in enumerate(concepts_in):
        if isinstance(concept, str):
            concept_path = concept
            repeats = 1.0
        elif isinstance(concept, dict):
            concept_path = concept.get("path")
            repeats = concept.get("num_repeats", concept.get("repeats", 1.0))
        else:
            continue

        if not concept_path:
            continue

        concept_path = str(Path(concept_path).expanduser())
        concepts_out.append(
            {
                "name": Path(concept_path).name or f"concept_{idx}",
                "path": concept_path,
                "enabled": True,
                "balancing": float(repeats),
            }
        )

    if not concepts_out:
        raise ValueError("No valid concepts found in config.")

    # Optional Flux2-style block swap hint -> layer offload fraction.
    block_swap_fraction = None
    if "blocks_to_swap" in config and "flux" in normalized_model_type:
        blocks_to_swap = float(config["blocks_to_swap"])
        total_blocks = 32.0 if "9b" in normalized_model_type else 25.0
        block_swap_fraction = max(0.0, min(blocks_to_swap / total_blocks, 1.0))

    default_layer_offload = 0.5 if "9b" in normalized_model_type else 0.0
    layer_offload_fraction = float(
        memory_block.get(
            "layer_offload_fraction",
            config.get(
                "layer_offload_fraction",
                block_swap_fraction if block_swap_fraction is not None else default_layer_offload,
            ),
        )
    )

    quantization_dtype = _map_quantization_dtype(
        memory_block.get("quantization") or config.get("quantization")
    )
    if normalized_model_type in {"qwen", "qwen_image_edit"} and quantization_dtype == "INT_8":
        print(
            "Warning: INT_8 quantization relies on bitsandbytes Linear8bitLt. "
            "Using INT_W8A8 compatibility mode for Qwen.",
            file=sys.stderr,
        )
        quantization_dtype = "INT_W8A8"

    transformer_weight_dtype = quantization_dtype or train_dtype
    unet_weight_dtype = quantization_dtype or train_dtype

    text_encoder_weight_dtype = train_dtype
    if normalized_model_type in {"qwen", "qwen_image_edit"} and quantization_dtype:
        text_encoder_weight_dtype = quantization_dtype

    cache_text_embeddings = bool(
        config.get("cache_text_embeddings")
        or data_block.get("cache_text_embeddings")
    )
    train_text_encoder = config.get("train_text_encoder")
    if train_text_encoder is None:
        if cache_text_embeddings or normalized_model_type in {"qwen", "qwen_image_edit"}:
            train_text_encoder = False

    train_text_encoder_2 = config.get("train_text_encoder_2")
    if train_text_encoder_2 is None and cache_text_embeddings:
        train_text_encoder_2 = False

    masked_training = bool(
        config.get("masked_training")
        or data_block.get("masked_training")
        or (normalized_model_type == "qwen_image_edit" and data_block.get("edit_mode"))
    )
    custom_conditioning_image = bool(
        config.get("custom_conditioning_image")
        or data_block.get("custom_conditioning_image")
    )

    dataloader_threads_default = 1 if (
        normalized_model_type in {"qwen", "qwen_image_edit", "zimage", "z_image"}
        or "flux" in normalized_model_type
    ) else 2

    dataloader_threads = int(
        data_block.get("num_workers")
        or config.get("dataloader_threads")
        or config.get("num_workers")
        or dataloader_threads_default
    )
    if normalized_model_type in {"qwen", "qwen_image_edit", "zimage", "z_image"} or "flux" in normalized_model_type:
        dataloader_threads = 1

    train_dict: dict[str, Any] = {
        "__version": 10,
        "training_method": _to_enum_name(
            str(config.get("training_method") or "lora"),
            {"lora": "LORA", "fine_tune": "FINE_TUNE", "finetune": "FINE_TUNE", "embedding": "EMBEDDING"},
            "LORA",
        ),
        "model_type": onetrainer_model_type,
        "peft_type": peft_type,
        "base_model_name": resolved_model,
        "output_model_destination": output_dir,
        "workspace_dir": workspace_dir,
        "cache_dir": str(
            Path(
                data_block.get("cache_dir")
                or config.get("cache_dir")
                or (Path(output_dir) / "cache")
            ).expanduser()
        ),
        "tensorboard": bool(logging_block.get("enable_tensorboard", False)),
        "concepts": concepts_out,
        "resolution": str(data_block.get("resolution") or config.get("resolution") or "1024"),
        "batch_size": int(data_block.get("batch_size") or config.get("batch_size") or 1),
        "dataloader_threads": dataloader_threads,
        "gradient_accumulation_steps": int(
            config.get("gradient_accumulation_steps")
            or config.get("gradient_accumulation")
            or 1
        ),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "epochs": int(config.get("epochs", 1)),
        "train_device": str(config.get("train_device", "cuda")),
        "temp_device": str(config.get("temp_device", "cpu")),
        "train_dtype": train_dtype,
        "fallback_train_dtype": "BFLOAT_16",
        "output_dtype": output_dtype,
        "gradient_checkpointing": gradient_checkpointing,
        "enable_activation_offloading": bool(
            memory_block.get("enable_activation_offloading", config.get("enable_activation_offloading", False))
        ),
        "enable_async_offloading": bool(
            memory_block.get("enable_async_offloading", config.get("enable_async_offloading", False))
        ),
        "layer_offload_fraction": float(layer_offload_fraction),
        "latent_caching": bool(
            data_block.get("cache_latents")
            if "cache_latents" in data_block
            else config.get("cache_latents", True)
        ),
        "learning_rate_scheduler": scheduler_name,
        "optimizer": {
            "optimizer": optimizer_name,
            "weight_decay": float(optimizer_block.get("weight_decay", 0.0)),
        },
        "lora_rank": int(adapter_block.get("rank", 16)),
        "lora_alpha": float(adapter_block.get("alpha", 16.0)),
        "dropout_probability": float(adapter_block.get("dropout", 0.0)),
        "lora_weight_dtype": train_dtype,
        "save_every": int(checkpoint_block.get("save_every", config.get("save_every", 0))),
        "save_every_unit": _to_enum_name(
            str(checkpoint_block.get("save_every_unit", "step")),
            {"step": "STEP", "epoch": "EPOCH", "never": "NEVER", "always": "ALWAYS"},
            "STEP",
        ),
        "ema": _to_enum_name(
            str(config.get("ema_mode", "off")),
            {"off": "OFF", "gpu": "GPU", "cpu": "CPU"},
            "OFF",
        ),
        "samples": [],
        "sample_after_unit": "NEVER",
    }

    if adapter_type_normalized == "lokr":
        lokr_dim = int(adapter_block.get("rank") or adapter_block.get("dim") or train_dict["lora_rank"])
        lokr_alpha = float(adapter_block.get("alpha") or adapter_block.get("lokr_alpha") or lokr_dim)
        train_dict.update(
            {
                "lokr_dim": lokr_dim,
                "lokr_alpha": lokr_alpha,
                "lokr_decompose_both": bool(adapter_block.get("decompose_both", False)),
                "lokr_decompose_factor": int(
                    adapter_block.get("decompose_factor", adapter_block.get("factor", -1))
                ),
                "lokr_use_tucker": bool(adapter_block.get("use_tucker", False)),
                "lokr_full_matrix": bool(adapter_block.get("full_matrix", False)),
                "lokr_weight_decompose": bool(adapter_block.get("weight_decompose", False)),
                "lokr_dora_on_output": bool(adapter_block.get("dora_on_output", True)),
                "lokr_rs_lora": bool(adapter_block.get("rs_lora", False)),
            }
        )

    if adapter_type_normalized == "lokr":
        lokr_block = adapter_block.get("lokr", {}) if isinstance(adapter_block.get("lokr"), dict) else adapter_block
        train_dict["lokr_factor"] = int(lokr_block.get("factor", 2))
        train_dict["lokr_decompose_both"] = bool(lokr_block.get("decompose_both", False))
        train_dict["lokr_use_tucker"] = bool(lokr_block.get("use_tucker", False))
        train_dict["lokr_full_matrix"] = bool(lokr_block.get("full_matrix", False))

    if onetrainer_model_type in {"FLUX_2", "Z_IMAGE", "QWEN", "STABLE_DIFFUSION_3", "STABLE_DIFFUSION_35"}:
        train_dict["transformer"] = {
            "weight_dtype": transformer_weight_dtype,
            "train": True,
        }
    if onetrainer_model_type in {"STABLE_DIFFUSION_XL_10_BASE", "STABLE_DIFFUSION_15"}:
        train_dict["unet"] = {
            "weight_dtype": unet_weight_dtype,
            "train": True,
        }

    text_encoder_config = {"weight_dtype": text_encoder_weight_dtype}
    if train_text_encoder is not None:
        text_encoder_config["train"] = bool(train_text_encoder)
    train_dict["text_encoder"] = text_encoder_config

    if onetrainer_model_type == "STABLE_DIFFUSION_XL_10_BASE":
        text_encoder_2_config = {"weight_dtype": text_encoder_weight_dtype}
        if train_text_encoder_2 is not None:
            text_encoder_2_config["train"] = bool(train_text_encoder_2)
        train_dict["text_encoder_2"] = text_encoder_2_config

    train_dict["vae"] = {
        "weight_dtype": train_dtype,
    }

    if masked_training:
        train_dict["masked_training"] = True
    if custom_conditioning_image:
        train_dict["custom_conditioning_image"] = True

    target_modules = adapter_block.get("target_modules")
    if isinstance(target_modules, list) and target_modules:
        train_dict["layer_filter"] = ",".join(str(x) for x in target_modules)

    if "min_noising_strength" in config:
        train_dict["min_noising_strength"] = float(config["min_noising_strength"])
    if "max_noising_strength" in config:
        train_dict["max_noising_strength"] = float(config["max_noising_strength"])

    if sample_block.get("enabled"):
        prompts = sample_block.get("prompts") or []
        seeds = sample_block.get("seeds") or [42] * max(len(prompts), 1)
        sample_entries: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            seed = int(seeds[i if i < len(seeds) else 0])
            sample_entries.append(
                {
                    "enabled": True,
                    "prompt": str(prompt),
                    "negative_prompt": str(sample_block.get("negative_prompt", "")),
                    "height": int(sample_block.get("height", 1024)),
                    "width": int(sample_block.get("width", 1024)),
                    "seed": seed,
                    "diffusion_steps": int(sample_block.get("num_inference_steps", 20)),
                    "cfg_scale": float(sample_block.get("guidance_scale", 1.0)),
                }
            )

        train_dict["samples"] = sample_entries
        train_dict["sample_after"] = int(sample_block.get("interval", 0))
        train_dict["sample_after_unit"] = _to_enum_name(
            str(sample_block.get("interval_unit", "step")),
            {"step": "STEP", "epoch": "EPOCH", "minute": "MINUTE", "hour": "HOUR", "never": "NEVER"},
            "STEP",
        )
        train_dict["sample_skip_first"] = 0 if sample_block.get("sample_at_start", False) else 1

    if "backup_every" in checkpoint_block:
        train_dict["backup_after"] = int(checkpoint_block["backup_every"])
        train_dict["backup_after_unit"] = "STEP"

    # Optional short-run override used for quick validation.
    if steps_override is not None and steps_override > 0:
        train_dict["epochs"] = 1
        train_dict["max_train_steps"] = int(steps_override)
        train_dict["latent_caching"] = False
        train_dict["save_every"] = 0
        train_dict["save_every_unit"] = "NEVER"
        train_dict["sample_after_unit"] = "NEVER"

    return train_dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eritrainer")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run training with an EriTrainer preset")
    train_parser.add_argument("config_positional", nargs="?", help="Path to config file")
    train_parser.add_argument("--config", dest="config_flag", help="Path to config file")
    train_parser.add_argument("--steps", type=int, default=None, help="Optional short-run step override")

    parser.add_argument("config_positional_root", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--config", dest="config_flag_root", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def _resolve_config_arg(ns: argparse.Namespace) -> str | None:
    return (
        getattr(ns, "config_flag", None)
        or getattr(ns, "config_positional", None)
        or getattr(ns, "config_flag_root", None)
        or getattr(ns, "config_positional_root", None)
    )


def train_command(args: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(args=args)

    command = ns.command or "train"
    if command != "train":
        parser.error(f"Unsupported command: {command}")

    config_arg = _resolve_config_arg(ns)
    if not config_arg:
        parser.error("Missing config path. Use `eritrainer train <config>` or `--config <path>`.")

    config_path = Path(config_arg).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = _load_config(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/object")

    normalized_model_type = _normalize_model_type(
        cfg.get("model_type")
        or (
            cfg.get("model", {}).get("type")
            if isinstance(cfg.get("model"), dict)
            else None
        )
    )
    if not _is_onetrainer_config(cfg) and _is_native_flux2_type(normalized_model_type):
        from eritrainer.cli.native_flux2 import run_native_flux2_training

        return run_native_flux2_training(
            cfg,
            source_path=config_path,
            steps_override=getattr(ns, "steps", None),
        )

    if _is_onetrainer_config(cfg):
        train_cfg = cfg
    else:
        train_cfg = _convert_eritrainer_to_onetrainer(
            cfg,
            source_path=config_path,
            steps_override=getattr(ns, "steps", None),
        )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(train_cfg, tmp, indent=2)
        tmp_path = tmp.name

    try:
        cmd = [sys.executable, "scripts/train.py", "--config-path", tmp_path]
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        return subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[2]), env=env)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
