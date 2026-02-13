"""Native Serenity samplers backed by diffusers pipelines."""

from __future__ import annotations

import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType

import torch

from PIL import Image

_MODEL_TYPE_ALIASES: dict[str, ModelType] = {
    "flux": ModelType.FLUX_DEV,
    "flux_dev": ModelType.FLUX_DEV,
    "flux_fill": ModelType.FLUX_FILL_DEV,
    "flux_fill_dev": ModelType.FLUX_FILL_DEV,
    "flux_fill_dev_1": ModelType.FLUX_FILL_DEV,
    "flux_schnell": ModelType.FLUX_SCHNELL,
    "flux_2": ModelType.FLUX_2,
    "flux2": ModelType.FLUX_2,
    "flux_2_dev": ModelType.FLUX_2_DEV,
    "flux2_dev": ModelType.FLUX_2_DEV,
    "flux2_klein": ModelType.FLUX_2_KLEIN,
    "flux_2_klein": ModelType.FLUX_2_KLEIN,
    "flux2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
    "flux_2_klein_4b": ModelType.FLUX_2_KLEIN_4B,
    "flux2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
    "flux_2_klein_9b": ModelType.FLUX_2_KLEIN_9B,
    "zimage": ModelType.ZIMAGE,
    "z_image": ModelType.Z_IMAGE,
    "sd15": ModelType.SD15,
    "sd_15": ModelType.SD15,
    "sd15_inpainting": ModelType.SD15_INPAINTING,
    "sd_15_inpainting": ModelType.SD15_INPAINTING,
    "sd15_inpaint": ModelType.SD15_INPAINTING,
    "sd20": ModelType.SD20,
    "sd_20": ModelType.SD20,
    "sd20_base": ModelType.SD20_BASE,
    "sd_20_base": ModelType.SD20_BASE,
    "sd20_inpainting": ModelType.SD20_INPAINTING,
    "sd_20_inpainting": ModelType.SD20_INPAINTING,
    "sd20_depth": ModelType.SD20_DEPTH,
    "sd_20_depth": ModelType.SD20_DEPTH,
    "sd21": ModelType.SD21,
    "sd_21": ModelType.SD21,
    "sd21_base": ModelType.SD21_BASE,
    "sd_21_base": ModelType.SD21_BASE,
    "sdxl": ModelType.SDXL,
    "sdxl_10_base": ModelType.SDXL_10_BASE,
    "sdxl_inpainting": ModelType.SDXL_INPAINTING,
    "sdxl_inpaint": ModelType.SDXL_INPAINTING,
    "sd3": ModelType.SD3,
    "sd_3": ModelType.SD3,
    "sd35": ModelType.SD35,
    "sd_35": ModelType.SD35,
    "sd3.5": ModelType.SD35,
    "stable_diffusion_3": ModelType.SD3,
    "stable_diffusion_35": ModelType.SD35,
    "stable_diffusion_3.5": ModelType.SD35,
    "wuerstchen": ModelType.WUERSTCHEN_2,
    "wuerstchen_2": ModelType.WUERSTCHEN_2,
    "stable_cascade": ModelType.STABLE_CASCADE_1,
    "stable_cascade_1": ModelType.STABLE_CASCADE_1,
    "pixart": ModelType.PIXART_ALPHA,
    "pixart_alpha": ModelType.PIXART_ALPHA,
    "pixart_sigma": ModelType.PIXART_SIGMA,
    "sana": ModelType.SANA,
    "hunyuan_video": ModelType.HUNYUAN_VIDEO,
    "hidream": ModelType.HI_DREAM_FULL,
    "hi_dream_full": ModelType.HI_DREAM_FULL,
    "chroma": ModelType.CHROMA_1,
    "chroma_1": ModelType.CHROMA_1,
    "ltx2": ModelType.LTX2,
    "ltx": ModelType.LTX2,
    "ltx_video": ModelType.LTX2,
    "ltxvideo": ModelType.LTX2,
    "qwen": ModelType.QWEN,
    "qwen_image_edit": ModelType.QWEN_IMAGE_EDIT,
}

def _coerce_model_type(value: ModelType | str) -> ModelType:
    if isinstance(value, ModelType):
        return value

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in _MODEL_TYPE_ALIASES:
        return _MODEL_TYPE_ALIASES[normalized]

    return ModelType(normalized)


def _coerce_dtype(value: torch.dtype | str | None, device: torch.device) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "").replace("_", "")
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "float"}:
            return torch.float32

    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        except (RuntimeError, AttributeError):  # pragma: no cover - defensive fallback
            return torch.float16

    return torch.float32


def _coerce_device(value: str | torch.device | None) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_model_source(model_source: str) -> str:
    expanded = Path(model_source).expanduser()
    if expanded.exists():
        return str(expanded)

    # If this is an absolute filesystem path and it does not exist, avoid
    # reinterpreting it as a repo id. Try a case-insensitive HF-cache recovery
    # for `models--ORG--NAME` tokens first.
    if expanded.is_absolute():
        marker = "models--"
        as_posix = expanded.as_posix()
        if marker in as_posix:
            token = as_posix.split(marker, 1)[1].split("/", 1)[0]
            hub_root = Path.home() / ".cache" / "huggingface" / "hub"
            expected = f"{marker}{token}".lower()
            for candidate in hub_root.glob("models--*"):
                if candidate.name.lower() == expected:
                    refs_main = candidate / "refs" / "main"
                    if refs_main.exists():
                        revision = refs_main.read_text().strip()
                        snapshot = candidate / "snapshots" / revision
                        if snapshot.exists():
                            return str(snapshot)
                    snapshots_dir = candidate / "snapshots"
                    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
                    if snapshots:
                        return str(snapshots[-1])
        raise FileNotFoundError(f"Model source path does not exist: {expanded}")

    if "/" not in model_source:
        raise FileNotFoundError(
            f"Model source not found locally: {model_source}. "
            "Use a local path or a Hugging Face repo id that exists in local cache."
        )

    org, name = model_source.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{org}--{name}"
    if not repo_dir.exists():
        raise FileNotFoundError(f"HF cache for {model_source} not found at {repo_dir}")

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

    raise FileNotFoundError(f"No HF snapshots found for {model_source} under {repo_dir}")


def _load_pipeline_class(class_name: str):
    import diffusers

    if not hasattr(diffusers, class_name):
        raise AttributeError(f"diffusers has no pipeline class named {class_name}")
    return getattr(diffusers, class_name)


def _load_zimage_pipeline_without_model_index(
    *,
    model_source: str,
    dtype: torch.dtype,
):
    """Build a ZImagePipeline from component subfolders when model_index.json is absent."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from serenity.models.zimage import _component_has_weights, _validate_zimage_model_path

    model_root = _validate_zimage_model_path(model_source)
    scheduler_dir = model_root / "scheduler"
    if scheduler_dir.exists():
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_root),
            subfolder="scheduler",
            local_files_only=True,
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        subfolder="tokenizer",
        local_files_only=True,
        use_fast=False,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        str(model_root),
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        str(model_root),
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    )
    transformer_dir = model_root / "transformer"
    if not _component_has_weights(transformer_dir):
        raise FileNotFoundError(f"No transformer weights found under {transformer_dir}")
    # Prefer direct from_pretrained for sampling. The training-oriented loader uses
    # an accelerate/meta init path that can leave this fallback pipeline effectively
    # CPU-dispatched and extremely slow for denoising.
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(model_root),
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    return ZImagePipeline(
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
    )


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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _call_with_filtered_kwargs(fn, *args: Any, **kwargs: Any):
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(*args, **kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in params}
    return fn(*args, **filtered)


def _extract_assistant_lora_settings(model: Any) -> tuple[Path | None, str | None, float | None]:
    if model is None:
        return None, None, None

    def _get_attr(key: str):
        return getattr(model, key, None)

    get_value = model.get if isinstance(model, dict) else _get_attr

    if _as_bool(get_value("disable_assistant_lora"), False):
        return None, None, None

    path_raw = (
        get_value("assistant_lora_path")
        or get_value("turbo_adapter_path")
        or get_value("assistant_adapter_path")
    )
    path_text = _optional_text(path_raw)
    if not path_text:
        return None, None, None

    path = Path(path_text).expanduser()
    if not path.exists():
        print(f"[sampler] warning: assistant LoRA path does not exist: {path}")
        return None, None, None
    if not path.is_file():
        print(f"[sampler] warning: assistant LoRA path must be a file: {path}")
        return None, None, None

    weight_name = _optional_text(
        get_value("assistant_lora_weight_name")
        or get_value("turbo_adapter_weight_name")
        or get_value("assistant_adapter_weight_name")
    )
    strength_raw = get_value("assistant_lora_inference_strength")
    if strength_raw is None:
        strength_raw = get_value("assistant_lora_strength")
    if strength_raw is None:
        strength_raw = get_value("turbo_adapter_strength")
    strength = _optional_float(strength_raw)
    return path, weight_name, strength


def _freeze_adapter_parameters(module: Any, adapter_name: str) -> None:
    if module is None:
        return
    for name, param in module.named_parameters():
        if param is None:
            continue
        if f".{adapter_name}" in name and "lora_" in name:
            param.requires_grad_(False)


def _is_standard_lora_file(path: Path) -> bool:
    """Check if a safetensors file contains standard LoRA keys (not LyCORIS)."""
    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(path))
    except Exception:
        return True  # Can't check, assume standard

    keys_joined = " ".join(state_dict.keys())
    # LyCORIS adapter types have distinctive key patterns
    lycoris_markers = ("lokr_", "hada_", "oft_", "boft_", "diag_oft_", "ia3_")
    for marker in lycoris_markers:
        if marker in keys_joined:
            return False
    return True


def _load_assistant_lora_into_transformer(
    *,
    pipeline: Any,
    lora_path: Path,
    adapter_name: str = "assistant",
    weight_name: str | None = None,
    strength: float | None = None,
) -> bool:
    # Skip diffusers path entirely for non-standard-LoRA files (LyCORIS etc.)
    if not _is_standard_lora_file(lora_path):
        return False

    transformer = getattr(pipeline, "transformer", None)
    pipeline_cls = pipeline.__class__
    lora_state_fn = getattr(pipeline_cls, "lora_state_dict", None)
    load_fn = getattr(pipeline_cls, "load_lora_into_transformer", None)
    if transformer is None or lora_state_fn is None or load_fn is None:
        return False

    # Diffusers requires weight_name in offline mode.  When the caller
    # provides a direct file path, split it into (directory, filename) so
    # that diffusers can resolve it without reaching out to the Hub.
    effective_path: str = str(lora_path)
    effective_weight_name = weight_name
    if effective_weight_name is None and lora_path.is_file():
        effective_path = str(lora_path.parent)
        effective_weight_name = lora_path.name

    state_kwargs: dict[str, Any] = {"local_files_only": True}
    if effective_weight_name is not None:
        state_kwargs["weight_name"] = effective_weight_name
    if "return_alphas" in inspect.signature(lora_state_fn).parameters:
        state_kwargs["return_alphas"] = True

    lora_state = lora_state_fn(effective_path, **state_kwargs)
    state_dict: dict[str, torch.Tensor]
    network_alphas: dict[str, float] | None = None
    metadata: Any = None
    if isinstance(lora_state, tuple):
        if len(lora_state) >= 1:
            state_dict = lora_state[0]
        else:
            return False
        if len(lora_state) >= 2 and isinstance(lora_state[1], dict):
            network_alphas = lora_state[1]
        if len(lora_state) >= 3:
            metadata = lora_state[2]
    else:
        state_dict = lora_state

    load_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "adapter_name": adapter_name,
        "_pipeline": None,
        "low_cpu_mem_usage": False,
    }
    if network_alphas:
        load_kwargs["network_alphas"] = network_alphas
    if metadata is not None:
        load_kwargs["metadata"] = metadata

    _call_with_filtered_kwargs(load_fn, state_dict, **load_kwargs)
    _freeze_adapter_parameters(transformer, adapter_name)

    with suppress(Exception):
        if hasattr(transformer, "set_adapters"):
            if strength is None:
                transformer.set_adapters([adapter_name])
            else:
                transformer.set_adapters([adapter_name], [float(strength)])
        elif hasattr(transformer, "set_adapter"):
            transformer.set_adapter(adapter_name)

    return True


def _load_adapter_native(
    *,
    pipeline: Any,
    adapter_path: Path,
    strength: float | None = None,
    model_type: str = "flux_2_klein",
) -> bool:
    """Load any adapter (LoRA, LyCORIS, DoRA, etc.) via the native Serenity adapter system.

    This is the fallback when diffusers' built-in LoRA loader can't handle
    the file (e.g. LyCORIS/LoHa/LoKr/OFT adapters).

    Uses lycoris.create_lycoris_from_weights for LyCORIS files (LoKr, LoHa, OFT, etc.)
    which infers shapes from the state dict directly — no need to guess factor/rank.
    Falls back to Serenity's native adapter system for standard LoRA.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return False

    target_device = None
    target_dtype = None
    with suppress(Exception):
        ref_param = next(transformer.parameters())
        target_device = ref_param.device
        target_dtype = ref_param.dtype

    multiplier = float(strength) if strength is not None else 1.0

    # Try lycoris.create_lycoris_from_weights — handles LoKr/LoHa/OFT/etc. natively
    try:
        from lycoris import create_lycoris_from_weights

        network, weights_sd = create_lycoris_from_weights(
            multiplier, str(adapter_path), transformer,
        )
        if not network.loras:
            # Not a LyCORIS-compatible state dict. Fall through to Serenity
            # native adapter loading instead of silently returning.
            raise RuntimeError("LyCORIS loader found 0 modules")

        network.weights_sd = weights_sd
        network.apply_to()
        network.merge_to(weight=multiplier)
        network.restore()

        for param in transformer.parameters():
            param.requires_grad_(False)

        return True
    except ImportError:
        pass
    except Exception as lycoris_exc:
        print(f"[sampler] lycoris direct load failed ({lycoris_exc}), trying serenity adapter")

    # Fallback: Serenity native adapter system
    from safetensors.torch import load_file
    from serenity.adapters import create_adapter, detect_adapter_type, detect_rank

    state_dict = load_file(str(adapter_path))
    adapter_type = detect_adapter_type(state_dict)
    rank = detect_rank(state_dict)

    adapter = create_adapter(
        adapter_type=adapter_type,
        rank=rank,
        alpha=float(rank),
        model_type=model_type,
        dropout=0.0,
    )

    adapter.inject(transformer)
    adapter.load(str(adapter_path))
    # Ensure all injected adapter tensors follow the transformer's runtime
    # placement. Without this, mixed CPU/CUDA adapter tensors can appear on
    # Z-Image and crash during sampling.
    if target_device is not None:
        with suppress(Exception):
            if isinstance(target_dtype, torch.dtype) and target_dtype.is_floating_point:
                transformer.to(device=target_device, dtype=target_dtype)
            else:
                transformer.to(device=target_device)
    adapter.merge()

    if strength is not None and strength != 1.0:
        print(f"[sampler] warning: native adapter merge uses strength=1.0 (requested {strength})")

    for param in transformer.parameters():
        param.requires_grad_(False)

    return True


def _load_assistant_lora_into_pipeline(
    *,
    pipeline: Any,
    lora_path: Path,
    adapter_name: str = "assistant",
    weight_name: str | None = None,
    strength: float | None = None,
) -> bool:
    if not _is_standard_lora_file(lora_path):
        return False

    load_fn = getattr(pipeline, "load_lora_weights", None)
    if load_fn is None:
        return False

    # Same directory/filename split as _load_assistant_lora_into_transformer.
    effective_path: str = str(lora_path)
    effective_weight_name = weight_name
    if effective_weight_name is None and lora_path.is_file():
        effective_path = str(lora_path.parent)
        effective_weight_name = lora_path.name

    load_kwargs: dict[str, Any] = {
        "adapter_name": adapter_name,
        "local_files_only": True,
    }
    if effective_weight_name is not None:
        load_kwargs["weight_name"] = effective_weight_name

    _call_with_filtered_kwargs(load_fn, effective_path, **load_kwargs)

    with suppress(Exception):
        if hasattr(pipeline, "set_adapters"):
            if strength is None:
                pipeline.set_adapters([adapter_name])
            else:
                pipeline.set_adapters([adapter_name], [float(strength)])
        elif hasattr(pipeline, "set_adapter"):
            pipeline.set_adapter(adapter_name)

    return True


class BaseSampler:
    """Base interface for Serenity sampling."""

    def __init__(self, model: Any = None, *, model_type: ModelType | None = None) -> None:
        self.model = model
        self.model_type = model_type

    def sample(self, *args, **kwargs):  # pragma: no cover - interface only
        raise NotImplementedError


class DiffusersSampler(BaseSampler):
    """Diffusers-backed sampler with offline-only model loading."""

    pipeline_candidates: tuple[str, ...] = ()
    default_steps: int = 28
    default_guidance: float = 3.5
    resolution_multiple: int = 8
    use_cpu_offload_on_cuda: bool = False
    use_sequential_cpu_offload_on_cuda: bool = False
    # "pipeline" → load_lora_weights on full pipeline (SD15, SDXL, etc.)
    # "transformer" → load directly into transformer (Flux2, ZImage)
    # "none" → skip assistant LoRA loading
    _lora_load_target: str = "pipeline"
    extra_pretrained_kwargs: dict[str, Any] = {}

    def __init__(self, model: Any = None, *, model_type: ModelType | None = None) -> None:
        super().__init__(model=model, model_type=model_type)
        self._pipeline = None
        self._pipeline_source = None
        self._pipeline_device: torch.device | None = None
        self._pipeline_dtype: torch.dtype | None = None
        self._assistant_signature: tuple[str, str | None, float | None] | None = None

    def _candidate_pipeline_names(self, **_: Any) -> tuple[str, ...]:
        return self.pipeline_candidates

    def _extract_model_source(self, override: str | None = None) -> str:
        if override:
            return _resolve_model_source(override)

        if isinstance(self.model, dict):
            for key in ("path", "model_path", "base_model", "base_model_name", "transformer_path"):
                value = self.model.get(key)
                if value:
                    return _resolve_model_source(str(value))

        for attr in ("path", "model_path", "base_model", "base_model_name", "transformer_path"):
            value = getattr(self.model, attr, None)
            if value:
                return _resolve_model_source(str(value))

        raise ValueError(
            "Missing model source. Provide `model_path=...` to `sample()` or set model.path/model_path on sampler model."
        )

    def _is_pipeline_instance(self) -> bool:
        if self.model is None:
            return False
        module_name = getattr(self.model.__class__, "__module__", "")
        return module_name.startswith("diffusers") and hasattr(self.model, "to") and callable(self.model)

    def _move_pipeline_to_device(self, pipeline: Any, device: torch.device) -> None:
        if device.type == "cuda" and self.use_cpu_offload_on_cuda:
            if self.use_sequential_cpu_offload_on_cuda and hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload(device.index or 0)
            elif hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload(device.index or 0)
            else:
                pipeline.to(device)
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            elif hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
        else:
            pipeline.to(device)

    def _load_pipeline(
        self,
        *,
        model_source: str,
        device: torch.device,
        dtype: torch.dtype,
        image: Any = None,
    ):
        errors: list[str] = []

        for class_name in self._candidate_pipeline_names(image=image):
            try:
                pipeline_cls = _load_pipeline_class(class_name)
            except Exception as exc:
                errors.append(f"{class_name}: unavailable ({exc})")
                continue

            try:
                pipeline = pipeline_cls.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    local_files_only=True,
                    **self.extra_pretrained_kwargs,
                )
                self._move_pipeline_to_device(pipeline, device)
                return pipeline
            except Exception as exc:
                if class_name == "ZImagePipeline":
                    try:
                        pipeline = _load_zimage_pipeline_without_model_index(
                            model_source=model_source,
                            dtype=dtype,
                        )
                        self._move_pipeline_to_device(pipeline, device)
                        print("[sampler] using ZImage component fallback loader (no model_index.json)")
                        return pipeline
                    except Exception as fallback_exc:
                        errors.append(f"{class_name}: load failed ({exc}); fallback failed ({fallback_exc})")
                else:
                    errors.append(f"{class_name}: load failed ({exc})")

        detail = " | ".join(errors) if errors else "No candidate pipelines configured."
        raise RuntimeError(f"Could not load a pipeline for {self.model_type}: {detail}")

    def _apply_assistant_lora(self, pipeline: Any):
        if self._lora_load_target == "none":
            return pipeline

        path, weight_name, strength = _extract_assistant_lora_settings(self.model)
        if path is None:
            return pipeline

        signature = (str(path), weight_name, strength)
        if self._assistant_signature == signature:
            return pipeline

        prefer_native_first = self.model_type in {ModelType.ZIMAGE, ModelType.Z_IMAGE}
        native_exc = None

        if prefer_native_first:
            try:
                loaded_native = _load_adapter_native(
                    pipeline=pipeline,
                    adapter_path=path,
                    strength=strength,
                    model_type=str(self.model_type.value) if self.model_type else "unknown",
                )
                if loaded_native:
                    self._assistant_signature = signature
                    print(f"[sampler] loaded adapter (native) from {path}")
                    return pipeline
            except Exception as exc:
                native_exc = exc

        # Pick diffusers loader based on target
        if self._lora_load_target == "transformer":
            load_fn = _load_assistant_lora_into_transformer
        else:
            load_fn = _load_assistant_lora_into_pipeline

        loaded = False
        diffusers_exc = None
        try:
            loaded = load_fn(
                pipeline=pipeline,
                lora_path=path,
                adapter_name="assistant",
                weight_name=weight_name,
                strength=strength,
            )
        except Exception as exc:
            diffusers_exc = exc

        if loaded:
            self._assistant_signature = signature
            print(f"[sampler] loaded assistant LoRA from {path}")
        else:
            # Diffusers skipped/failed — try native adapter system unless already attempted first.
            if not prefer_native_first:
                try:
                    loaded = _load_adapter_native(
                        pipeline=pipeline,
                        adapter_path=path,
                        strength=strength,
                        model_type=str(self.model_type.value) if self.model_type else "unknown",
                    )
                    if loaded:
                        self._assistant_signature = signature
                        print(f"[sampler] loaded adapter (native) from {path}")
                    else:
                        reason = f"diffusers={diffusers_exc}" if diffusers_exc else "diffusers=skipped(non-lora)"
                        print(f"[sampler] warning: failed to load adapter {path}: {reason}, native=unsupported")
                except Exception as exc2:
                    reason = f"diffusers={diffusers_exc}" if diffusers_exc else "diffusers=skipped(non-lora)"
                    print(f"[sampler] warning: failed to load adapter {path}: {reason}, native={exc2}")
            else:
                native_reason = f", native_first={native_exc}" if native_exc is not None else ""
                reason = f"diffusers={diffusers_exc}" if diffusers_exc else "diffusers=skipped(non-lora)"
                print(f"[sampler] warning: failed to load adapter {path}: {reason}{native_reason}")
        return pipeline

    def _ensure_pipeline(
        self,
        *,
        model_source: str,
        device: torch.device,
        dtype: torch.dtype,
        image: Any = None,
    ):
        if self._pipeline is not None:
            if self._pipeline_source == model_source and self._pipeline_dtype == dtype:
                if self._pipeline_device != device:
                    if (
                        device.type == "cuda"
                        and self.use_cpu_offload_on_cuda
                    ):
                        if (
                            self.use_sequential_cpu_offload_on_cuda
                            and hasattr(self._pipeline, "enable_sequential_cpu_offload")
                        ):
                            self._pipeline.enable_sequential_cpu_offload(device.index or 0)
                        elif hasattr(self._pipeline, "enable_model_cpu_offload"):
                            self._pipeline.enable_model_cpu_offload(device.index or 0)
                        else:
                            self._pipeline.to(device)
                    else:
                        self._pipeline.to(device)
                    self._pipeline_device = device
                return self._apply_assistant_lora(self._pipeline)

        if self._is_pipeline_instance():
            pipeline = self.model
            if (
                device.type == "cuda"
                and self.use_cpu_offload_on_cuda
            ):
                if (
                    self.use_sequential_cpu_offload_on_cuda
                    and hasattr(pipeline, "enable_sequential_cpu_offload")
                ):
                    pipeline.enable_sequential_cpu_offload(device.index or 0)
                elif hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload(device.index or 0)
                else:
                    pipeline.to(device)
            else:
                pipeline.to(device)
            self._pipeline = pipeline
            self._pipeline_source = model_source
            self._pipeline_dtype = dtype
            self._pipeline_device = device
            return self._apply_assistant_lora(pipeline)

        pipeline = self._load_pipeline(model_source=model_source, device=device, dtype=dtype, image=image)
        self._pipeline = pipeline
        self._pipeline_source = model_source
        self._pipeline_dtype = dtype
        self._pipeline_device = device
        return self._apply_assistant_lora(pipeline)

    def unload_pipeline(self) -> None:
        if self._pipeline is not None:
            with suppress(Exception):  # pragma: no cover - best effort cleanup
                self._pipeline.to("cpu")
        self._pipeline = None
        self._pipeline_source = None
        self._pipeline_device = None
        self._pipeline_dtype = None

    def _quantize_resolution(self, value: int | None) -> int | None:
        if value is None or self.resolution_multiple <= 1:
            return value
        quantized = int(round(float(value) / float(self.resolution_multiple)) * self.resolution_multiple)
        return max(self.resolution_multiple, quantized)

    def _build_call_kwargs(
        self,
        pipeline,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        image: Any,
        height: int | None,
        width: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        output_type: str,
        num_frames: int | None,
        frame_rate: float | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        params = inspect.signature(pipeline.__call__).parameters
        kwargs: dict[str, Any] = {}

        if "prompt" in params and prompt is not None:
            kwargs["prompt"] = prompt

        if negative_prompt is not None:
            if "negative_prompt" in params:
                kwargs["negative_prompt"] = negative_prompt
            elif "negative_prompt_embeds" in params and not isinstance(negative_prompt, str | list):
                kwargs["negative_prompt_embeds"] = negative_prompt

        if image is not None and "image" in params:
            kwargs["image"] = image

        if height is not None and "height" in params:
            kwargs["height"] = height
        if width is not None and "width" in params:
            kwargs["width"] = width

        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = int(num_inference_steps)
        if "guidance_scale" in params:
            kwargs["guidance_scale"] = float(guidance_scale)
        if "true_cfg_scale" in params:
            kwargs["true_cfg_scale"] = float(guidance_scale)

        if "generator" in params:
            kwargs["generator"] = generator
        if "output_type" in params:
            kwargs["output_type"] = output_type
        if "return_dict" in params:
            kwargs["return_dict"] = True

        if num_frames is not None and "num_frames" in params:
            kwargs["num_frames"] = int(num_frames)
        if frame_rate is not None and "frame_rate" in params:
            kwargs["frame_rate"] = float(frame_rate)

        kwargs.update({key: value for key, value in extra_kwargs.items() if key in params and value is not None})

        return kwargs

    @staticmethod
    def _extract_output(result: Any) -> Any:
        if hasattr(result, "images") and result.images:
            return result.images[0]
        if hasattr(result, "frames") and result.frames:
            frames = result.frames
            if isinstance(frames, list) and frames and isinstance(frames[0], list):
                return frames[0]
            return frames
        if isinstance(result, tuple | list) and result:
            return result[0]
        return result

    @staticmethod
    def _load_image(image: Any = None, image_path: str | Path | None = None) -> Any:
        if image is not None:
            return image
        if image_path is None:
            return None

        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return Image.open(path).convert("RGB")

    @staticmethod
    def _save_output(output: Any, output_path: str | Path | None, frame_rate: float = 24.0) -> None:
        if output_path is None:
            return

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(output, Image.Image):
            if not path.suffix:
                path = path.with_suffix(".png")
            output.save(path)
            return

        if isinstance(output, list) and output and all(isinstance(frame, Image.Image) for frame in output):
            if path.suffix.lower() in {".gif", ".webp"}:
                duration_ms = int(1000.0 / max(frame_rate, 1.0))
                output[0].save(
                    path,
                    save_all=True,
                    append_images=output[1:],
                    duration=duration_ms,
                    loop=0,
                )
                return
            fallback = path if path.suffix else path.with_suffix(".png")
            output[0].save(fallback)
            return

        raise ValueError(f"Unsupported sampler output type for saving: {type(output)}")

    def sample(
        self,
        prompt: str | list[str] | None = None,
        *,
        model_path: str | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = 42,
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        image: Any = None,
        image_path: str | Path | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        output_type: str = "pil",
        output_path: str | Path | None = None,
        unload: bool = False,
        **kwargs: Any,
    ) -> Any:
        resolved_device = _coerce_device(device)
        resolved_dtype = _coerce_dtype(dtype, resolved_device)
        resolved_model_source = self._extract_model_source(override=model_path)
        resolved_image = self._load_image(image=image, image_path=image_path)
        resolved_height = self._quantize_resolution(height)
        resolved_width = self._quantize_resolution(width)
        resolved_steps = int(num_inference_steps or self.default_steps)
        resolved_guidance = float(guidance_scale if guidance_scale is not None else self.default_guidance)

        generator = torch.Generator(device=resolved_device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        pipeline = self._ensure_pipeline(
            model_source=resolved_model_source,
            device=resolved_device,
            dtype=resolved_dtype,
            image=resolved_image,
        )

        call_kwargs = self._build_call_kwargs(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resolved_image,
            height=resolved_height,
            width=resolved_width,
            num_inference_steps=resolved_steps,
            guidance_scale=resolved_guidance,
            generator=generator,
            output_type=output_type,
            num_frames=num_frames,
            frame_rate=frame_rate,
            extra_kwargs=kwargs,
        )

        result = pipeline(**call_kwargs)
        output = self._extract_output(result)
        self._save_output(output, output_path, frame_rate=frame_rate or 24.0)

        if unload:
            self.unload_pipeline()

        return output


class FluxSampler(DiffusersSampler):
    pipeline_candidates = ("FluxPipeline",)
    default_steps = 28
    default_guidance = 3.5
    resolution_multiple = 64


class FluxFillSampler(DiffusersSampler):
    pipeline_candidates = ("FluxFillPipeline", "FluxPipeline")
    default_steps = 30
    default_guidance = 3.5
    resolution_multiple = 64


class Flux2Sampler(DiffusersSampler):
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 64
    use_cpu_offload_on_cuda = True
    # model_cpu_offload (not sequential) keeps params on CPU, not meta device,
    # so LoRA weights can be loaded into the transformer after offloading.
    use_sequential_cpu_offload_on_cuda = False
    _lora_load_target = "transformer"

    def _candidate_pipeline_names(self, **_: Any) -> tuple[str, ...]:
        klein_types = {
            ModelType.FLUX_2_KLEIN,
            ModelType.FLUX_2_KLEIN_4B,
            ModelType.FLUX_2_KLEIN_9B,
            ModelType.FLUX_2_KLEIN_4B_BASE,
            ModelType.FLUX_2_KLEIN_9B_BASE,
        }
        if self.model_type in klein_types:
            return ("Flux2KleinPipeline", "Flux2Pipeline")
        return ("Flux2Pipeline", "Flux2KleinPipeline")


class ZImageSampler(DiffusersSampler):
    pipeline_candidates = ("ZImagePipeline",)
    default_steps = 20
    default_guidance = 5.0
    resolution_multiple = 64
    _lora_load_target = "transformer"


class ChromaSampler(DiffusersSampler):
    pipeline_candidates = ("ChromaPipeline",)
    default_steps = 20
    default_guidance = 4.0
    resolution_multiple = 64


class SD15Sampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusionPipeline",)
    default_steps = 30
    default_guidance = 7.5
    resolution_multiple = 8
    extra_pretrained_kwargs = {
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }


class SDInpaintingSampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusionInpaintPipeline",)
    default_steps = 30
    default_guidance = 7.5
    resolution_multiple = 8


class SDDepthSampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusionDepth2ImgPipeline",)
    default_steps = 30
    default_guidance = 7.5
    resolution_multiple = 8


class SDXLSampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusionXLPipeline",)
    default_steps = 30
    default_guidance = 5.0
    resolution_multiple = 8


class SDXLInpaintingSampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusionXLInpaintPipeline",)
    default_steps = 30
    default_guidance = 5.0
    resolution_multiple = 8


class SD3Sampler(DiffusersSampler):
    pipeline_candidates = ("StableDiffusion3Pipeline",)
    default_steps = 28
    default_guidance = 5.0
    resolution_multiple = 16


class WuerstchenSampler(DiffusersSampler):
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 32

    def _candidate_pipeline_names(self, **_: Any) -> tuple[str, ...]:
        if self.model_type == ModelType.STABLE_CASCADE_1:
            return ("StableCascadeCombinedPipeline", "WuerstchenCombinedPipeline")
        return ("WuerstchenCombinedPipeline", "StableCascadeCombinedPipeline")


class PixArtSampler(DiffusersSampler):
    default_steps = 28
    default_guidance = 4.5
    resolution_multiple = 32

    def _candidate_pipeline_names(self, **_: Any) -> tuple[str, ...]:
        if self.model_type == ModelType.PIXART_SIGMA:
            return ("PixArtSigmaPipeline", "PixArtAlphaPipeline")
        return ("PixArtAlphaPipeline", "PixArtSigmaPipeline")


class SanaSampler(DiffusersSampler):
    pipeline_candidates = ("SanaPipeline",)
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 32


class HunyuanVideoSampler(DiffusersSampler):
    pipeline_candidates = ("HunyuanVideoPipeline",)
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 32


class HiDreamSampler(DiffusersSampler):
    pipeline_candidates = ("HiDreamImagePipeline",)
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 32


class LTX2Sampler(DiffusersSampler):
    pipeline_candidates = ("LTX2Pipeline", "LTXPipeline")
    default_steps = 30
    default_guidance = 4.0
    resolution_multiple = 32


class QwenSampler(DiffusersSampler):
    pipeline_candidates = ("QwenImagePipeline",)
    default_steps = 20
    default_guidance = 4.0
    resolution_multiple = 64
    use_cpu_offload_on_cuda = True
    use_sequential_cpu_offload_on_cuda = True


class QwenImageEditSampler(DiffusersSampler):
    default_steps = 20
    default_guidance = 4.0
    resolution_multiple = 64
    use_cpu_offload_on_cuda = True
    use_sequential_cpu_offload_on_cuda = True

    def _candidate_pipeline_names(self, **_: Any) -> tuple[str, ...]:
        return ("QwenImageImg2ImgPipeline", "QwenImageEditPipeline", "QwenImagePipeline")


def create_sampler(model_type: ModelType | str, model: Any = None) -> BaseSampler:
    resolved_type = _coerce_model_type(model_type)

    if resolved_type in {ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL}:
        return FluxSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.FLUX_FILL_DEV:
        return FluxFillSampler(model, model_type=resolved_type)
    if resolved_type in {
        ModelType.FLUX_2,
        ModelType.FLUX_2_DEV,
        ModelType.FLUX_2_KLEIN,
        ModelType.FLUX_2_KLEIN_4B,
        ModelType.FLUX_2_KLEIN_9B,
        ModelType.FLUX_2_KLEIN_4B_BASE,
        ModelType.FLUX_2_KLEIN_9B_BASE,
    }:
        return Flux2Sampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.ZIMAGE, ModelType.Z_IMAGE}:
        return ZImageSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.CHROMA_1:
        return ChromaSampler(model, model_type=resolved_type)
    if resolved_type in {
        ModelType.SD15,
        ModelType.SD20,
        ModelType.SD20_BASE,
        ModelType.SD21,
        ModelType.SD21_BASE,
    }:
        return SD15Sampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.SD15_INPAINTING, ModelType.SD20_INPAINTING}:
        return SDInpaintingSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.SD20_DEPTH:
        return SDDepthSampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.SDXL, ModelType.SDXL_10_BASE}:
        return SDXLSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.SDXL_INPAINTING:
        return SDXLInpaintingSampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.SD3, ModelType.SD35}:
        return SD3Sampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.WUERSTCHEN_2, ModelType.STABLE_CASCADE_1}:
        return WuerstchenSampler(model, model_type=resolved_type)
    if resolved_type in {ModelType.PIXART_ALPHA, ModelType.PIXART_SIGMA}:
        return PixArtSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.SANA:
        return SanaSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.HUNYUAN_VIDEO:
        return HunyuanVideoSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.HI_DREAM_FULL:
        return HiDreamSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.LTX2:
        return LTX2Sampler(model, model_type=resolved_type)
    if resolved_type == ModelType.QWEN:
        return QwenSampler(model, model_type=resolved_type)
    if resolved_type == ModelType.QWEN_IMAGE_EDIT:
        return QwenImageEditSampler(model, model_type=resolved_type)

    return BaseSampler(model, model_type=resolved_type)
