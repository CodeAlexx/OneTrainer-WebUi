"""LTX2 model adapter and utility helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from serenity.core.interfaces import ModelType
from serenity.models.base import BaseModelImpl

import torch

_PROMPT_MAX_LENGTH = 226
_PIPELINE_CLASS_CANDIDATES = ("LTX2Pipeline", "LTXPipeline")
_REQUIRED_SUBFOLDERS: dict[str, tuple[str, ...]] = {
    "scheduler": ("scheduler_config.json", "config.json"),
    "tokenizer": ("tokenizer_config.json", "tokenizer.json"),
    "text_encoder": ("config.json",),
    "transformer": ("config.json",),
    "vae": ("config.json",),
}
_WEIGHT_MARKERS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".msgpack",
)
_DEFAULT_COMPONENT_FILES = {
    "transformer": "diffusion_pytorch_model.safetensors",
    "vae": "diffusion_pytorch_model.safetensors",
}
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "spiece.model")
_COMPONENT_ALIAS_KEYS: dict[str, tuple[str, ...]] = {
    "transformer": ("transformer_path", "transformer"),
    "vae": ("vae_path", "vae"),
    "text_encoder": ("text_encoder_path", "text_encoder", "clip_path"),
    "tokenizer": ("tokenizer_path", "tokenizer"),
    "scheduler": ("scheduler_path", "scheduler_dir"),
}
_TEMPLATE_ALIAS_KEYS = ("template_path", "ltx_template_path", "template")


def _resolve_snapshot_from_repo_dir(repo_dir: Path) -> Path | None:
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            return snapshot

    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
    if snapshots:
        return snapshots[-1]

    return None


def _is_weight_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _WEIGHT_MARKERS


def _is_valid_ltx_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        _validate_ltx_model_path(str(path))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _resolve_component_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.exists():
        return path
    return None


def _find_ltx_template_root(explicit_template: Any = None) -> Path | None:
    candidates: list[Path] = []
    explicit = _resolve_component_path(explicit_template)
    if explicit is not None:
        candidates.append(explicit)

    env_template = _resolve_component_path(os.environ.get("SERENITY_LTX_TEMPLATE_PATH"))
    if env_template is not None:
        candidates.append(env_template)

    local_models_root = Path.home() / "models"
    if local_models_root.exists():
        candidates.extend(sorted(local_models_root.glob("*LTX*")))

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for repo_dir in sorted(hf_cache.glob("models--*LTX*")):
            snapshot = _resolve_snapshot_from_repo_dir(repo_dir)
            if snapshot is not None:
                candidates.append(snapshot)

    for candidate in candidates:
        if _is_valid_ltx_root(candidate):
            return candidate

    return None


def _extract_component_overrides(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    component_paths: dict[str, Any] = {}
    raw_component_paths = kwargs.get("ltx_component_paths") or kwargs.get("component_paths")
    if isinstance(raw_component_paths, dict):
        component_paths.update(raw_component_paths)

    for component_name, aliases in _COMPONENT_ALIAS_KEYS.items():
        if component_paths.get(component_name):
            continue
        for alias in aliases:
            value = kwargs.get(alias)
            if value:
                component_paths[component_name] = value
                break

    template_path = kwargs.get("ltx_template_path")
    if not template_path:
        for alias in _TEMPLATE_ALIAS_KEYS:
            value = kwargs.get(alias)
            if value:
                template_path = value
                break

    return component_paths, template_path


def _infer_split_component_paths(primary_model_path: Path) -> dict[str, Path]:
    inferred: dict[str, Path] = {}
    search_roots: list[Path] = []

    if primary_model_path.is_file():
        search_roots.append(primary_model_path.parent)
        search_roots.append(primary_model_path.parent.parent)
    else:
        search_roots.append(primary_model_path)
        search_roots.append(primary_model_path.parent)

    candidate_dirs: dict[str, tuple[str, ...]] = {
        "vae": ("VAE", "vae"),
        "text_encoder": ("text_encoder", "text-encoder", "clip"),
        "tokenizer": ("tokenizer", "clip", "text_encoder"),
        "scheduler": ("scheduler", "schedulers"),
    }

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(root)
        for component_name, folder_names in candidate_dirs.items():
            if component_name in inferred:
                continue
            for folder_name in folder_names:
                candidate = root / folder_name
                if candidate.exists():
                    inferred[component_name] = candidate
                    break

    return inferred


def _try_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
        return
    except OSError:
        pass

    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _resolve_vae_source(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "diffusion_pytorch_model.safetensors"
        if direct.exists():
            return direct
        for candidate_name in ("ae.safetensors", "vae.safetensors", "ltx_vae.safetensors"):
            candidate = path / candidate_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve LTX VAE source from {path}")


def _resolve_text_encoder_source(path: Path) -> Path:
    if path.is_dir() and (path / "config.json").exists() and (
        _component_has_weights(path) or (path / "model.safetensors.index.json").exists()
    ):
        return path

    if path.is_dir():
        preferred_names = ("gemma-3-12b-it", "gemma_combined", "text_encoder")
        for name in preferred_names:
            candidate = path / name
            if candidate.is_dir() and (candidate / "config.json").exists():
                return candidate

        for candidate in sorted(path.iterdir()):
            if candidate.is_dir() and (candidate / "config.json").exists():
                return candidate

    raise FileNotFoundError(f"Could not resolve LTX text encoder source from {path}")


def _resolve_tokenizer_source(path: Path, text_encoder_source: Path) -> Path:
    if path.is_file() and path.name in _TOKENIZER_FILES:
        return path.parent

    if path.is_dir() and any((path / file_name).exists() for file_name in _TOKENIZER_FILES):
        return path

    if path.is_dir():
        for candidate in sorted(path.iterdir()):
            if candidate.is_dir() and any((candidate / file_name).exists() for file_name in _TOKENIZER_FILES):
                return candidate

    if text_encoder_source.is_dir() and any((text_encoder_source / file_name).exists() for file_name in _TOKENIZER_FILES):
        return text_encoder_source

    raise FileNotFoundError(f"Could not resolve LTX tokenizer source from {path}")


def _resolve_scheduler_source(path: Path) -> Path:
    if path.is_file() and path.name in {"scheduler_config.json", "config.json"}:
        return path.parent
    if path.is_dir() and ((path / "scheduler_config.json").exists() or (path / "config.json").exists()):
        return path
    raise FileNotFoundError(f"Could not resolve LTX scheduler source from {path}")


def _build_ltx_bundle(
    *,
    primary_model_path: Path,
    template_root: Path,
    component_paths: dict[str, Any],
    dtype: torch.dtype,
):
    transformer_override = _resolve_component_path(component_paths.get("transformer"))
    vae_override = _resolve_component_path(component_paths.get("vae"))
    text_encoder_override = _resolve_component_path(component_paths.get("text_encoder"))
    tokenizer_override = _resolve_component_path(component_paths.get("tokenizer"))
    scheduler_override = _resolve_component_path(component_paths.get("scheduler"))

    transformer_source = transformer_override or primary_model_path
    if not transformer_source.exists():
        raise FileNotFoundError(f"LTX transformer source not found: {transformer_source}")

    vae_source_raw = vae_override or (template_root / "vae")
    text_encoder_source_raw = text_encoder_override or (template_root / "text_encoder")
    tokenizer_source_raw = tokenizer_override or text_encoder_source_raw
    scheduler_source_raw = scheduler_override or (template_root / "scheduler")

    vae_source = _resolve_vae_source(vae_source_raw)
    text_encoder_source = _resolve_text_encoder_source(text_encoder_source_raw)
    tokenizer_source = _resolve_tokenizer_source(tokenizer_source_raw, text_encoder_source)
    scheduler_source = _resolve_scheduler_source(scheduler_source_raw)

    with tempfile.TemporaryDirectory(prefix="serenity_ltx_bundle_") as temp_dir:
        bundle_root = Path(temp_dir)
        _try_link_or_copy(template_root / "model_index.json", bundle_root / "model_index.json")

        _try_link_or_copy(scheduler_source, bundle_root / "scheduler")
        _try_link_or_copy(text_encoder_source, bundle_root / "text_encoder")
        _try_link_or_copy(tokenizer_source, bundle_root / "tokenizer")

        vae_target = bundle_root / "vae"
        vae_target.mkdir(parents=True, exist_ok=True)
        _try_link_or_copy(template_root / "vae" / "config.json", vae_target / "config.json")
        _try_link_or_copy(vae_source, vae_target / _DEFAULT_COMPONENT_FILES["vae"])

        transformer_target = bundle_root / "transformer"
        if transformer_source.is_dir():
            _try_link_or_copy(transformer_source, transformer_target)
        else:
            transformer_target.mkdir(parents=True, exist_ok=True)
            _try_link_or_copy(template_root / "transformer" / "config.json", transformer_target / "config.json")
            _try_link_or_copy(transformer_source, transformer_target / _DEFAULT_COMPONENT_FILES["transformer"])

        return _load_pipeline_from_candidates(str(bundle_root), dtype)


def _validate_ltx_model_path(model_path: str) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"LTX model path does not exist: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"LTX model path must be a directory: {root}")

    missing_subfolders: list[str] = []
    invalid_components: list[str] = []
    for subfolder, required_files in _REQUIRED_SUBFOLDERS.items():
        component_path = root / subfolder
        if not component_path.exists():
            missing_subfolders.append(subfolder)
            continue
        if not any((component_path / required).exists() for required in required_files):
            invalid_components.append(subfolder)

    if missing_subfolders:
        raise FileNotFoundError(
            f"LTX model at {root} is missing required components: {', '.join(sorted(missing_subfolders))}"
        )
    if invalid_components:
        raise FileNotFoundError(
            f"LTX model at {root} has invalid components (missing config files): "
            f"{', '.join(sorted(invalid_components))}"
        )

    return root


def _component_has_weights(component_dir: Path) -> bool:
    return any(entry.is_file() and entry.name.endswith(_WEIGHT_MARKERS) for entry in component_dir.iterdir())


def _load_pipeline_from_candidates(
    model_path: str,
    dtype: torch.dtype,
):
    import diffusers

    load_errors: list[str] = []
    for class_name in _PIPELINE_CLASS_CANDIDATES:
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            load_errors.append(f"{class_name}: unavailable")
            continue
        try:
            return pipeline_cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            load_errors.append(f"{class_name}: {exc}")

    details = " | ".join(load_errors) if load_errors else "no pipeline candidates"
    raise RuntimeError(f"Could not load LTX pipeline from {model_path}: {details}")


def normalize_ltx2_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float,
    reverse: bool = False,
) -> torch.Tensor:
    """Normalize or denormalize LTX latents."""

    mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)

    if reverse:
        return (latents / scaling_factor) * std + mean

    return (latents - mean) * scaling_factor / std


def pack_ltx2_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack latents from [B, C, T, H, W] to [B, S, C*ps*ps*pt]."""

    b, c, t, h, w = latents.shape
    if t % patch_size_t != 0 or h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Latent dimensions must be divisible by patch sizes")

    t2 = t // patch_size_t
    h2 = h // patch_size
    w2 = w // patch_size

    latents = latents.view(b, c, t2, patch_size_t, h2, patch_size, w2, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    latents = latents.reshape(b, t2 * h2 * w2, c * patch_size_t * patch_size * patch_size)
    return latents


def unpack_ltx2_latents(
    packed: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """Unpack latents from [B, S, C*ps*ps*pt] to [B, C, T, H, W]."""

    b, _seq_len, hidden = packed.shape
    t2, h2, w2 = frames, height, width

    c = hidden // (patch_size_t * patch_size * patch_size)
    if c <= 0:
        raise ValueError("Invalid hidden dimension for unpacking")

    packed = packed.view(b, t2, h2, w2, c, patch_size_t, patch_size, patch_size)
    packed = packed.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = packed.reshape(b, c, t2 * patch_size_t, h2 * patch_size, w2 * patch_size)
    return latents


def adjust_video_frames(frame_count: int) -> int:
    """Adjust frame count to satisfy LTX constraint (frames % 8 == 1)."""

    if frame_count <= 1:
        return 1
    remainder = (frame_count - 1) % 8
    return frame_count - remainder


class LTX2Model(BaseModelImpl):
    """Native LTX behavior used by Serenity training paths."""

    family = "ltx2"
    resolution_multiple = 32
    train_module_attr = "transformer"
    flow_objective = True

    def __init__(self) -> None:
        super().__init__(model_type=ModelType.LTX2)

    def load_pipeline(
        self,
        model_path: str,
        dtype: torch.dtype,
        train_device: torch.device,
        **kwargs: Any,
    ):
        del train_device

        component_paths, template_override = _extract_component_overrides(kwargs)
        requested_path = Path(model_path).expanduser()

        model_root: Path | None = None
        if requested_path.exists() and requested_path.is_dir():
            with suppress(Exception):
                model_root = _validate_ltx_model_path(str(requested_path))

        if model_root is not None and _component_has_weights(model_root / "transformer"):
            pipeline = _load_pipeline_from_candidates(str(model_root), dtype)
            pipeline.to("cpu")
            return pipeline

        transformer_override = _resolve_component_path(component_paths.get("transformer"))
        primary_model_path = transformer_override or requested_path
        if not primary_model_path.exists():
            raise FileNotFoundError(f"LTX model path does not exist: {primary_model_path}")

        if not transformer_override and primary_model_path.is_dir():
            nested_transformer = primary_model_path / "transformer"
            if nested_transformer.exists() and nested_transformer.is_dir():
                primary_model_path = nested_transformer

        for component_name, inferred_path in _infer_split_component_paths(primary_model_path).items():
            component_paths.setdefault(component_name, str(inferred_path))

        template_hint: Any = template_override
        if template_hint is None and requested_path.is_dir():
            template_hint = requested_path
        template_root = _find_ltx_template_root(template_hint)
        if template_root is None:
            raise FileNotFoundError(
                "Could not locate a local LTX template snapshot. "
                "Set model.ltx_template_path/template_path or SERENITY_LTX_TEMPLATE_PATH."
            )

        pipeline = _build_ltx_bundle(
            primary_model_path=primary_model_path,
            template_root=template_root,
            component_paths=component_paths,
            dtype=dtype,
        )
        pipeline.to("cpu")
        return pipeline

    def get_train_module(self, pipeline: Any) -> torch.nn.Module:
        return pipeline.transformer

    def encode_latents(self, pipeline: Any, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)

        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)

        latents_mean_cfg = getattr(pipeline.vae.config, "latents_mean", None)
        latents_std_cfg = getattr(pipeline.vae.config, "latents_std", None)
        scaling_factor = float(getattr(pipeline.vae.config, "scaling_factor", 1.0))
        if latents_mean_cfg is None or latents_std_cfg is None:
            return latents * scaling_factor

        latents_mean = torch.tensor(latents_mean_cfg, device=latents.device, dtype=latents.dtype)
        latents_std = torch.tensor(latents_std_cfg, device=latents.device, dtype=latents.dtype)
        return normalize_ltx2_latents(
            latents,
            latents_mean=latents_mean,
            latents_std=latents_std,
            scaling_factor=scaling_factor,
            reverse=False,
        )

    def encode_prompt_features(
        self,
        pipeline: Any,
        prompt: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        tokenizer = getattr(pipeline, "tokenizer", None)
        tokenizer_limit = int(getattr(tokenizer, "model_max_length", _PROMPT_MAX_LENGTH) or _PROMPT_MAX_LENGTH)
        max_sequence_length = max(1, min(_PROMPT_MAX_LENGTH, tokenizer_limit))

        encoded = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        values = list(encoded) if isinstance(encoded, tuple | list) else [encoded]

        prompt_embeds = next(
            (
                value
                for value in values
                if torch.is_tensor(value) and value.is_floating_point() and value.dim() >= 2
            ),
            None,
        )
        if prompt_embeds is None:
            raise RuntimeError("LTX encode_prompt did not return prompt embeddings.")

        prompt_mask: torch.Tensor | None = None
        for value in values:
            if not torch.is_tensor(value) or value is prompt_embeds:
                continue
            if value.dtype in {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                prompt_mask = value
                break

        return prompt_embeds, None, prompt_mask

    def cache_prompt_device(self, pipeline: Any, train_device: torch.device) -> torch.device:
        del pipeline
        return train_device

    def move_text_encoders_to_device(self, pipeline: Any, device: torch.device) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        text_encoder.to(device)

    def offload_text_encoders(self, pipeline: Any) -> None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return
        with suppress(Exception):
            text_encoder.to("cpu")

    def get_patch_sizes(self, pipeline: Any) -> tuple[int, int]:
        transformer_config = getattr(getattr(pipeline, "transformer", None), "config", None)
        patch_size = int(getattr(transformer_config, "patch_size", 1) or 1)
        patch_size_t = int(getattr(transformer_config, "patch_size_t", 1) or 1)
        return patch_size, patch_size_t

    @staticmethod
    def resolve_rope_interpolation_scale(pipeline: Any) -> tuple[float, float, float] | torch.Tensor | None:
        transformer_config = getattr(getattr(pipeline, "transformer", None), "config", None)
        scale = getattr(transformer_config, "rope_interpolation_scale", None)
        if scale is None:
            return None
        if torch.is_tensor(scale):
            return scale
        if isinstance(scale, tuple | list) and len(scale) == 3:
            return tuple(float(value) for value in scale)
        return None

    @staticmethod
    def pack_latents(
        latents: torch.Tensor,
        *,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        return pack_ltx2_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)

    @staticmethod
    def unpack_latents(
        packed: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        return unpack_ltx2_latents(
            packed,
            frames=frames,
            height=height,
            width=width,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
        )

    @staticmethod
    def validate_frame_count(frame_count: int) -> int:
        return adjust_video_frames(frame_count)

    @staticmethod
    def adjust_video_frames(frame_count: int) -> int:
        return adjust_video_frames(frame_count)

    @staticmethod
    def get_valid_frame_counts(max_frames: int) -> list[int]:
        return list(range(1, max_frames + 1, 8))
