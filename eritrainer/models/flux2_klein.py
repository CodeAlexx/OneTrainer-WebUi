"""FLUX.2 Klein model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from eritrainer.core.interfaces import ModelType
from eritrainer.models.base import BaseModelImpl
from eritrainer.models.flux2_transformer import Flux2Transformer2DModel


FLUX2_KLEIN_CONFIG = {
    "klein-4b": {
        "double_blocks": 5,
        "single_blocks": 20,
        "total_blocks": 25,
        "max_swappable": 24,
        "joint_attention_dim": 7680,
        "text_encoder_layers": (9, 18, 27),
        "hidden_size": 2560,
        "default_steps_distilled": 4,
        "default_steps_base": 50,
    },
    "klein-9b": {
        "double_blocks": 8,
        "single_blocks": 24,
        "total_blocks": 32,
        "max_swappable": 31,
        "joint_attention_dim": 12288,
        "text_encoder_layers": (9, 18, 27),
        "hidden_size": 4096,
        "default_steps_distilled": 4,
        "default_steps_base": 50,
    },
}


@dataclass
class Flux2KleinModel(BaseModelImpl):
    transformer: Optional[torch.nn.Module] = None
    vae: Optional[torch.nn.Module] = None
    text_encoder: Optional[torch.nn.Module] = None
    tokenizer: Optional[object] = None

    def __init__(self, model_type: ModelType = ModelType.FLUX_2_KLEIN_4B) -> None:
        super().__init__(model_type=model_type)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)


class Flux2KleinModelLoader:
    """Load FLUX.2 Klein models without network access."""

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path)
        if path.exists():
            return path

        # Attempt to resolve HF cache layout
        if "/" in model_path:
            org, name = model_path.split("/", 1)
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            candidate = cache_root / f"models--{org}--{name}"
            if candidate.exists():
                refs_main = candidate / "refs" / "main"
                if refs_main.exists():
                    revision = refs_main.read_text().strip()
                    snapshot = candidate / "snapshots" / revision
                    if snapshot.exists():
                        return snapshot
                # Fallback to first snapshot
                snapshots = list((candidate / "snapshots").glob("*") if (candidate / "snapshots").exists() else [])
                if snapshots:
                    return snapshots[0]

        raise FileNotFoundError(f"Model path not found: {model_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        model_type: Optional[ModelType] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        use_flux2_transformer: bool = True,
    ) -> Flux2KleinModel:
        resolved = cls._resolve_model_path(model_path)
        model_type = model_type or ModelType.FLUX_2_KLEIN_4B

        model = Flux2KleinModel(model_type=model_type)
        model.transformer = cls._load_transformer(resolved / "transformer", dtype, device, use_flux2_transformer)
        return model

    @classmethod
    def _load_transformer(
        cls,
        transformer_path: Path,
        dtype: torch.dtype,
        device: str,
        use_flux2_transformer: bool,
    ) -> Optional[torch.nn.Module]:
        if not transformer_path.exists():
            return None

        if not use_flux2_transformer:
            try:
                from diffusers import FluxTransformer2DModel
            except Exception as exc:  # pragma: no cover
                raise ImportError("diffusers is required to load FluxTransformer2DModel") from exc

            return FluxTransformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)

        return Flux2Transformer2DModel.from_pretrained_klein(
            str(transformer_path),
            torch_dtype=dtype,
            device=device,
        )

    @classmethod
    def load_transformer_only(
        cls,
        transformer_path: str,
        model_type: Optional[ModelType] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_flux2_transformer: bool = True,
    ) -> Flux2KleinModel:
        model = Flux2KleinModel(model_type=model_type or ModelType.FLUX_2_KLEIN_4B)
        model.transformer = cls._load_transformer(Path(transformer_path), dtype, "cpu", use_flux2_transformer)
        return model


class Flux2KleinSampler:
    """Sampler stub for Flux2 Klein models."""

    def __init__(self, model: Flux2KleinModel, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

    def default_steps(self) -> int:
        variant = "klein-4b"
        if self.model.model_type in {ModelType.FLUX_2_KLEIN_9B, ModelType.FLUX_2_KLEIN_9B_BASE}:
            variant = "klein-9b"
        config = FLUX2_KLEIN_CONFIG[variant]
        if self.model.model_type in {ModelType.FLUX_2_KLEIN_4B_BASE, ModelType.FLUX_2_KLEIN_9B_BASE}:
            return config["default_steps_base"]
        return config["default_steps_distilled"]

    def timesteps(self, steps: Optional[int] = None) -> torch.Tensor:
        steps = steps or self.default_steps()
        return torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=self.dtype)
