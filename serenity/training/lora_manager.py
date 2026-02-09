"""Native LoRA/DoRA adapter management for Serenity.

This manager uses Serenity's native LoRA implementation (no PEFT dependency),
wrapping nn.Linear layers with low-rank decomposition matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from serenity.training.adapter_utils import (
    normalize_model_type as _normalize_model_type,
    coerce_dtype as _coerce_dtype,
    resolve_target_module as _resolve_target_module_impl,
    dedupe as _dedupe_impl,
)
from serenity.training.lycoris_manager import DEFAULT_TARGETS

logger = logging.getLogger(__name__)

# Model families that use a UNet (prefix "unet.") rather than a transformer.
_UNET_FAMILIES: frozenset[str] = frozenset({
    "sd15", "sd15_inpainting",
    "sd20", "sd20_base", "sd20_inpainting", "sd20_depth",
    "sd21", "sd21_base",
    "sdxl", "sdxl_10_base", "sdxl_inpainting",
})


def _to_diffusers_state_dict(
    state: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    """Convert native LoRA keys to diffusers/PEFT format.

    Native format:  ``transformer_blocks.0.attn.to_k.lora_down.weight``
    Diffusers:      ``transformer.transformer_blocks.0.attn.to_k.lora_A.weight``
    """
    prefix = "unet" if model_type in _UNET_FAMILIES else "transformer"
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
        out[f"{prefix}.{new_key}"] = value
    return out


def _from_diffusers_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert diffusers/PEFT keys back to native LoRA format.

    Strips ``transformer.`` / ``unet.`` prefix and renames lora_A→lora_down,
    lora_B→lora_up so that ``load_lora_state_dict`` can match internal modules.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for pfx in ("transformer.", "unet."):
            if new_key.startswith(pfx):
                new_key = new_key[len(pfx):]
                break
        new_key = new_key.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up.")
        out[new_key] = value
    return out


@dataclass
class AdapterConfig:
    rank: int
    alpha: float = 1.0
    target_modules: list[str] = field(default_factory=list)
    dropout: float = 0.0
    rs_lora: bool = False
    multiplier: float = 1.0
    adapter_name: str = "serenity_train"


class LoRAManager:
    """Manage native LoRA attachment using Serenity's own LoRALinear layers."""

    def __init__(self, config: AdapterConfig, model_type: str) -> None:
        self.config = config
        self.model_type = _normalize_model_type(model_type)
        self.target_modules = (
            list(config.target_modules)
            if config.target_modules
            else list(DEFAULT_TARGETS.get(self.model_type, []))
        )
        self.adapter_name = str(config.adapter_name or "serenity_train")
        self._target_module: nn.Module | None = None
        self._lora_linears: list[Any] | None = None
        self._trainable_params: list[nn.Parameter] = []

    @staticmethod
    def _resolve_target_module(model_or_pipeline: Any) -> nn.Module:
        return _resolve_target_module_impl(model_or_pipeline)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return _dedupe_impl(values, strip=True)

    def _get_apply_fn(self):
        """Return the apply function for this manager type."""
        from serenity.training.adapters.lora import apply_lora
        return apply_lora

    def _refresh_trainable_params(self) -> None:
        if self._target_module is None:
            self._trainable_params = []
            return

        from serenity.training.adapters.lora import get_lora_params

        self._trainable_params = get_lora_params(self._target_module)

    def _set_multiplier(self, multiplier: float | None = None) -> None:
        """No-op for native LoRA. Scaling is handled by alpha/rank ratio."""
        _ = multiplier
        logger.debug(
            "Multiplier adjustment is a no-op for native LoRA; "
            "scaling is determined by alpha/rank (%.4f).",
            self.config.alpha / max(self.config.rank, 1),
        )

    def apply(self, model_or_pipeline: Any) -> Any:
        target_module = self._resolve_target_module(model_or_pipeline)

        if (
            self._target_module is target_module
            and self._lora_linears is not None
            and len(self._lora_linears) > 0
        ):
            self._refresh_trainable_params()
            return target_module

        targets = self._dedupe(list(self.target_modules))
        if not targets:
            targets = ["to_q", "to_k", "to_v"]

        apply_fn = self._get_apply_fn()
        self._lora_linears = apply_fn(
            target_module,
            rank=max(1, int(self.config.rank)),
            alpha=float(self.config.alpha),
            target_modules=targets,
            dropout=float(self.config.dropout),
        )

        self._target_module = target_module
        self._refresh_trainable_params()
        logger.info(
            "Applied native LoRA: rank=%d alpha=%.1f targets=%d layers=%d",
            self.config.rank,
            self.config.alpha,
            len(targets),
            len(self._lora_linears),
        )
        return target_module

    def is_attached(self) -> bool:
        return self._lora_linears is not None and len(self._lora_linears) > 0

    def set_multiplier(self, multiplier: float) -> None:
        self._set_multiplier(multiplier)

    def prepare_optimizer_params(self, lr: float | None = None) -> list[dict[str, Any]]:
        if self._target_module is None:
            raise RuntimeError("LoRA adapter is not attached. Call apply() before building optimizer params.")
        self._refresh_trainable_params()
        group: dict[str, Any] = {"params": self._trainable_params}
        if lr is not None:
            group["lr"] = float(lr)
        return [group]

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._target_module is None:
            return {}

        from serenity.training.adapters.lora import extract_lora_state_dict

        return extract_lora_state_dict(self._target_module)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> Any:
        if self._target_module is None:
            raise RuntimeError("LoRA adapter is not attached. Call apply() before loading adapter weights.")

        # Accept both diffusers format (lora_A/lora_B with prefix) and native
        # format (lora_down/lora_up without prefix).
        first_key = next(iter(state_dict), "")
        if first_key.startswith(("transformer.", "unet.")) or ".lora_A." in first_key or ".lora_B." in first_key:
            state_dict = _from_diffusers_state_dict(state_dict)

        from serenity.training.adapters.lora import load_lora_state_dict

        load_lora_state_dict(self._target_module, state_dict, strict=strict)
        self._refresh_trainable_params()

    def load_weights(self, input_path: str | Path) -> None:
        if self._target_module is None:
            raise RuntimeError("LoRA adapter is not attached. Call apply() before loading adapter weights.")

        path = Path(input_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

        # Accept both diffusers format and native format.
        first_key = next(iter(state_dict), "")
        if first_key.startswith(("transformer.", "unet.")) or ".lora_A." in first_key or ".lora_B." in first_key:
            state_dict = _from_diffusers_state_dict(state_dict)

        from serenity.training.adapters.lora import load_lora_state_dict

        load_lora_state_dict(self._target_module, state_dict, strict=False)
        self._refresh_trainable_params()

    def merge_to(self, weight: float = 1.0, *, precise: bool = False) -> None:
        if self._target_module is None:
            raise RuntimeError("LoRA adapter is not attached. Call apply() before merge_to().")

        from serenity.training.adapters.lora import merge_lora

        merge_lora(self._target_module)

    def restore(self) -> None:
        if self._target_module is None:
            return

        from serenity.training.adapters.lora import unmerge_lora

        unmerge_lora(self._target_module)

    def save_weights(
        self,
        output_path: str | Path,
        dtype: torch.dtype | str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        if self._target_module is None:
            raise RuntimeError("LoRA adapter is not attached. Call apply() before save_weights().")

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        target_dtype = _coerce_dtype(dtype)
        state = self.state_dict()

        # Convert to diffusers/PEFT format so saved LoRAs are loadable by
        # diffusers pipelines, ComfyUI, and Serenity's own sampler.
        state = _to_diffusers_state_dict(state, self.model_type)

        if target_dtype is not None:
            state = {
                key: value.to(dtype=target_dtype) if torch.is_tensor(value) else value
                for key, value in state.items()
            }

        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import save_file

            save_file(state, str(path), metadata=metadata or {})
        else:
            torch.save(state, str(path))
        return path


class DoRAManager(LoRAManager):
    """Manage native DoRA attachment. Identical to LoRAManager but uses DoRALinear."""

    def _get_apply_fn(self):
        """Return the DoRA apply function."""
        from serenity.training.adapters.dora import apply_dora
        return apply_dora

    def merge_to(self, weight: float = 1.0, *, precise: bool = False) -> None:
        raise NotImplementedError(
            "DoRA does not support merge/unmerge; use LoRA if merge is needed."
        )

    def restore(self) -> None:
        raise NotImplementedError(
            "DoRA does not support merge/unmerge; use LoRA if merge is needed."
        )


__all__ = [
    "AdapterConfig",
    "LoRAManager",
    "DoRAManager",
    "_to_diffusers_state_dict",
    "_from_diffusers_state_dict",
]
