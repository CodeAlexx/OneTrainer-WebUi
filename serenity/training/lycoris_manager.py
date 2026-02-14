"""Native LyCORIS adapter management for Serenity."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
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

try:  # pragma: no cover - optional dependency guard
    from lycoris import create_lycoris
except ImportError:  # pragma: no cover - optional dependency guard
    create_lycoris = None


logger = logging.getLogger(__name__)


class AdapterType(str, Enum):
    LORA = "lora"
    LOCON = "locon"
    LOHA = "loha"
    LOKR = "lokr"
    IA3 = "ia3"
    OFT = "oft"
    BOFT = "boft"
    DIAG_OFT = "diag_oft"
    GLORA = "glora"
    DYLORA = "dylora"
    DORA = "dora"
    FULL = "full"


_ALGO_MAP: dict[AdapterType, str] = {
    AdapterType.LORA: "lora",
    AdapterType.LOCON: "locon",
    AdapterType.LOHA: "loha",
    AdapterType.LOKR: "lokr",
    AdapterType.OFT: "diag-oft",
    AdapterType.BOFT: "boft",
    AdapterType.DIAG_OFT: "diag-oft",
    AdapterType.GLORA: "glora",
    AdapterType.DYLORA: "dylora",
    AdapterType.DORA: "lora",
    AdapterType.FULL: "full",
    # IA3 is not registered in create_lycoris() for this lycoris build; use LoRA path.
    AdapterType.IA3: "lora",
}


# _MODEL_ALIASES, _normalize_model_type, _coerce_dtype — from adapter_utils

_LYCORIS_DISABLED_MODEL_TYPES: set[str] = {"ltx2"}


DEFAULT_TARGETS: dict[str, list[str]] = {
    "zimage": ["attention.to_q", "attention.to_k", "attention.to_v", "feed_forward.w1", "feed_forward.w2"],
    "qwen": ["attn.to_q", "attn.to_k", "attn.to_v", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "qwen_image_edit": ["attn.to_q", "attn.to_k", "attn.to_v", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "flux_dev": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "flux_fill_dev": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "flux_2": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "flux_2_klein": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "flux_2_klein_4b": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "flux_2_klein_9b": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "sd15": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd15_inpainting": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd20": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd20_base": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd20_inpainting": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd20_depth": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd21": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sd21_base": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k", "attn2.to_v"],
    "sdxl": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
    "sdxl_10_base": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
    "sdxl_inpainting": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
    "sd3": ["joint_transformer.attn", "joint_transformer.to_q", "joint_transformer.to_k"],
    "sd35": ["joint_transformer.attn", "joint_transformer.to_q", "joint_transformer.to_k"],
    "pixart_alpha": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "pixart_sigma": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "sana": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "hunyuan_video": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "hi_dream_full": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "chroma_1": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "ltx2": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    "wan": [
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        "ffn.net.0.proj", "ffn.net.2",
    ],
    "wuerstchen_2": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
    "stable_cascade_1": ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"],
}


# _normalize_model_type, _coerce_dtype — imported from adapter_utils above


@dataclass
class LoConConfig:
    use_conv2d: bool = True
    use_linear: bool = True
    conv_rank: int = 4
    conv_alpha: float = 1.0


@dataclass
class LoHaConfig:
    use_effective_conv2d: bool = True


@dataclass
class LoKrConfig:
    factor: int = 2
    decompose_both: bool = False
    use_tucker: bool = False
    full_matrix: bool = False


@dataclass
class IA3Config:
    feedforward_modules: list[str] = None
    init_ia3_weights: float = 1.0

    def __post_init__(self) -> None:
        if self.feedforward_modules is None:
            self.feedforward_modules = ["mlp", "ff"]


@dataclass
class OFTConfig:
    block_size: int = 4
    is_coft: bool = False
    boft_m: int = 8
    boft_block_dim: int = 4


@dataclass
class GLoRAConfig:
    gate_init: float = 0.0


@dataclass
class DyLoRAConfig:
    min_rank: int = 4
    max_rank: int = 32
    rank_schedule: str = "linear"


@dataclass
class AdapterConfig:
    adapter_type: AdapterType
    rank: int
    alpha: float = 1.0
    target_modules: list[str] = field(default_factory=list)
    conv_rank: int | None = None
    conv_alpha: float | None = None
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    factor: int = 2
    decompose_both: bool = False
    use_tucker: bool = False
    full_matrix: bool = False
    weight_decompose: bool = False
    dora_on_output: bool = True
    rs_lora: bool = False
    block_size: int = 4
    constraint: float = 0.0
    rescaled: bool = False
    multiplier: float = 1.0


class LyCORISManager:
    """Manage native LyCORIS adapter creation, optimization params and persistence."""

    def __init__(self, config: AdapterConfig, model_type: str) -> None:
        self.config = config
        self.model_type = _normalize_model_type(model_type)
        if self.model_type in _LYCORIS_DISABLED_MODEL_TYPES:
            raise ValueError(
                f"LyCORIS adapters are temporarily disabled for model type '{self.model_type}'."
            )
        self.target_modules = (
            list(config.target_modules)
            if config.target_modules
            else list(DEFAULT_TARGETS.get(self.model_type, []))
        )
        self._network: Any | None = None
        self._target_module: nn.Module | None = None

    def _resolve_target_module(self, model_or_pipeline: Any) -> nn.Module:
        return _resolve_target_module_impl(model_or_pipeline)

    def _algo_name(self) -> str:
        return _ALGO_MAP.get(self.config.adapter_type, "lora")

    def _network_kwargs(self) -> dict[str, Any]:
        algo = self._algo_name()
        kwargs: dict[str, Any] = {
            "algo": algo,
            "conv_dim": int(self.config.conv_rank or self.config.rank),
            "conv_alpha": float(self.config.conv_alpha or self.config.alpha),
            "dropout": float(self.config.dropout),
            "rank_dropout": float(self.config.rank_dropout),
            "module_dropout": float(self.config.module_dropout),
            "block_size": int(self.config.block_size),
        }

        if algo == "lokr":
            kwargs.update(
                {
                    "factor": int(self.config.factor),
                    "decompose_both": bool(self.config.decompose_both),
                    "use_tucker": bool(self.config.use_tucker),
                    "full_matrix": bool(self.config.full_matrix),
                    "dora_wd": bool(self.config.weight_decompose),
                    "wd_on_output": bool(self.config.dora_on_output),
                    "rs_lora": bool(self.config.rs_lora),
                }
            )
        elif self.config.adapter_type == AdapterType.DORA:
            kwargs["dora_wd"] = True
            kwargs["wd_on_output"] = bool(self.config.dora_on_output)

        if algo in {"diag-oft", "boft"}:
            kwargs["constraint"] = float(self.config.constraint)
            kwargs["rescaled"] = bool(self.config.rescaled)

        return kwargs

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return _dedupe_impl(values, strip=False)

    def _target_preset_path(self) -> Path | None:
        """
        Build a LyCORIS preset TOML that scopes adaptation to configured target names.

        The upstream generic `lycoris.create_lycoris()` API only accepts a preset key/path
        and does not consume `target_modules` directly. We materialize a small preset file
        and pass its path to ensure strict module targeting.
        """
        if not self.target_modules:
            return None

        module_targets: list[str] = []
        name_targets: list[str] = []
        for raw_target in self.target_modules:
            target = str(raw_target).strip()
            if not target:
                continue

            looks_like_class_name = (
                target[0].isupper()
                and "." not in target
                and "*" not in target
                and "?" not in target
            )
            if looks_like_class_name:
                module_targets.append(target)
                continue

            if any(ch in target for ch in "*?[]"):
                name_targets.append(target)
            else:
                # Match the leaf/suffix anywhere in full module paths.
                name_targets.append(f"*{target}")

        module_targets = self._dedupe(module_targets)
        name_targets = self._dedupe(name_targets)
        if not module_targets and not name_targets:
            return None

        preset_payload: dict[str, Any] = {
            "enable_conv": False,
            "target_module": module_targets,
            "target_name": name_targets,
            "use_fnmatch": True,
        }
        preset_key = hashlib.sha1(
            json.dumps(preset_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        preset_dir = Path(tempfile.gettempdir()) / "serenity_lycoris_presets"
        preset_dir.mkdir(parents=True, exist_ok=True)
        preset_path = preset_dir / f"{preset_key}.toml"

        if not preset_path.exists():
            toml_lines = [
                "enable_conv = false",
                f"use_fnmatch = {str(bool(preset_payload['use_fnmatch'])).lower()}",
                f"target_module = {json.dumps(preset_payload['target_module'])}",
                f"target_name = {json.dumps(preset_payload['target_name'])}",
                "",
            ]
            preset_path.write_text("\n".join(toml_lines), encoding="utf-8")

        return preset_path

    def apply(self, model_or_pipeline: Any) -> Any:
        if create_lycoris is None:
            raise RuntimeError("LyCORIS is not installed. Install `lycoris` in the active environment.")

        target_module = self._resolve_target_module(model_or_pipeline)
        if self._network is not None and self._target_module is target_module:
            return self._network

        self._target_module = target_module
        network_kwargs = self._network_kwargs()
        preset_path = self._target_preset_path()
        if preset_path is not None:
            network_kwargs["preset"] = str(preset_path)
            logger.info(
                "Applying LyCORIS target filter preset (%d targets): %s",
                len(self.target_modules),
                preset_path,
            )
        network = create_lycoris(
            target_module,
            multiplier=float(self.config.multiplier),
            linear_dim=int(self.config.rank),
            linear_alpha=float(self.config.alpha),
            **network_kwargs,
        )
        network.apply_to()
        self._network = network
        return network

    def get_targets(self) -> list[str]:
        return list(self.target_modules)

    def is_attached(self) -> bool:
        return self._network is not None

    def set_multiplier(self, multiplier: float) -> None:
        if self._network is None:
            return
        self._network.set_multiplier(float(multiplier))

    def prepare_optimizer_params(self, lr: float | None = None) -> list[dict[str, Any]]:
        if self._network is None:
            raise RuntimeError("LyCORIS network is not attached. Call apply() before building optimizer params.")
        return self._network.prepare_optimizer_params(lr)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._network is None:
            return {}
        return self._network.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> Any:
        if self._network is None:
            raise RuntimeError("LyCORIS network is not attached. Call apply() before loading adapter weights.")
        return self._network.load_state_dict(state_dict, strict=strict)

    def merge_to(self, weight: float = 1.0, *, precise: bool = False) -> None:
        if self._network is None:
            raise RuntimeError("LyCORIS network is not attached. Call apply() before merge_to().")
        self._network.merge_to(float(weight), precise=bool(precise))

    def restore(self) -> None:
        if self._network is None:
            return
        self._network.restore()

    def save_weights(
        self,
        output_path: str | Path,
        dtype: torch.dtype | str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        if self._network is None:
            raise RuntimeError("LyCORIS network is not attached. Call apply() before save_weights().")

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        target_dtype = _coerce_dtype(dtype)
        self._network.save_weights(str(path), target_dtype, metadata or {})
        return path


__all__ = [
    "AdapterType",
    "DEFAULT_TARGETS",
    "LoConConfig",
    "LoHaConfig",
    "LoKrConfig",
    "IA3Config",
    "OFTConfig",
    "GLoRAConfig",
    "DyLoRAConfig",
    "AdapterConfig",
    "LyCORISManager",
]
