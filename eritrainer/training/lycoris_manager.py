"""LyCORIS adapter configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


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


DEFAULT_TARGETS: Dict[str, List[str]] = {
    "zimage": ["attention.to_q", "attention.to_k", "attention.to_v", "feed_forward.w1", "feed_forward.w2"],
    "z_image": ["attention.to_q", "attention.to_k", "attention.to_v", "feed_forward.w1", "feed_forward.w2"],
    "sdxl": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
    "sd3": ["joint_transformer.attn", "joint_transformer.to_q", "joint_transformer.to_k"],
}


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
    feedforward_modules: List[str] = None
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
    target_modules: List[str] = field(default_factory=list)


class LyCORISManager:
    """Minimal LyCORIS manager placeholder for native training wiring."""

    def __init__(self, config: AdapterConfig, model_type: str) -> None:
        self.config = config
        self.model_type = str(model_type).lower()
        self.target_modules = (
            config.target_modules
            if config.target_modules
            else DEFAULT_TARGETS.get(self.model_type, [])
        )

    def get_targets(self) -> List[str]:
        return list(self.target_modules)


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
