"""Detect model architecture from state-dict keys.

Examines the key structure of a checkpoint to identify which diffusion model
architecture it represents, without needing to fully load the weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ModelArchitecture",
    "ModelConfig",
    "detect_from_file",
    "detect_model_type",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class ModelArchitecture(str, Enum):
    """Known model architectures that can be auto-detected."""

    SD15 = "sd15"
    SDXL = "sdxl"
    SDXL_REFINER = "sdxl_refiner"
    SD3 = "sd3"
    FLUX_DEV = "flux_dev"
    FLUX_SCHNELL = "flux_schnell"
    CHROMA = "chroma"
    WAN = "wan"
    QWEN = "qwen"
    LUMINA = "lumina"
    ZIMAGE = "zimage"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Result of model type detection."""

    architecture: ModelArchitecture
    unet_config: dict = field(default_factory=dict)
    unet_key_prefix: list[str] = field(default_factory=list)
    vae_key_prefix: list[str] = field(default_factory=list)
    prediction_type: str = "eps"


# ---------------------------------------------------------------------------
# Key-pattern helpers
# ---------------------------------------------------------------------------


def _has_prefix(keys: set[str], prefix: str) -> bool:
    """Return True if any key starts with *prefix*."""
    return any(k.startswith(prefix) for k in keys)


def _has_key(keys: set[str], key: str) -> bool:
    """Return True if *key* exists in the set."""
    return key in keys


def _count_prefix(keys: set[str], prefix: str) -> int:
    """Count keys starting with *prefix*."""
    return sum(1 for k in keys if k.startswith(prefix))


def _count_blocks(keys: set[str], template: str) -> int:
    """Count sequential numbered blocks matching *template* with ``{}``."""
    count = 0
    while True:
        prefix = template.format(count)
        if any(k.startswith(prefix) for k in keys):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Unet key prefix detection (mirrors Forge detection.py)
# ---------------------------------------------------------------------------


def _detect_unet_prefix(keys: set[str]) -> str:
    """Determine the unet key prefix."""
    candidates = {
        "model.diffusion_model.": 0,
        "model.model.": 0,
        "net.": 0,
    }
    for k in keys:
        for c in candidates:
            if k.startswith(c):
                candidates[c] += 1
                break

    top = max(candidates, key=candidates.get)
    if candidates[top] > 5:
        return top
    return "model."


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


def _detect_chroma(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Chroma variant (distilled flux)."""
    # Chroma has distilled_guidance_layer keys alongside double_blocks
    dgk1 = f"{prefix}distilled_guidance_layer.0.norms.0.scale"
    dgk2 = f"{prefix}distilled_guidance_layer.norms.0.scale"
    if _has_key(keys, dgk1) or _has_key(keys, dgk2):
        return ModelConfig(
            architecture=ModelArchitecture.CHROMA,
            unet_config={"image_model": "chroma"},
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )
    return None


def _detect_flux(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Flux.1 / Flux.2 architectures."""
    norm_key = f"{prefix}double_blocks.0.img_attn.norm.key_norm.scale"
    img_in_key = f"{prefix}img_in.weight"
    distilled_key = f"{prefix}distilled_guidance_layer.norms.0.scale"

    if not _has_key(keys, norm_key):
        return None
    if not (_has_key(keys, img_in_key) or _has_key(keys, distilled_key)):
        return None

    # Check for Chroma first (it has flux-like keys plus distilled guidance)
    chroma = _detect_chroma(keys, prefix)
    if chroma is not None:
        return chroma

    # Flux.2 vs Flux.1
    is_flux2 = _has_key(keys, f"{prefix}double_stream_modulation_img.lin.weight")

    guidance_embed = _has_key(keys, f"{prefix}guidance_in.in_layer.weight")

    depth = _count_blocks(keys, f"{prefix}double_blocks." + "{}.")
    depth_single = _count_blocks(keys, f"{prefix}single_blocks." + "{}.")

    if is_flux2:
        config = {
            "image_model": "flux2",
            "depth": depth,
            "depth_single_blocks": depth_single,
        }
        return ModelConfig(
            architecture=ModelArchitecture.FLUX_DEV,
            unet_config=config,
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )

    # Flux.1 dev vs schnell: dev has guidance_embed
    arch = ModelArchitecture.FLUX_DEV if guidance_embed else ModelArchitecture.FLUX_SCHNELL
    config = {
        "image_model": "flux",
        "guidance_embed": guidance_embed,
        "depth": depth,
        "depth_single_blocks": depth_single,
    }
    return ModelConfig(
        architecture=arch,
        unet_config=config,
        unet_key_prefix=[prefix],
        vae_key_prefix=["vae."],
        prediction_type="flow",
    )


def _detect_flux_nf4(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect BNB NF4 quantized Flux."""
    nf4_key = f"{prefix}double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4"
    if _has_key(keys, nf4_key):
        return ModelConfig(
            architecture=ModelArchitecture.FLUX_DEV,
            unet_config={"image_model": "flux", "guidance_embed": True},
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )
    return None


def _detect_sd3(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Stable Diffusion 3 (joint transformer)."""
    if _has_prefix(keys, f"{prefix}joint_blocks.0."):
        depth = _count_blocks(keys, f"{prefix}joint_blocks." + "{}.")
        return ModelConfig(
            architecture=ModelArchitecture.SD3,
            unet_config={"depth": depth},
            unet_key_prefix=[prefix],
            vae_key_prefix=["first_stage_model."],
            prediction_type="flow",
        )
    return None


def _detect_lumina_zimage(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Lumina 2 or Z-Image (both use cap_embedder)."""
    cap_key = f"{prefix}cap_embedder.1.weight"
    if not _has_key(keys, cap_key):
        return None

    # Differentiate by layer count / dim — we check layers
    n_layers = _count_blocks(keys, f"{prefix}layers." + "{}.")

    # Z-Image has dim=3840 (30 heads), Lumina2 has dim=2304 (24 heads)
    # We can heuristically check for z_image_modulation-related keys
    # or just count the number of heads by checking for specific keys
    # For detection, we check a broader pattern
    ffn_key_0 = f"{prefix}layers.0.feed_forward.w1.weight"
    pad_token_key = f"{prefix}cap_pad_token"

    # Z-Image usually has pad_token and more layers
    if _has_key(keys, pad_token_key):
        return ModelConfig(
            architecture=ModelArchitecture.ZIMAGE,
            unet_config={"image_model": "lumina2", "n_layers": n_layers},
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )

    return ModelConfig(
        architecture=ModelArchitecture.LUMINA,
        unet_config={"image_model": "lumina2", "n_layers": n_layers},
        unet_key_prefix=[prefix],
        vae_key_prefix=["vae."],
        prediction_type="flow",
    )


def _detect_wan(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Wan 2.1 architecture."""
    modulation_key = f"{prefix}head.modulation"
    if _has_key(keys, modulation_key):
        n_layers = _count_blocks(keys, f"{prefix}blocks." + "{}.")
        is_i2v = _has_key(keys, f"{prefix}img_emb.proj.0.bias")
        model_type = "i2v" if is_i2v else "t2v"
        return ModelConfig(
            architecture=ModelArchitecture.WAN,
            unet_config={
                "image_model": "wan2.1",
                "model_type": model_type,
                "num_layers": n_layers,
            },
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )
    return None


def _detect_qwen(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect Qwen Image architecture."""
    txt_norm_key = f"{prefix}txt_norm.weight"
    if _has_key(keys, txt_norm_key):
        n_layers = _count_blocks(keys, f"{prefix}transformer_blocks." + "{}.")
        return ModelConfig(
            architecture=ModelArchitecture.QWEN,
            unet_config={"image_model": "qwen_image", "num_layers": n_layers},
            unet_key_prefix=[prefix],
            vae_key_prefix=["vae."],
            prediction_type="flow",
        )
    return None


def _detect_unet_sd(keys: set[str], prefix: str) -> ModelConfig | None:
    """Detect SD 1.5 / SDXL / SDXL Refiner (UNet-based)."""
    input_key = f"{prefix}input_blocks.0.0.weight"
    if not _has_key(keys, input_key):
        return None

    # Check for ADM (class conditioning) — present in SDXL/refiner
    label_key = f"{prefix}label_emb.0.0.weight"
    has_adm = _has_key(keys, label_key)

    # Count input blocks to determine architecture
    input_block_count = _count_blocks(keys, f"{prefix}input_blocks." + "{}.")

    # Check for transformer blocks to detect context_dim
    # Look for the first transformer attention key
    context_dim = None
    use_linear = False

    for i in range(input_block_count):
        attn_key = f"{prefix}input_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight"
        if _has_key(keys, attn_key):
            # We can't read the tensor shape without loading, so detect from
            # other structural cues
            proj_in = f"{prefix}input_blocks.{i}.1.proj_in.weight"
            if _has_key(keys, proj_in):
                # proj_in is a linear layer in SDXL, conv in SD15
                # We check if proj_in has 2D weight (linear) vs 4D (conv)
                use_linear = True  # Can't check shape here
            break

    # Distinguish SDXL from SD1.5 by structural keys
    # SDXL has label_emb (ADM), SD1.5 does not
    if not has_adm:
        # SD 1.5
        return ModelConfig(
            architecture=ModelArchitecture.SD15,
            unet_config={
                "context_dim": 768,
                "model_channels": 320,
                "use_linear_in_transformer": False,
            },
            unet_key_prefix=[prefix],
            vae_key_prefix=["first_stage_model."],
            prediction_type="eps",
        )

    # SDXL vs Refiner — check model_channels by counting output blocks
    # Refiner has model_channels=384, SDXL has 320
    # Heuristic: refiner has context_dim 1280 with adm_in_channels 2560
    # SDXL has context_dim 2048 with adm_in_channels 2816
    # We can distinguish by checking transformer_depth patterns
    # Refiner: [0,0,4,4,4,4,0,0], SDXL: [0,0,2,2,10,10]

    # Count middle block transformer depth
    middle_depth = _count_blocks(
        keys, f"{prefix}middle_block.1.transformer_blocks." + "{}"
    )

    # SDXL has middle_depth=10, refiner has middle_depth=4
    if middle_depth <= 4 and middle_depth > 0:
        # Check for refiner-specific channel count
        # Refiner output_blocks structure differs
        return ModelConfig(
            architecture=ModelArchitecture.SDXL_REFINER,
            unet_config={
                "model_channels": 384,
                "context_dim": 1280,
                "adm_in_channels": 2560,
                "use_linear_in_transformer": True,
            },
            unet_key_prefix=[prefix],
            vae_key_prefix=["first_stage_model."],
            prediction_type="eps",
        )

    return ModelConfig(
        architecture=ModelArchitecture.SDXL,
        unet_config={
            "model_channels": 320,
            "context_dim": 2048,
            "adm_in_channels": 2816,
            "use_linear_in_transformer": True,
        },
        unet_key_prefix=[prefix],
        vae_key_prefix=["first_stage_model."],
        prediction_type="eps",
    )


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

# Ordered list of detectors — first match wins.
_DETECTORS = [
    _detect_lumina_zimage,
    _detect_wan,
    _detect_flux_nf4,
    _detect_qwen,
    _detect_flux,  # also handles chroma
    _detect_sd3,
    _detect_unet_sd,
]


def detect_model_type(state_dict_keys: set[str]) -> ModelConfig | None:
    """Detect model architecture from checkpoint key names.

    Args:
        state_dict_keys: Set of key names from the state dict.

    Returns:
        A :class:`ModelConfig` if the architecture is recognised, else ``None``.
    """
    keys = set(state_dict_keys)

    # Determine the unet prefix
    prefix = _detect_unet_prefix(keys)

    for detector in _DETECTORS:
        result = detector(keys, prefix)
        if result is not None:
            logger.info(
                "Detected model architecture: %s (prefix=%s)",
                result.architecture.value,
                prefix,
            )
            return result

    logger.warning("Could not detect model architecture from state-dict keys")
    return None


# ---------------------------------------------------------------------------
# File-based detection
# ---------------------------------------------------------------------------


def detect_from_file(path: str) -> ModelConfig | None:
    """Detect model architecture from a safetensors or checkpoint file.

    For safetensors files, reads only the metadata/header to extract key
    names without loading the full weights.

    Args:
        path: Path to ``.safetensors`` or ``.ckpt`` / ``.pt`` file.

    Returns:
        A :class:`ModelConfig` if recognised, else ``None``.
    """
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.endswith(".safetensors"):
        try:
            from safetensors import safe_open  # type: ignore[import-untyped]

            with safe_open(path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
            return detect_model_type(keys)
        except ImportError:
            logger.warning(
                "safetensors not installed; falling back to torch.load for %s",
                path,
            )

    # Fallback: load with torch (maps to CPU, weights_only=True)
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = set(sd.keys())
    else:
        raise ValueError(f"Unexpected checkpoint format in {path}")

    return detect_model_type(keys)
