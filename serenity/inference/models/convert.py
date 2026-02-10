"""Key remapping from original/LDM checkpoint format to diffusers format.

Checkpoints from StabilityAI, BFL, etc. use a different key naming convention
than diffusers model classes.  This module provides conversion functions so we
can construct diffusers ``nn.Module`` instances for their architecture and then
load original-format weights into them.

Key mapping tables derived from the huggingface_guess package (Forge Neo).
"""

from __future__ import annotations

import logging
import re

import torch
import torch.nn as nn

__all__ = [
    "convert_ldm_unet_to_diffusers",
    "convert_ldm_vae_to_diffusers",
    "extract_submodel",
    "safe_load_state_dict",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefix detection and extraction
# ---------------------------------------------------------------------------

# Known prefixes for each submodel component in full checkpoints.
_UNET_PREFIXES = (
    "model.diffusion_model.",
    "model.model.",
)

_VAE_PREFIXES = (
    "first_stage_model.",
    "vae.",
)

_TEXT_PREFIXES = (
    "conditioner.embedders.",
    "cond_stage_model.",
    "text_encoders.",
)


def extract_submodel(
    state_dict: dict[str, object],
    prefixes: tuple[str, ...] | list[str],
    *,
    keep_unmatched: bool = False,
) -> dict[str, object]:
    """Extract keys matching any *prefix* and strip the prefix.

    If no keys match any prefix, returns the original dict unchanged
    (assumes keys are already unprefixed / in native format).

    Args:
        state_dict: Full checkpoint state dict.
        prefixes: Candidate prefixes to detect and strip.
        keep_unmatched: When True, also include keys that match no prefix.

    Returns:
        State dict with matching prefixes removed.
    """
    # Find which prefix (if any) has the most matches
    best_prefix = ""
    best_count = 0
    for p in prefixes:
        count = sum(1 for k in state_dict if k.startswith(p))
        if count > best_count:
            best_count = count
            best_prefix = p

    if best_count == 0:
        # No prefix matched — keys are likely already in native format.
        # Return them filtered to exclude known other-component prefixes.
        if keep_unmatched:
            return dict(state_dict)
        # Filter out keys that belong to other components
        other_prefixes = _UNET_PREFIXES + _VAE_PREFIXES + _TEXT_PREFIXES
        out = {}
        for k, v in state_dict.items():
            if not any(k.startswith(op) for op in other_prefixes):
                out[k] = v
        return out

    out = {}
    for k, v in state_dict.items():
        if k.startswith(best_prefix):
            out[k[len(best_prefix):]] = v
    return out


# ---------------------------------------------------------------------------
# UNet conversion: LDM (input_blocks/output_blocks) → diffusers (down_blocks)
# ---------------------------------------------------------------------------

# Direct (exact) key mappings: LDM name → diffusers name.
_UNET_DIRECT_MAP: dict[str, str] = {
    "time_embed.0.weight": "time_embedding.linear_1.weight",
    "time_embed.0.bias": "time_embedding.linear_1.bias",
    "time_embed.2.weight": "time_embedding.linear_2.weight",
    "time_embed.2.bias": "time_embedding.linear_2.bias",
    "input_blocks.0.0.weight": "conv_in.weight",
    "input_blocks.0.0.bias": "conv_in.bias",
    "out.0.weight": "conv_norm_out.weight",
    "out.0.bias": "conv_norm_out.bias",
    "out.2.weight": "conv_out.weight",
    "out.2.bias": "conv_out.bias",
    # SDXL addition embeddings (class conditioning)
    "label_emb.0.0.weight": "add_embedding.linear_1.weight",
    "label_emb.0.0.bias": "add_embedding.linear_1.bias",
    "label_emb.0.2.weight": "add_embedding.linear_2.weight",
    "label_emb.0.2.bias": "add_embedding.linear_2.bias",
}

# ResNet sub-key replacements (applied within a matched block).
_RESNET_MAP: list[tuple[str, str]] = [
    # (LDM, diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

# Block structure prefix map: (LDM prefix, diffusers prefix).
# Built programmatically for up to 4 resolution levels.
_LAYER_MAP: list[tuple[str, str]] = []

for _i in range(4):
    for _j in range(2):
        # Down-block resnets
        _LAYER_MAP.append((
            f"input_blocks.{3 * _i + _j + 1}.0.",
            f"down_blocks.{_i}.resnets.{_j}.",
        ))
        # Down-block attentions (last block has no attention)
        if _i < 3:
            _LAYER_MAP.append((
                f"input_blocks.{3 * _i + _j + 1}.1.",
                f"down_blocks.{_i}.attentions.{_j}.",
            ))

    for _j in range(3):
        # Up-block resnets
        _LAYER_MAP.append((
            f"output_blocks.{3 * _i + _j}.0.",
            f"up_blocks.{_i}.resnets.{_j}.",
        ))
        # Up-block attentions (first block has no attention)
        if _i > 0:
            _LAYER_MAP.append((
                f"output_blocks.{3 * _i + _j}.1.",
                f"up_blocks.{_i}.attentions.{_j}.",
            ))

    if _i < 3:
        # Downsamplers
        _LAYER_MAP.append((
            f"input_blocks.{3 * (_i + 1)}.0.op.",
            f"down_blocks.{_i}.downsamplers.0.conv.",
        ))
        # Upsamplers
        _LAYER_MAP.append((
            f"output_blocks.{3 * _i + 2}.{1 if _i == 0 else 2}.",
            f"up_blocks.{_i}.upsamplers.0.",
        ))

# Mid-block
_LAYER_MAP.append(("middle_block.1.", "mid_block.attentions.0."))
_LAYER_MAP.append(("middle_block.0.", "mid_block.resnets.0."))
_LAYER_MAP.append(("middle_block.2.", "mid_block.resnets.1."))

# Sort by LDM prefix length descending for longest-prefix matching.
_LAYER_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def _apply_resnet_remap(suffix: str) -> str:
    """Apply resnet sub-key replacements to *suffix*."""
    for ldm_part, hf_part in _RESNET_MAP:
        suffix = suffix.replace(ldm_part, hf_part)
    return suffix


def convert_ldm_unet_to_diffusers(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert a UNet state dict from LDM format to diffusers format.

    Handles SD 1.5, SDXL, and SDXL Refiner checkpoint keys.
    Keys that don't match any known pattern are passed through unchanged
    (transformer_blocks, proj_in, proj_out, etc. share the same names).

    Args:
        state_dict: UNet state dict with LDM-format keys (prefix already
            stripped).

    Returns:
        New state dict with diffusers-format keys.
    """
    # Quick check: if keys already look like diffusers format, skip conversion.
    sample_keys = set(list(state_dict.keys())[:50])
    if any("down_blocks" in k for k in sample_keys):
        logger.debug("State dict already in diffusers format, skipping conversion")
        return dict(state_dict)

    if not any("input_blocks" in k or "time_embed" in k for k in sample_keys):
        # Not LDM format — pass through.
        logger.debug("State dict not in LDM format, passing through unchanged")
        return dict(state_dict)

    new_sd: dict[str, object] = {}
    converted = 0

    for key, value in state_dict.items():
        # 1. Check direct (exact) key map
        if key in _UNET_DIRECT_MAP:
            new_sd[_UNET_DIRECT_MAP[key]] = value
            converted += 1
            continue

        # 2. Try layer structure prefix matching
        matched = False
        for ldm_prefix, hf_prefix in _LAYER_MAP:
            if key.startswith(ldm_prefix):
                suffix = key[len(ldm_prefix):]
                # Apply resnet sub-key remap within the matched block
                suffix = _apply_resnet_remap(suffix)
                new_sd[hf_prefix + suffix] = value
                matched = True
                converted += 1
                break

        if not matched:
            # Pass through unchanged (attention/transformer keys, etc.)
            new_sd[key] = value

    logger.info(
        "Converted %d/%d UNet keys from LDM to diffusers format",
        converted,
        len(state_dict),
    )
    return new_sd


# ---------------------------------------------------------------------------
# VAE conversion: LDM → diffusers
# ---------------------------------------------------------------------------

# VAE block structure maps.
_VAE_MAP: list[tuple[str, str]] = [
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for _i in range(4):
    for _j in range(2):
        _VAE_MAP.append((
            f"encoder.down.{_i}.block.{_j}.",
            f"encoder.down_blocks.{_i}.resnets.{_j}.",
        ))

    if _i < 3:
        _VAE_MAP.append((
            f"down.{_i}.downsample.",
            f"down_blocks.{_i}.downsamplers.0.",
        ))
        _VAE_MAP.append((
            f"up.{3 - _i}.upsample.",
            f"up_blocks.{_i}.upsamplers.0.",
        ))

    for _j in range(3):
        _VAE_MAP.append((
            f"decoder.up.{3 - _i}.block.{_j}.",
            f"decoder.up_blocks.{_i}.resnets.{_j}.",
        ))

for _i in range(2):
    _VAE_MAP.append((
        f"mid.block_{_i + 1}.",
        f"mid_block.resnets.{_i}.",
    ))

_VAE_MAP.sort(key=lambda x: len(x[0]), reverse=True)

# VAE attention sub-key maps.
_VAE_ATTN_MAP: list[tuple[str, str]] = [
    ("norm.", "group_norm."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
]


def convert_ldm_vae_to_diffusers(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert a VAE state dict from LDM format to diffusers format.

    Args:
        state_dict: VAE state dict with LDM-format keys (prefix already
            stripped).

    Returns:
        New state dict with diffusers-format keys.
    """
    # Quick check: already diffusers format?
    sample_keys = set(list(state_dict.keys())[:50])
    if any("down_blocks" in k for k in sample_keys):
        return dict(state_dict)
    if not any("mid.block_1" in k or "encoder.down.0" in k for k in sample_keys):
        return dict(state_dict)

    new_sd: dict[str, object] = {}
    for key, value in state_dict.items():
        new_key = key

        # Apply block structure maps
        for ldm_part, hf_part in _VAE_MAP:
            new_key = new_key.replace(ldm_part, hf_part)

        # Apply attention maps within attention blocks
        if "attentions" in new_key:
            for ldm_part, hf_part in _VAE_ATTN_MAP:
                new_key = new_key.replace(ldm_part, hf_part)

        new_sd[new_key] = value

    return new_sd


# ---------------------------------------------------------------------------
# Flux / SD3 / Wan key conversion (prefix-based, minimal remapping)
# ---------------------------------------------------------------------------

# Flux BFL format → diffusers FluxTransformer2DModel format.
# BFL checkpoints use the same key names as diffusers (double_blocks,
# single_blocks, etc.) so usually only prefix stripping is needed.
# However, some community checkpoints may have minor differences.

_FLUX_REMAP: list[tuple[re.Pattern, str]] = [
    # img_in → x_embedder  (some older BFL checkpoints)
    (re.compile(r"^img_in\."), "x_embedder."),
    # txt_in → context_embedder  (some older BFL checkpoints)
    (re.compile(r"^txt_in\."), "context_embedder."),
    # time_in → time_text_embed.timestep_embedder  (some checkpoints)
    (re.compile(r"^time_in\."), "time_text_embed.timestep_embedder."),
    # vector_in → time_text_embed.text_embedder  (some checkpoints)
    (re.compile(r"^vector_in\."), "time_text_embed.text_embedder."),
    # guidance_in → time_text_embed.guidance_embedder  (Flux dev)
    (re.compile(r"^guidance_in\."), "time_text_embed.guidance_embedder."),
    # final_layer → norm_out  (some checkpoints)
    (re.compile(r"^final_layer\."), "norm_out."),
]


def convert_flux_to_diffusers(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert Flux state dict from BFL format to diffusers format.

    Only applies remapping when needed — if keys already match diffusers
    naming, passes through unchanged.
    """
    sample_keys = list(state_dict.keys())[:20]
    # Check if already in diffusers format
    if any("transformer_blocks" in k or "x_embedder" in k for k in sample_keys):
        return dict(state_dict)

    # Check if BFL format (img_in, double_blocks, etc.)
    has_bfl_keys = any(
        k.startswith(("img_in.", "double_blocks.", "single_blocks.", "time_in."))
        for k in sample_keys
    )
    if not has_bfl_keys:
        return dict(state_dict)

    new_sd: dict[str, object] = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in _FLUX_REMAP:
            new_key = pattern.sub(replacement, new_key)
        new_sd[new_key] = value

    return new_sd


def convert_sd3_to_diffusers(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Convert SD3 state dict from original format to diffusers format.

    SD3 checkpoints may use ``joint_blocks`` (original) or
    ``transformer_blocks`` (diffusers) naming.
    """
    sample_keys = list(state_dict.keys())[:20]
    if any("transformer_blocks" in k for k in sample_keys):
        return dict(state_dict)

    # SD3 original format uses joint_blocks — diffusers also uses this name,
    # so typically no conversion is needed.
    return dict(state_dict)


# ---------------------------------------------------------------------------
# Safe state dict loading (shape-aware)
# ---------------------------------------------------------------------------


def safe_load_state_dict(
    model: nn.Module,
    state_dict: dict[str, object],
) -> tuple[list[str], list[str], list[str]]:
    """Load state dict into model, skipping keys with shape mismatches.

    Unlike ``model.load_state_dict(strict=False)`` this will not raise on
    size mismatches — it simply skips those keys and reports them.

    Returns:
        Tuple of (missing_keys, unexpected_keys, mismatched_keys).
    """
    model_state = model.state_dict()

    # Partition incoming keys
    loadable: dict[str, object] = {}
    unexpected: list[str] = []
    mismatched: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
        elif not isinstance(value, torch.Tensor):
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            mismatched.append(key)
        else:
            loadable[key] = value

    # Load only matching keys
    missing_keys, _ = model.load_state_dict(loadable, strict=False)

    if mismatched:
        logger.warning(
            "%d keys skipped due to shape mismatch: %s",
            len(mismatched),
            mismatched[:5],
        )

    return list(missing_keys), unexpected, mismatched
