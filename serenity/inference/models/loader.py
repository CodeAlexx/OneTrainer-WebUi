"""Model loading utilities for the Serenity inference engine."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from serenity.inference.models.detection import ModelConfig, detect_from_file, detect_model_type
from serenity.inference.quantization.ops import OperationContext

__all__ = [
    "extract_vae_state_dict",
    "load_model",
    "load_state_dict",
    "load_vae",
]

logger = logging.getLogger(__name__)

# Known VAE key prefixes in full checkpoints.
_VAE_PREFIXES = ("first_stage_model.", "vae.")


# ---------------------------------------------------------------------------
# State-dict loading
# ---------------------------------------------------------------------------


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a state dict from a safetensors or PyTorch checkpoint.

    Args:
        path: Filesystem path to ``.safetensors``, ``.pt``, ``.bin``,
            or ``.ckpt`` file.

    Returns:
        Flat mapping of parameter names to tensors.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file  # type: ignore[import-untyped]

            return load_file(path, device="cpu")
        except ImportError:
            logger.warning(
                "safetensors package not installed; falling back to torch.load "
                "for %s",
                path,
            )

    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict):
        # Unwrap nested state_dict key used by some checkpoints
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return sd
    raise ValueError(f"Unexpected checkpoint format in {path}")


# ---------------------------------------------------------------------------
# VAE extraction
# ---------------------------------------------------------------------------


def extract_vae_state_dict(full_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract VAE-related keys from a full model checkpoint.

    Looks for keys starting with ``"first_stage_model."`` or ``"vae."``
    and strips the prefix so the returned dict can be loaded directly
    into a standalone VAE model.

    Args:
        full_state_dict: Complete checkpoint state dict.

    Returns:
        State dict containing only VAE parameters with prefixes removed.
    """
    vae_sd: dict[str, torch.Tensor] = {}
    for key, value in full_state_dict.items():
        for prefix in _VAE_PREFIXES:
            if key.startswith(prefix):
                stripped = key[len(prefix) :]
                vae_sd[stripped] = value
                break
    return vae_sd


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    path: str,
    config: ModelConfig | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    ops_context: OperationContext | None = None,
) -> nn.Module:
    """Load a diffusion model from a checkpoint file.

    Args:
        path: Path to the model file.
        config: Optional pre-detected :class:`ModelConfig`.  When
            ``None``, the architecture is detected automatically.
        device: Target device for the loaded model.
        dtype: Target dtype for model parameters.
        ops_context: Optional operation context for quantized layers.

    Returns:
        The loaded model on the specified device.
    """
    sd = load_state_dict(path)

    if config is None:
        config = detect_model_type(set(sd.keys()))
        if config is None:
            config = detect_from_file(path)
        if config is None:
            raise ValueError(
                f"Cannot detect model architecture from {path}. "
                "Please provide a ModelConfig explicitly."
            )

    logger.info(
        "Loading %s model from %s (device=%s, dtype=%s)",
        config.architecture.value,
        path,
        device,
        dtype,
    )

    # Import adapter registry lazily to avoid circular imports
    adapter = _get_adapter(config)
    if adapter is None:
        raise NotImplementedError(
            f"No model adapter registered for {config.architecture.value}. "
            "Architecture-specific adapters must be implemented."
        )

    model = adapter.create_model(
        sd,
        device=device,
        dtype=dtype,
        ops_context=ops_context,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _get_adapter(config: ModelConfig) -> Any:
    """Look up a model adapter for *config*.

    Returns ``None`` when no adapter has been registered yet.  This is
    the expected state during early development -- callers should raise
    a clear error.
    """
    # Adapter registry will be populated by architecture-specific modules.
    # For now we return None -- load_model raises NotImplementedError.
    return None


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------


def load_vae(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load a VAE model from a standalone or full checkpoint.

    If the file contains a full model checkpoint, the VAE weights are
    extracted automatically.

    Args:
        path: Path to the VAE or full model file.
        device: Target device.
        dtype: Target dtype.

    Returns:
        The loaded VAE ``nn.Module``.
    """
    sd = load_state_dict(path)

    # Check whether this looks like a full checkpoint with a VAE inside
    has_vae_prefix = any(
        k.startswith(prefix) for k in sd for prefix in _VAE_PREFIXES
    )
    if has_vae_prefix:
        sd = extract_vae_state_dict(sd)

    # Try loading via diffusers AutoencoderKL if available
    try:
        from diffusers.models import AutoencoderKL  # type: ignore[import-untyped]

        vae = AutoencoderKL()
        vae.load_state_dict(sd, strict=False)
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()
        logger.info("Loaded VAE via diffusers AutoencoderKL from %s", path)
        return vae
    except Exception:
        logger.debug(
            "diffusers AutoencoderKL not available or failed; "
            "returning raw state dict as a fallback is not supported. "
            "Ensure diffusers is installed for VAE loading."
        )
        raise
