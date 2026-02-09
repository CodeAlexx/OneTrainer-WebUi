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
    if ops_context is not None:
        model = model.to(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


_ADAPTER_REGISTRY: dict | None = None

# Architectures that can be *detected* but not yet *loaded*.
_UNSUPPORTED_ARCHITECTURES: dict[str, str] = {
    "lumina": "Lumina adapter is not yet implemented. Contributions welcome!",
    "zimage": "Z-Image adapter is not yet implemented. Contributions welcome!",
    "qwen": "Qwen adapter is not yet implemented. Contributions welcome!",
}


def _get_adapter(config: ModelConfig) -> Any:
    """Look up a model adapter for *config*."""
    global _ADAPTER_REGISTRY
    if _ADAPTER_REGISTRY is None:
        _ADAPTER_REGISTRY = {}
        # Import all adapter modules and merge their ADAPTERS dicts.
        # Lumina, ZImage, and Qwen are intentionally excluded — their
        # ADAPTERS dicts are empty because create_model() is not yet
        # implemented for those architectures.
        from serenity.inference.models import sd15, sdxl, sd3, flux, chroma, wan
        for mod in (sd15, sdxl, sd3, flux, chroma, wan):
            _ADAPTER_REGISTRY.update(mod.ADAPTERS)

    # Check for detected-but-unsupported architectures first so we give
    # a clear, helpful error instead of a generic "no adapter" message.
    arch_value = config.architecture.value
    if arch_value in _UNSUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Architecture '{arch_value}' is not yet supported for loading. "
            f"{_UNSUPPORTED_ARCHITECTURES[arch_value]}"
        )

    adapter_cls = _ADAPTER_REGISTRY.get(config.architecture)
    if adapter_cls is None:
        return None
    return adapter_cls()


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

        # Infer latent channels from decoder input shape
        latent_ch = 4  # default
        if "decoder.conv_in.weight" in sd:
            latent_ch = sd["decoder.conv_in.weight"].shape[1]

        vae = AutoencoderKL(latent_channels=latent_ch)
        vae.load_state_dict(sd, strict=False)
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()
        logger.info("Loaded VAE via diffusers AutoencoderKL from %s (latent_ch=%d)", path, latent_ch)
        return vae
    except (ImportError, OSError, RuntimeError):
        logger.debug(
            "diffusers AutoencoderKL not available or failed; "
            "returning raw state dict as a fallback is not supported. "
            "Ensure diffusers is installed for VAE loading."
        )
        raise
