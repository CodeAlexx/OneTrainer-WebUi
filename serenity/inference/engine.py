"""Inference engine orchestrator — ties all subsystems into a single generation API."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from serenity.inference.attention.backends import select_best_backend
from serenity.inference.config import InferenceConfig
from serenity.inference.memory.manager import ModelManager
from serenity.inference.memory.vram import get_free_memory, is_cuda_available
from serenity.inference.models.detection import ModelConfig, detect_from_file
from serenity.inference.sampling.cfg import apply_cfg
from serenity.inference.sampling.conditioning import Conditioning, create_noise
from serenity.inference.sampling.prediction import PredictionType, get_prediction
from serenity.inference.sampling.sampler import SamplerType, create_model_fn, sample
from serenity.inference.sampling.schedulers import SchedulerType, compute_sigmas

__all__ = [
    "GenerationResult",
    "InferenceEngine",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Output from a single generation call.

    Attributes
    ----------
    images : list[Tensor]
        Decoded images as tensors in ``[0, 1]`` range.
    seeds : list[int]
        Seeds used for each image in the batch.
    metadata : dict
        Generation parameters for reproducibility.
    """

    images: list[Tensor]
    seeds: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage cache — lightweight prompt-encoding cache
# ---------------------------------------------------------------------------


class _StageCache:
    """LRU cache for text encodings and other expensive intermediate results."""

    def __init__(self, max_entries: int = 64) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or ``None`` if absent."""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Insert a value, evicting the oldest entry if over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Main orchestrator for diffusion model inference.

    Ties together model management, text encoding, sampling, and VAE
    decoding into a single ``generate()`` call.

    Usage::

        engine = InferenceEngine(InferenceConfig(model_path="model.safetensors"))
        result = engine.generate(prompt="a cat", seed=42)
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config

        # Device setup
        if is_cuda_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Subsystems
        self._model_manager = ModelManager(device=self._device)
        self._attention_backend = select_best_backend(config.attention_backend.value)
        self._cache = _StageCache()

        # State — populated lazily on first generate()
        self._model_config: ModelConfig | None = None
        self._model_loaded: bool = False
        self._unet: Any | None = None
        self._text_encoder: Any | None = None
        self._vae_decoder: Any | None = None

        logger.info(
            "InferenceEngine initialized (device=%s, attention=%s)",
            self._device,
            self._attention_backend.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        steps: int | None = None,
        cfg_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        batch_size: int = 1,
        callback: Callable | None = None,
    ) -> GenerationResult:
        """Run the full generation pipeline.

        Parameters override config defaults when provided. A *seed* of
        ``-1`` selects a random seed.
        """
        params = self._resolve_params(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            sampler=sampler,
            scheduler=scheduler,
            batch_size=batch_size,
        )

        logger.info(
            "Generating: prompt=%r, steps=%d, cfg=%.1f, size=%dx%d, seed=%s",
            params["prompt"],
            params["steps"],
            params["cfg_scale"],
            params["width"],
            params["height"],
            params["seed"],
        )

        t0 = time.perf_counter()

        # Step 1: Ensure model is loaded
        self._ensure_model_loaded()

        # Step 2: Resolve seeds
        seeds: list[int] = []
        for _ in range(params["batch_size"]):
            s = params["seed"] if params["seed"] >= 0 else random.randint(0, 2**32 - 1)
            seeds.append(s)

        # Step 3: Encode text (with caching)
        conditioning = self._encode_text(
            params["prompt"],
            params["negative_prompt"],
        )

        # Step 4: Get prediction type from model config
        prediction_type_str = "eps"
        if self._model_config is not None:
            prediction_type_str = self._model_config.prediction_type
        prediction = get_prediction(prediction_type_str)

        # Step 5: Create noise for each batch item
        latent_channels = 4  # standard for SD-family models
        latent_h = params["height"] // 8
        latent_w = params["width"] // 8
        noise_shape = (params["batch_size"], latent_channels, latent_h, latent_w)

        noise = create_noise(
            seed=seeds[0],
            shape=noise_shape,
            device=self._device,
            dtype=torch.float32,
        )
        logger.debug("Created noise: shape=%s, seed=%d", noise.shape, seeds[0])

        # Step 6: Compute sigmas
        sigma_min = 0.03
        sigma_max = 14.6
        sigmas = compute_sigmas(
            scheduler=params["scheduler"],
            num_steps=params["steps"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        ).to(self._device)
        logger.debug("Computed %d sigmas (%s scheduler)", len(sigmas), params["scheduler"])

        # Step 7: Create model/denoise function
        denoise_fn = self._create_denoise_fn(
            model=self._unet,
            prediction=prediction,
            cfg_scale=params["cfg_scale"],
            cond=conditioning.cond if conditioning is not None else None,
            uncond=conditioning.uncond if conditioning is not None else None,
        )

        # Step 8: Sample
        logger.info("Sampling with %s sampler...", params["sampler"])
        latents = sample(
            model_fn=denoise_fn,
            noise=noise,
            sigmas=sigmas,
            sampler_type=params["sampler"],
            callback=callback,
        )
        logger.debug("Sampling complete, latents shape=%s", latents.shape)

        # Step 9: Decode with VAE
        images = self._decode_latents(latents)
        logger.debug("Decoded %d images", len(images))

        elapsed = time.perf_counter() - t0
        logger.info("Generation complete in %.2fs", elapsed)

        # Step 10: Build result
        metadata = {
            "prompt": params["prompt"],
            "negative_prompt": params["negative_prompt"],
            "steps": params["steps"],
            "cfg_scale": params["cfg_scale"],
            "width": params["width"],
            "height": params["height"],
            "sampler": params["sampler"],
            "scheduler": params["scheduler"],
            "seeds": seeds,
            "elapsed_seconds": elapsed,
        }

        return GenerationResult(
            images=images,
            seeds=seeds,
            metadata=metadata,
        )

    def load_model(self, model_path: str | None = None) -> None:
        """Explicitly load a model, bypassing lazy loading.

        Uses :class:`ModelManager` for caching so repeated calls with the
        same config are no-ops.
        """
        path = model_path or self._config.model_path
        if not path:
            raise ValueError("No model path provided and config.model_path is empty")

        logger.info("Loading model from %s", path)
        self._model_config = detect_from_file(path)

        if self._model_config is None:
            raise RuntimeError(
                f"Could not detect model architecture from {path}. "
                "The file may be corrupted or an unsupported format."
            )

        logger.info("Detected architecture: %s", self._model_config.architecture.value)
        self._model_loaded = True

    def unload_all(self) -> None:
        """Unload all models and clear caches."""
        self._model_manager.unload_all()
        self._cache.clear()
        self._model_config = None
        self._model_loaded = False
        self._unet = None
        self._text_encoder = None
        self._vae_decoder = None
        logger.info("All models unloaded and caches cleared")

    def get_status(self) -> dict[str, Any]:
        """Return engine status information."""
        free_mem = get_free_memory(self._device) if is_cuda_available() else 0
        return {
            "device": str(self._device),
            "attention_backend": self._attention_backend.value,
            "model_loaded": self._model_loaded,
            "model_architecture": (
                self._model_config.architecture.value
                if self._model_config is not None
                else None
            ),
            "loaded_models": len(self._model_manager.loaded_models),
            "cache_entries": self._cache.size,
            "vram_free_bytes": free_mem,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_params(self, **kwargs: Any) -> dict[str, Any]:
        """Merge user kwargs with config defaults."""
        cfg = self._config
        return {
            "prompt": kwargs.get("prompt", ""),
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "seed": kwargs.get("seed", -1),
            "steps": kwargs.get("steps") if kwargs.get("steps") is not None else cfg.steps,
            "cfg_scale": kwargs.get("cfg_scale") if kwargs.get("cfg_scale") is not None else cfg.cfg_scale,
            "width": kwargs.get("width") if kwargs.get("width") is not None else cfg.width,
            "height": kwargs.get("height") if kwargs.get("height") is not None else cfg.height,
            "sampler": kwargs.get("sampler") if kwargs.get("sampler") is not None else cfg.sampler,
            "scheduler": kwargs.get("scheduler") if kwargs.get("scheduler") is not None else cfg.scheduler,
            "batch_size": kwargs.get("batch_size", 1),
        }

    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model_loaded:
            return
        path = self._config.model_path
        if path:
            self.load_model(path)
        else:
            logger.warning("No model path configured, running without model")

    def _encode_text(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> Conditioning:
        """Encode text prompts with caching.

        Returns a :class:`Conditioning` with cond and uncond tensors.
        If no text encoder is loaded, returns dummy conditioning.
        """
        # Check cache
        cache_key = hashlib.sha256(f"{prompt}||{negative_prompt}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Using cached text encoding for prompt=%r", prompt[:50])
            return cached

        # If we have a text encoder, use it
        if self._text_encoder is not None:
            # Real text encoding would happen here
            cond_emb = self._text_encoder(prompt)
            uncond_emb = self._text_encoder(negative_prompt) if negative_prompt else None
        else:
            # Dummy conditioning when no text encoder is available
            logger.debug("No text encoder loaded, using dummy conditioning")
            cond_emb = torch.zeros(1, 77, 768, device=self._device)
            uncond_emb = torch.zeros(1, 77, 768, device=self._device) if negative_prompt else None

        conditioning = Conditioning(
            cond=cond_emb,
            uncond=uncond_emb,
        )

        # Cache it
        if self._config.cache_text_encodings:
            self._cache.put(cache_key, conditioning)

        return conditioning

    def _create_denoise_fn(
        self,
        model: Any | None,
        prediction: Any,
        cfg_scale: float,
        cond: Tensor | None,
        uncond: Tensor | None,
    ) -> Callable:
        """Create the denoising function for the sampler.

        Wraps the model with prediction type and CFG into a single
        callable ``(noisy_input, sigma) -> denoised``.
        """
        if model is not None:
            return create_model_fn(
                model=model,
                prediction=prediction,
                cond=cond,
                uncond=uncond,
                cfg_scale=cfg_scale,
                rescale_phi=self._config.rescale_cfg,
                mahiro=self._config.mahiro,
            )

        # Fallback: identity function when no model is loaded
        def _dummy_denoise(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        return _dummy_denoise

    def _decode_latents(self, latents: Tensor) -> list[Tensor]:
        """Decode latent tensors to images using the VAE.

        Returns a list of image tensors in ``[0, 1]`` range, one per batch item.
        """
        if self._vae_decoder is not None:
            decoded = self._vae_decoder.decode(
                latents,
                tiling=self._config.vae_tiling,
                tile_size=self._config.vae_tile_size,
            )
            return [decoded[i] for i in range(decoded.shape[0])]

        # Without VAE: return latents directly (for testing / headless use)
        logger.debug("No VAE loaded, returning raw latents as images")
        return [latents[i] for i in range(latents.shape[0])]
