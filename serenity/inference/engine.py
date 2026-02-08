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
from serenity.inference.models.detection import ModelArchitecture, ModelConfig, detect_from_file
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

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


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
        self._tokenizer: Any | None = None
        self._vae_decoder: Any | None = None

        # Sigma schedule (built from model's alphas_cumprod)
        self._log_sigmas: Tensor | None = None
        self._sigma_min: float = 0.0292
        self._sigma_max: float = 14.6146

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

        # Step 5b: Compute sigmas early so we can scale noise
        sigmas = compute_sigmas(
            scheduler=params["scheduler"],
            num_steps=params["steps"],
            sigma_min=self._sigma_min,
            sigma_max=self._sigma_max,
        ).to(self._device)
        logger.debug("Computed %d sigmas (%s scheduler)", len(sigmas), params["scheduler"])

        # Scale noise to sigma_max (k-diffusion convention)
        noise = noise * sigmas[0]

        # Step 6: Create model/denoise function
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
        """Load a model with full weight loading for supported architectures.

        Detects the architecture from the checkpoint header, then loads
        UNet, text encoder, tokenizer, and VAE for supported families.
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

        arch = self._model_config.architecture
        logger.info("Detected architecture: %s", arch.value)

        dtype = _DTYPE_MAP.get(self._config.model_dtype, torch.float16)

        if arch == ModelArchitecture.SD15:
            self._load_sd15(path, dtype)
        elif arch in (ModelArchitecture.SDXL, ModelArchitecture.SDXL_REFINER):
            self._load_sdxl(path, dtype)
        else:
            logger.warning(
                "Full weight loading not yet implemented for %s — "
                "architecture detected but generation will use fallbacks",
                arch.value,
            )

        self._model_loaded = True

    def unload_all(self) -> None:
        """Unload all models and clear caches."""
        self._model_manager.unload_all()
        self._cache.clear()
        self._model_config = None
        self._model_loaded = False
        self._unet = None
        self._text_encoder = None
        self._tokenizer = None
        self._vae_decoder = None
        self._log_sigmas = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    # Model loading backends
    # ------------------------------------------------------------------

    def _load_sd15(self, path: str, dtype: torch.dtype) -> None:
        """Load SD1.5 checkpoint via diffusers from_single_file."""
        from diffusers import StableDiffusionPipeline

        from serenity.inference.vae.decoder import VAEDecoder

        logger.info("Loading SD1.5 pipeline from %s", path)
        pipe = StableDiffusionPipeline.from_single_file(
            path, torch_dtype=dtype, safety_checker=None,
        )

        self._unet = pipe.unet.to(self._device)
        self._unet.eval()

        self._text_encoder = pipe.text_encoder.to(self._device)
        self._text_encoder.eval()

        self._tokenizer = pipe.tokenizer

        vae = pipe.vae.to(self._device)
        vae.eval()
        self._vae_decoder = VAEDecoder(
            vae_model=vae, dtype=dtype,
            device=str(self._device), scaling_factor=0.18215,
        )

        self._build_sigma_schedule(pipe.scheduler.alphas_cumprod)

        del pipe
        torch.cuda.empty_cache()
        logger.info(
            "SD1.5 loaded: UNet + CLIP + VAE on %s (%s)", self._device, dtype,
        )

    def _load_sdxl(self, path: str, dtype: torch.dtype) -> None:
        """Load SDXL checkpoint via diffusers from_single_file."""
        from diffusers import StableDiffusionXLPipeline

        from serenity.inference.vae.decoder import VAEDecoder

        logger.info("Loading SDXL pipeline from %s", path)
        pipe = StableDiffusionXLPipeline.from_single_file(
            path, torch_dtype=dtype,
        )

        self._unet = pipe.unet.to(self._device)
        self._unet.eval()

        self._text_encoder = pipe.text_encoder.to(self._device)
        self._text_encoder.eval()

        self._tokenizer = pipe.tokenizer

        vae = pipe.vae.to(self._device)
        vae.eval()
        self._vae_decoder = VAEDecoder(
            vae_model=vae, dtype=dtype,
            device=str(self._device), scaling_factor=0.13025,
        )

        self._build_sigma_schedule(pipe.scheduler.alphas_cumprod)

        del pipe
        torch.cuda.empty_cache()
        logger.info(
            "SDXL loaded: UNet + CLIP + VAE on %s (%s)", self._device, dtype,
        )

    def _build_sigma_schedule(self, alphas_cumprod: Tensor) -> None:
        """Build log-sigma lookup table from alphas_cumprod."""
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self._log_sigmas = sigmas.log().to(self._device)
        self._sigma_min = float(sigmas[sigmas > 0].min())
        self._sigma_max = float(sigmas.max())
        logger.debug(
            "Sigma schedule: %d steps, min=%.4f, max=%.4f",
            len(sigmas), self._sigma_min, self._sigma_max,
        )

    @staticmethod
    def _sigma_to_discrete(sigma: Tensor, log_sigmas: Tensor) -> Tensor:
        """Convert continuous sigma to discrete timestep via log-sigma lookup."""
        log_sigma = sigma.reshape(-1).log()
        dists = (log_sigma.unsqueeze(1) - log_sigmas.unsqueeze(0)).abs()
        t = dists.argmin(dim=-1).float()
        return t.reshape(sigma.shape)

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
        cache_key = hashlib.sha256(f"{prompt}||{negative_prompt}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Using cached text encoding for prompt=%r", prompt[:50])
            return cached

        if self._text_encoder is not None and self._tokenizer is not None:
            cond_emb = self._run_text_encoder(prompt)
            # CFG always needs unconditional — use empty string if not provided
            uncond_emb = self._run_text_encoder(negative_prompt or "")
        else:
            logger.debug("No text encoder loaded, using dummy conditioning")
            cond_emb = torch.zeros(1, 77, 768, device=self._device)
            uncond_emb = torch.zeros(1, 77, 768, device=self._device)

        conditioning = Conditioning(cond=cond_emb, uncond=uncond_emb)

        if self._config.cache_text_encodings:
            self._cache.put(cache_key, conditioning)

        return conditioning

    def _run_text_encoder(self, text: str) -> Tensor:
        """Tokenize and encode text through the loaded CLIP text encoder."""
        tokens = self._tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self._device)
        with torch.no_grad():
            output = self._text_encoder(input_ids)
        return output.last_hidden_state

    def _create_denoise_fn(
        self,
        model: Any | None,
        prediction: Any,
        cfg_scale: float,
        cond: Tensor | None,
        uncond: Tensor | None,
    ) -> Callable:
        """Create the denoising function for the sampler.

        When a real diffusers UNet is loaded, creates a custom function
        that handles sigma-to-discrete timestep conversion and the
        diffusers ``encoder_hidden_states`` call signature.
        """
        if model is not None and self._log_sigmas is not None:
            log_sigmas = self._log_sigmas
            rescale_phi = self._config.rescale_cfg
            use_mahiro = self._config.mahiro
            model_dtype = next(model.parameters()).dtype

            def denoise_fn(x: Tensor, sigma: Tensor) -> Tensor:
                # c_in scaling for model input; denoised uses raw x
                model_input = prediction.calculate_input(sigma, x)
                timestep = InferenceEngine._sigma_to_discrete(sigma, log_sigmas)
                inp = model_input.to(dtype=model_dtype)

                with torch.no_grad():
                    cond_out = model(
                        inp, timestep, encoder_hidden_states=cond,
                    ).sample.to(x.dtype)
                cond_denoised = prediction.calculate_denoised(
                    sigma, cond_out, x,
                )

                if uncond is None or cfg_scale == 1.0:
                    return cond_denoised

                with torch.no_grad():
                    uncond_out = model(
                        inp, timestep, encoder_hidden_states=uncond,
                    ).sample.to(x.dtype)
                uncond_denoised = prediction.calculate_denoised(
                    sigma, uncond_out, x,
                )

                return apply_cfg(
                    cond_denoised, uncond_denoised, cfg_scale,
                    rescale_phi=rescale_phi, mahiro=use_mahiro,
                )

            return denoise_fn

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
