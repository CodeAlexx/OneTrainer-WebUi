"""Inference engine orchestrator — ties all subsystems into a single generation API."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from serenity.inference.attention.backends import select_best_backend
from serenity.inference.cache.store import StageCache
from serenity.inference.config import InferenceConfig
from serenity.inference.memory.manager import ModelManager
from serenity.inference.memory.vram import get_free_memory, is_cuda_available
from serenity.inference.models.detection import ModelArchitecture, ModelConfig, detect_from_file
from serenity.inference.models.loader import (
    _get_adapter,
    extract_vae_state_dict,
    load_state_dict,
)
from serenity.inference.sampling.cfg import apply_cfg
from serenity.inference.sampling.conditioning import Conditioning, create_noise
from serenity.inference.sampling.prediction import PredictionType, get_prediction
from serenity.inference.sampling.sampler import SamplerType, create_model_fn, sample
from serenity.inference.sampling.schedulers import SchedulerType, compute_sigmas
from serenity.inference.text.encoders import TextEncoderManager, get_required_encoders
from serenity.inference.vae.decoder import VAEDecoder

__all__ = [
    "GenerationResult",
    "InferenceEngine",
]

logger = logging.getLogger(__name__)


def _extract_model_output(output: Any) -> Tensor:
    """Extract the noise prediction tensor from a model forward pass output.

    Handles both diffusers BaseOutput objects (.sample attribute) and
    raw tensor returns from standalone models.
    """
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unexpected model output type: {type(output)}")


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
        self._cache = StageCache(max_memory_bytes=256 * 1024 * 1024)

        # State — populated lazily on first generate()
        self._model_config: ModelConfig | None = None
        self._model_loaded: bool = False
        self._adapter: Any | None = None
        self._unet: Any | None = None
        self._text_enc_manager: TextEncoderManager = TextEncoderManager()
        self._vae_decoder: VAEDecoder | None = None

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

        # Step 4: Get prediction type from adapter or model config
        prediction_type_str = "eps"
        if self._adapter is not None:
            prediction_type_str = self._adapter.get_prediction_type()
        elif self._model_config is not None:
            prediction_type_str = self._model_config.prediction_type
        prediction = get_prediction(prediction_type_str)

        # Step 5: Create noise for each batch item
        # Get latent dimensions from model architecture
        latent_channels, downscale_factor = self._get_latent_params()
        latent_h = params["height"] // downscale_factor
        latent_w = params["width"] // downscale_factor
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

        # Step 6: Build SDXL added_cond_kwargs if needed
        added_cond_kwargs_cond = None
        added_cond_kwargs_uncond = None
        if conditioning.pooled is not None:
            # SDXL: build time_ids [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
            h, w = params["height"], params["width"]
            time_ids = torch.tensor(
                [[h, w, 0, 0, h, w]],
                dtype=conditioning.pooled.dtype,
                device=self._device,
            )
            added_cond_kwargs_cond = {
                "text_embeds": conditioning.pooled,
                "time_ids": time_ids,
            }
            pooled_uncond = conditioning.extra.get("pooled_uncond")
            if pooled_uncond is not None:
                added_cond_kwargs_uncond = {
                    "text_embeds": pooled_uncond,
                    "time_ids": time_ids,
                }

        # Step 6b: Create model/denoise function
        denoise_fn = self._create_denoise_fn(
            model=self._unet,
            prediction=prediction,
            cfg_scale=params["cfg_scale"],
            cond=conditioning.cond if conditioning is not None else None,
            uncond=conditioning.uncond if conditioning is not None else None,
            added_cond_kwargs_cond=added_cond_kwargs_cond,
            added_cond_kwargs_uncond=added_cond_kwargs_uncond,
        )

        # Step 8: Sample (with OOM recovery — retry at batch_size=1)
        logger.info("Sampling with %s sampler...", params["sampler"])
        try:
            latents = sample(
                model_fn=denoise_fn,
                noise=noise,
                sigmas=sigmas,
                sampler_type=params["sampler"],
                callback=callback,
            )
        except torch.cuda.OutOfMemoryError:
            if noise.shape[0] > 1:
                logger.warning(
                    "CUDA OOM during sampling — retrying with batch_size=1",
                )
                torch.cuda.empty_cache()
                noise = noise[:1]
                seeds = seeds[:1]
                latents = sample(
                    model_fn=denoise_fn,
                    noise=noise,
                    sigmas=sigmas,
                    sampler_type=params["sampler"],
                    callback=callback,
                )
            else:
                raise
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
        UNet/transformer, text encoders, and VAE through the adapter system.
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

        # Resolve adapter from registry
        adapter = _get_adapter(self._model_config)
        if adapter is None:
            logger.warning(
                "No adapter for %s — architecture detected but generation "
                "will use fallbacks",
                arch.value,
            )
            self._model_loaded = True
            return

        self._adapter = adapter

        # 1. Load state dict and create model via adapter
        sd = load_state_dict(path)
        self._unet = adapter.create_model(
            sd, device=str(self._device), dtype=dtype,
        )
        logger.info("Created %s model via adapter", arch.value)

        # 2. Load VAE from checkpoint
        self._load_vae_from_checkpoint(sd, dtype)

        # 3. Load text encoders via TextEncoderManager
        self._text_enc_manager.load_for_model(
            arch, dtype=dtype, device=str(self._device),
        )

        # 4. Build sigma schedule
        if adapter.get_prediction_type() in ("flow", "flow_flux"):
            self._build_flow_sigma_schedule()
        else:
            self._build_ddpm_sigma_schedule()

        self._model_loaded = True
        logger.info(
            "%s loaded: model + text encoders + VAE on %s (%s)",
            arch.value, self._device, dtype,
        )

    def unload_all(self) -> None:
        """Unload all models and clear caches."""
        self._model_manager.unload_all()
        self._cache.invalidate()
        self._model_config = None
        self._model_loaded = False
        self._adapter = None
        self._unet = None
        self._text_enc_manager.unload_all()
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
            "cache_entries": self._cache.stats["entries"],
            "vram_free_bytes": free_mem,
        }

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_vae_from_checkpoint(
        self, state_dict: dict[str, Tensor], dtype: torch.dtype,
    ) -> None:
        """Extract and load VAE weights from a full checkpoint state dict."""
        vae_sd = extract_vae_state_dict(state_dict)
        if not vae_sd:
            logger.info("No VAE keys in checkpoint, skipping VAE loading")
            return

        scaling_factor = 0.18215  # default
        if self._adapter is not None:
            scaling_factor = self._adapter.get_vae_scaling_factor()

        # SDXL VAE should use float32 to avoid NaN/overflow
        vae_dtype = dtype
        if self._model_config is not None and self._model_config.architecture in (
            ModelArchitecture.SDXL, ModelArchitecture.SDXL_REFINER,
        ):
            vae_dtype = torch.float32

        try:
            from diffusers.models import AutoencoderKL  # type: ignore[import-untyped]

            latent_ch = 4
            if "decoder.conv_in.weight" in vae_sd:
                latent_ch = vae_sd["decoder.conv_in.weight"].shape[1]

            vae = AutoencoderKL(latent_channels=latent_ch)
            vae.load_state_dict(vae_sd, strict=False)
            vae = vae.to(device=self._device, dtype=vae_dtype)
            vae.eval()

            self._vae_decoder = VAEDecoder(
                vae_model=vae,
                dtype=vae_dtype,
                device=str(self._device),
                scaling_factor=scaling_factor,
            )
            logger.info(
                "VAE loaded (latent_ch=%d, scaling=%.5f, dtype=%s)",
                latent_ch, scaling_factor, vae_dtype,
            )
        except Exception as exc:
            logger.warning("Failed to load VAE: %s", exc)

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

    def _build_ddpm_sigma_schedule(self) -> None:
        """Build sigma schedule from the standard DDPM linear beta schedule.

        Uses the canonical schedule shared by SD1.5 and SDXL:
        ``beta_start=0.00085``, ``beta_end=0.012``, 1000 steps,
        scaled-linear spacing.
        """
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self._build_sigma_schedule(alphas_cumprod)

    def _build_flow_sigma_schedule(self) -> None:
        """Build sigma schedule for flow matching models (Flux, SD3).

        Flow matching uses linear timesteps from 1 (full noise) to 0.
        The sigma_max=1.0 and sigma_min is a small epsilon.
        """
        self._sigma_min = 1e-4
        self._sigma_max = 1.0
        # For flow matching, sigmas are the timesteps themselves
        sigmas = torch.linspace(1.0, 0.0, 1000)
        self._log_sigmas = sigmas.log().to(self._device)
        logger.debug(
            "Flow sigma schedule: min=%.4f, max=%.4f",
            self._sigma_min, self._sigma_max,
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

    def _get_latent_params(self) -> tuple[int, int]:
        """Return (latent_channels, downscale_factor) for the loaded model."""
        if self._model_config is None:
            return 4, 8  # SD1.5/SDXL defaults
        arch = self._model_config.architecture
        # Flux and SD3 use 16 channels with downscale 8
        if arch in (
            ModelArchitecture.FLUX_DEV, ModelArchitecture.FLUX_SCHNELL,
            ModelArchitecture.SD3, ModelArchitecture.CHROMA,
        ):
            return 16, 8
        # Wan video models use 16 channels
        if arch == ModelArchitecture.WAN:
            return 16, 8
        # Default: SD1.5/SDXL
        return 4, 8

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

        Delegates to :class:`TextEncoderManager` for architecture-specific
        text encoding.  Returns a :class:`Conditioning` with cond and uncond
        tensors.  If no text encoder is loaded, returns dummy conditioning.
        """
        cache_key = hashlib.sha256(f"{prompt}||{negative_prompt}".encode()).hexdigest()
        cached = self._cache.get("text", cache_key)
        if cached is not None:
            logger.debug("Using cached text encoding for prompt=%r", prompt[:50])
            return Conditioning(**cached)

        arch = self._model_config.architecture if self._model_config else None

        # Try TextEncoderManager if we have a known architecture
        if arch is not None and get_required_encoders(arch):
            try:
                enc_result = self._text_enc_manager.encode_for_model(
                    arch, prompt, negative_prompt or "",
                    clip_skip=self._config.clip_skip,
                )
                conditioning = self._conditioning_from_enc_result(enc_result)
            except (ValueError, RuntimeError) as exc:
                logger.debug("TextEncoderManager unavailable: %s", exc)
                conditioning = self._dummy_conditioning()
        else:
            conditioning = self._dummy_conditioning()

        if self._config.cache_text_encodings:
            cache_dict = {
                "cond": conditioning.cond,
                "uncond": conditioning.uncond,
                "pooled": conditioning.pooled,
                "extra": conditioning.extra,
            }
            self._cache.put("text", cache_key, cache_dict)

        return conditioning

    @staticmethod
    def _conditioning_from_enc_result(
        enc_result: dict[str, Any],
    ) -> Conditioning:
        """Convert TextEncoderManager output dict to a Conditioning object."""
        cond = enc_result.get("cond")
        uncond = enc_result.get("uncond")
        pooled = enc_result.get("pooled")

        extra: dict[str, Any] = {}
        neg_pooled = enc_result.get("neg_pooled")
        if neg_pooled is not None:
            extra["pooled_uncond"] = neg_pooled

        # Flux-specific: clip_cond carries CLIP-L hidden states
        clip_cond = enc_result.get("clip_cond")
        if clip_cond is not None:
            extra["clip_cond"] = clip_cond

        return Conditioning(
            cond=cond,
            uncond=uncond,
            pooled=pooled,
            extra=extra,
        )

    def _dummy_conditioning(self) -> Conditioning:
        """Return zero-filled conditioning when no text encoder is available."""
        logger.debug("No text encoder loaded, using dummy conditioning")
        cond_emb = torch.zeros(1, 77, 768, device=self._device)
        uncond_emb = torch.zeros(1, 77, 768, device=self._device)
        return Conditioning(cond=cond_emb, uncond=uncond_emb)

    def _create_denoise_fn(
        self,
        model: Any | None,
        prediction: Any,
        cfg_scale: float,
        cond: Tensor | None,
        uncond: Tensor | None,
        added_cond_kwargs_cond: dict | None = None,
        added_cond_kwargs_uncond: dict | None = None,
    ) -> Callable:
        """Create the denoising function for the sampler.

        When a real diffusers UNet is loaded, creates a custom function
        that handles sigma-to-discrete timestep conversion and the
        diffusers ``encoder_hidden_states`` call signature. For SDXL,
        also passes ``added_cond_kwargs`` with pooled embeds and time_ids.
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

                unet_kwargs: dict[str, Any] = {
                    "encoder_hidden_states": cond,
                }
                if added_cond_kwargs_cond is not None:
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs_cond

                with torch.no_grad():
                    raw_out = model(inp, timestep, **unet_kwargs)
                    cond_out = _extract_model_output(raw_out).to(x.dtype)
                cond_denoised = prediction.calculate_denoised(
                    sigma, cond_out, x,
                )

                if uncond is None or cfg_scale == 1.0:
                    return cond_denoised

                uncond_kwargs: dict[str, Any] = {
                    "encoder_hidden_states": uncond,
                }
                if added_cond_kwargs_uncond is not None:
                    uncond_kwargs["added_cond_kwargs"] = added_cond_kwargs_uncond

                with torch.no_grad():
                    raw_out = model(inp, timestep, **uncond_kwargs)
                    uncond_out = _extract_model_output(raw_out).to(x.dtype)
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
