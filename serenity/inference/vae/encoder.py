"""VAE pixel-to-latent encoder with optional tiled encoding."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

__all__ = [
    "VAEEncoder",
]

logger = logging.getLogger(__name__)

# Default VAE scaling factor (SD 1.5 / SDXL legacy value).
DEFAULT_SCALING_FACTOR: float = 0.18215


class VAEEncoder:
    """Encode pixel-space images into latent tensors using a VAE model.

    Supports both standard full-image encoding and tiled encoding for
    images that would otherwise exceed GPU memory.

    Args:
        vae_model: Pre-loaded VAE ``nn.Module``, or ``None`` to load later.
        dtype: Computation dtype for the VAE forward pass.
        device: Target device (``"cpu"``, ``"cuda"``, etc.).
        scaling_factor: Latent scaling factor applied after encoding.
    """

    def __init__(
        self,
        vae_model: nn.Module | None = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ) -> None:
        self._model: nn.Module | None = vae_model
        self._dtype = dtype
        self._device = device
        self.scaling_factor = scaling_factor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_or_path: nn.Module | str) -> None:
        """Load or assign a VAE model.

        Args:
            model_or_path: An ``nn.Module`` instance or a filesystem path.
        """
        if isinstance(model_or_path, nn.Module):
            self._model = model_or_path
        else:
            sd = torch.load(model_or_path, map_location="cpu", weights_only=True)
            if hasattr(sd, "eval"):
                self._model = sd
            else:
                raise ValueError(
                    "load() received a state-dict path but needs a full model. "
                    "Use load_vae() from serenity.inference.models.loader instead."
                )
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

    def unload(self) -> None:
        """Release the VAE model and free memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when a VAE model is available."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(
        self,
        image: torch.Tensor,
        tiling: bool = False,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> torch.Tensor:
        """Encode a pixel image to latent space.

        Args:
            image: Pixel tensor in ``[0, 1]`` range, shape
                ``(B, 3, H, W)``.
            tiling: When ``True``, encode in overlapping tiles.
            tile_size: Tile size **in pixel space** when tiling.
            overlap: Overlap between tiles **in pixel space**.

        Returns:
            Latent tensor of shape ``(B, C, H // scale, W // scale)``.
        """
        if self._model is None:
            raise RuntimeError("VAE model not loaded. Call load() first.")

        image = image.to(device=self._device, dtype=self._dtype)
        # Scale pixels from [0, 1] to [-1, 1]
        scaled = image * 2.0 - 1.0

        if tiling:
            latents = self._encode_tiled(scaled, tile_size, overlap)
        else:
            latents = self._encode_single(scaled)

        return latents * self.scaling_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_single(self, image: torch.Tensor) -> torch.Tensor:
        """Standard full-image encode."""
        result = self._model.encode(image)  # type: ignore[union-attr]
        # diffusers returns an EncoderOutput with latent_dist
        if hasattr(result, "latent_dist"):
            return result.latent_dist.sample()
        # Raw models may return a tensor or object with .sample attribute
        if hasattr(result, "sample"):
            return result.sample
        return result

    def _encode_tiled(
        self,
        image: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """Tiled encode: split image, encode each tile, stitch latents."""
        _b, _c, h, w = image.shape
        stride = tile_size - overlap

        # Encode first tile to determine latent scale
        first_tile = image[:, :, :min(tile_size, h), :min(tile_size, w)]
        first_latent = self._encode_single(first_tile)
        scale_h = first_tile.shape[2] // first_latent.shape[2]
        scale_w = first_tile.shape[3] // first_latent.shape[3]
        latent_c = first_latent.shape[1]

        latent_h = h // scale_h
        latent_w = w // scale_w
        output = torch.zeros(
            (_b, latent_c, latent_h, latent_w),
            dtype=first_latent.dtype,
            device=first_latent.device,
        )
        weight = torch.zeros(
            (1, 1, latent_h, latent_w),
            dtype=first_latent.dtype,
            device=first_latent.device,
        )

        overlap_lat_h = overlap // scale_h
        overlap_lat_w = overlap // scale_w

        y = 0
        while y < h:
            x = 0
            th = min(tile_size, h - y)
            while x < w:
                tw = min(tile_size, w - x)
                tile = image[:, :, y : y + th, x : x + tw]
                latent = self._encode_single(tile)

                ly = y // scale_h
                lx = x // scale_w
                lh = latent.shape[2]
                lw = latent.shape[3]

                # Build weight mask with linear overlap blending
                tile_w = torch.ones(
                    (1, 1, lh, lw), dtype=latent.dtype, device=latent.device
                )
                if y > 0 and overlap_lat_h > 0:
                    ramp = torch.linspace(
                        0.0, 1.0, overlap_lat_h,
                        device=latent.device, dtype=latent.dtype,
                    )
                    tile_w[:, :, :overlap_lat_h, :] *= ramp.view(1, 1, -1, 1)
                if y + th < h and overlap_lat_h > 0:
                    ramp = torch.linspace(
                        1.0, 0.0, overlap_lat_h,
                        device=latent.device, dtype=latent.dtype,
                    )
                    tile_w[:, :, -overlap_lat_h:, :] *= ramp.view(1, 1, -1, 1)
                if x > 0 and overlap_lat_w > 0:
                    ramp = torch.linspace(
                        0.0, 1.0, overlap_lat_w,
                        device=latent.device, dtype=latent.dtype,
                    )
                    tile_w[:, :, :, :overlap_lat_w] *= ramp.view(1, 1, 1, -1)
                if x + tw < w and overlap_lat_w > 0:
                    ramp = torch.linspace(
                        1.0, 0.0, overlap_lat_w,
                        device=latent.device, dtype=latent.dtype,
                    )
                    tile_w[:, :, :, -overlap_lat_w:] *= ramp.view(1, 1, 1, -1)

                output[:, :, ly : ly + lh, lx : lx + lw] += latent * tile_w
                weight[:, :, ly : ly + lh, lx : lx + lw] += tile_w

                x += stride
                if x >= w:
                    break
            y += stride
            if y >= h:
                break

        weight = weight.clamp(min=1e-8)
        return output / weight
