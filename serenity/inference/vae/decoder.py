"""VAE latent-to-pixel decoder with optional tiled decoding."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "VAEDecoder",
]

logger = logging.getLogger(__name__)

# Default VAE scaling factor (SD 1.5 / SDXL legacy value).
DEFAULT_SCALING_FACTOR: float = 0.18215


class VAEDecoder:
    """Decode latent tensors to pixel space using a VAE model.

    Supports both standard full-image decoding and tiled decoding for
    images that would otherwise exceed GPU memory.

    Args:
        vae_model: Pre-loaded VAE ``nn.Module``, or ``None`` to load later.
        dtype: Computation dtype for the VAE forward pass.
        device: Target device (``"cpu"``, ``"cuda"``, etc.).
        scaling_factor: Latent scaling factor applied before decoding.
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
                When a path is given the model is loaded via
                ``torch.load`` (CPU, weights-only).
        """
        if isinstance(model_or_path, nn.Module):
            self._model = model_or_path
        else:
            sd = torch.load(model_or_path, map_location="cpu", weights_only=True)
            if hasattr(sd, "eval"):
                # Already a module
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
    # Decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def decode(
        self,
        latents: torch.Tensor,
        tiling: bool = False,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> torch.Tensor:
        """Decode latent tensor to pixel space.

        Args:
            latents: Latent tensor of shape ``(B, C, H, W)``.
            tiling: When ``True``, decode in overlapping tiles to save
                memory on high-resolution images.
            tile_size: Tile size **in latent space** when tiling.
            overlap: Overlap between tiles **in latent space**.

        Returns:
            Pixel tensor in ``[0, 1]`` range, shape
            ``(B, 3, H * scale, W * scale)``.
        """
        if self._model is None:
            raise RuntimeError("VAE model not loaded. Call load() first.")

        latents = latents.to(device=self._device, dtype=self._dtype)
        scaled = latents / self.scaling_factor

        if tiling:
            pixels = self._decode_tiled(scaled, tile_size, overlap)
        else:
            pixels = self._decode_single(scaled)

        # Clamp to [0, 1]
        return pixels.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_single(self, latents: torch.Tensor) -> torch.Tensor:
        """Standard full-image decode."""
        result = self._model.decode(latents)  # type: ignore[union-attr]
        # diffusers returns DecoderOutput; raw models return a tensor
        if hasattr(result, "sample"):
            result = result.sample
        # Normalise from [-1, 1] to [0, 1]
        return (result + 1.0) / 2.0

    def _decode_tiled(
        self,
        latents: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """Tiled decode: split latents, decode each tile, blend seams."""
        _b, _c, h, w = latents.shape
        stride = tile_size - overlap

        # Collect tiles and their positions
        tiles: list[torch.Tensor] = []
        positions: list[tuple[int, int, int, int]] = []  # (y, x, th, tw)

        y = 0
        while y < h:
            x = 0
            th = min(tile_size, h - y)
            while x < w:
                tw = min(tile_size, w - x)
                tile_latent = latents[:, :, y : y + th, x : x + tw]
                decoded = self._decode_single(tile_latent)
                tiles.append(decoded)
                positions.append((y, x, th, tw))
                x += stride
                if x >= w:
                    break
            y += stride
            if y >= h:
                break

        # Determine output scale from first tile
        scale_h = tiles[0].shape[2] // positions[0][2]
        scale_w = tiles[0].shape[3] // positions[0][3]
        full_shape = (latents.shape[0], 3, h * scale_h, w * scale_w)

        return _blend_tiles(tiles, positions, full_shape, overlap, scale_h, scale_w)


def _blend_tiles(
    tiles: list[torch.Tensor],
    positions: list[tuple[int, int, int, int]],
    full_shape: tuple[int, int, int, int],
    overlap: int,
    scale_h: int = 8,
    scale_w: int = 8,
) -> torch.Tensor:
    """Blend decoded tiles using linear gradients at overlap regions.

    Args:
        tiles: List of decoded tile tensors.
        positions: ``(y, x, tile_h, tile_w)`` in *latent* space.
        full_shape: ``(B, C, H_px, W_px)`` of the output image.
        overlap: Overlap size in latent space.
        scale_h: Vertical upscale factor from latent to pixel space.
        scale_w: Horizontal upscale factor from latent to pixel space.

    Returns:
        Blended full-resolution image tensor.
    """
    output = torch.zeros(full_shape, dtype=tiles[0].dtype, device=tiles[0].device)
    weight = torch.zeros(
        (1, 1, full_shape[2], full_shape[3]),
        dtype=tiles[0].dtype,
        device=tiles[0].device,
    )

    overlap_px_h = overlap * scale_h
    overlap_px_w = overlap * scale_w

    for tile, (ly, lx, lth, ltw) in zip(tiles, positions):
        py = ly * scale_h
        px = lx * scale_w
        ph = lth * scale_h
        pw = ltw * scale_w

        # Build per-tile weight mask with linear ramps on overlap edges
        tile_weight = torch.ones(
            (1, 1, ph, pw), dtype=tile.dtype, device=tile.device
        )

        # Top edge ramp
        if ly > 0 and overlap_px_h > 0:
            ramp = torch.linspace(0.0, 1.0, overlap_px_h, device=tile.device, dtype=tile.dtype)
            tile_weight[:, :, :overlap_px_h, :] *= ramp.view(1, 1, -1, 1)

        # Bottom edge ramp
        if py + ph < full_shape[2] and overlap_px_h > 0:
            ramp = torch.linspace(1.0, 0.0, overlap_px_h, device=tile.device, dtype=tile.dtype)
            tile_weight[:, :, -overlap_px_h:, :] *= ramp.view(1, 1, -1, 1)

        # Left edge ramp
        if lx > 0 and overlap_px_w > 0:
            ramp = torch.linspace(0.0, 1.0, overlap_px_w, device=tile.device, dtype=tile.dtype)
            tile_weight[:, :, :, :overlap_px_w] *= ramp.view(1, 1, 1, -1)

        # Right edge ramp
        if px + pw < full_shape[3] and overlap_px_w > 0:
            ramp = torch.linspace(1.0, 0.0, overlap_px_w, device=tile.device, dtype=tile.dtype)
            tile_weight[:, :, :, -overlap_px_w:] *= ramp.view(1, 1, 1, -1)

        output[:, :, py : py + ph, px : px + pw] += tile * tile_weight
        weight[:, :, py : py + ph, px : px + pw] += tile_weight

    # Avoid division by zero
    weight = weight.clamp(min=1e-8)
    return output / weight
