"""VAE latent-to-pixel decoder with optional tiled decoding.

Supports 2D image latents ``(B, C, H, W)`` and 3D video latents
``(B, C, T, H, W)`` with spatial tiling, temporal chunking, and
multi-pass offset averaging for artifact reduction.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VAEDecoder",
]

logger = logging.getLogger(__name__)

# Default VAE scaling factor (SD 1.5 / SDXL legacy value).
DEFAULT_SCALING_FACTOR: float = 0.18215


class VAEDecoder:
    """Decode latent tensors to pixel space using a VAE model.

    Supports both standard full-image decoding and tiled decoding for
    images that would otherwise exceed GPU memory.  For video models the
    decoder handles 5-D latents ``(B, C, T, H, W)`` with spatial tiling
    and optional temporal chunking.

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
        multipass: bool = False,
        temporal_chunk: int = 0,
    ) -> torch.Tensor:
        """Decode latent tensor to pixel space.

        Args:
            latents: Latent tensor of shape ``(B, C, H, W)`` for images or
                ``(B, C, T, H, W)`` for video.
            tiling: When ``True``, decode in overlapping tiles to save
                memory on high-resolution images.
            tile_size: Tile size **in latent space** when tiling.
            overlap: Overlap between tiles **in latent space**.
            multipass: When ``True``, use multi-pass offset averaging
                (3 passes) to reduce tiling artefacts.  Only effective
                when *tiling* is also ``True``.
            temporal_chunk: For 3D (video) latents, the number of
                temporal frames per chunk.  ``0`` means process the full
                temporal extent at once.

        Returns:
            Pixel tensor in ``[0, 1]`` range.  Shape is
            ``(B, 3, H_px, W_px)`` for 2D inputs or
            ``(B, 3, T_out, H_px, W_px)`` for 3D inputs.
        """
        if self._model is None:
            raise RuntimeError("VAE model not loaded. Call load() first.")

        latents = latents.to(device=self._device, dtype=self._dtype)
        scaled = latents / self.scaling_factor
        is_3d = scaled.ndim == 5

        if tiling and multipass:
            pixels = self._decode_tiled_multipass(
                scaled, tile_size, overlap, temporal_chunk=temporal_chunk,
            )
        elif tiling:
            if is_3d:
                pixels = self._decode_tiled_3d(
                    scaled, tile_size, overlap, temporal_chunk,
                )
            else:
                pixels = self._decode_tiled(scaled, tile_size, overlap)
        else:
            try:
                if is_3d:
                    pixels = self._decode_single_3d(scaled)
                else:
                    pixels = self._decode_single(scaled)
            except torch.cuda.OutOfMemoryError:
                logger.warning("VAE decode OOM -- falling back to tiled decoding")
                torch.cuda.empty_cache()
                if is_3d:
                    pixels = self._decode_tiled_3d(scaled, tile_size, overlap)
                else:
                    pixels = self._decode_tiled(scaled, tile_size, overlap)

        # Clamp to [0, 1]
        return pixels.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers -- 2D
    # ------------------------------------------------------------------

    def _decode_single(self, latents: torch.Tensor) -> torch.Tensor:
        """Standard full-image decode for 2D latents."""
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

    # ------------------------------------------------------------------
    # Internal helpers -- 3D (video)
    # ------------------------------------------------------------------

    def _decode_single_3d(self, latents: torch.Tensor) -> torch.Tensor:
        """Full decode for 3D video latents ``(B, C, T, H, W)``.

        Attempts a native 3D decode first.  If the VAE raises on the 5-D
        input (i.e. it is a 2D-only VAE) we fall back to decoding each
        temporal frame independently.
        """
        try:
            result = self._model.decode(latents)  # type: ignore[union-attr]
            if hasattr(result, "sample"):
                result = result.sample
            return (result + 1.0) / 2.0
        except (RuntimeError, TypeError):
            # 2D-only VAE -- reshape to process frame-by-frame
            return self._decode_3d_frame_by_frame(latents)

    def _decode_3d_frame_by_frame(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode 3D latents frame-by-frame using a 2D VAE.

        Args:
            latents: ``(B, C, T, H, W)`` tensor.

        Returns:
            ``(B, 3, T, H_px, W_px)`` tensor.
        """
        b, c, t, h, w = latents.shape
        # Reshape (B, C, T, H, W) -> (B*T, C, H, W) for 2D decode
        flat = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        decoded = self._decode_single(flat)  # (B*T, 3, H_px, W_px)
        _, c_out, h_px, w_px = decoded.shape
        # Reshape back to (B, 3, T, H_px, W_px)
        return decoded.reshape(b, t, c_out, h_px, w_px).permute(0, 2, 1, 3, 4)

    def _decode_tiled_3d(
        self,
        latents: torch.Tensor,
        tile_size: int,
        overlap: int,
        temporal_chunk: int = 0,
    ) -> torch.Tensor:
        """Tiled 3D decode for video latents.

        Tiles across spatial dimensions (H, W) and optionally chunks the
        temporal dimension.  Spatial seams are blended using the same
        linear gradient approach as the 2D path.

        Args:
            latents: ``(B, C, T, H, W)`` tensor.
            tile_size: Spatial tile size in latent space.
            overlap: Spatial overlap in latent space.
            temporal_chunk: Temporal chunk size.  ``0`` means process the
                full temporal extent at once.

        Returns:
            ``(B, 3, T_out, H_px, W_px)`` tensor.
        """
        b, c, t, h, w = latents.shape

        if temporal_chunk <= 0 or temporal_chunk >= t:
            return self._decode_tiled_3d_spatial(latents, tile_size, overlap)

        # Process in temporal chunks and concatenate along T
        chunks: list[torch.Tensor] = []
        t_start = 0
        while t_start < t:
            t_end = min(t_start + temporal_chunk, t)
            chunk_latent = latents[:, :, t_start:t_end, :, :]
            decoded_chunk = self._decode_tiled_3d_spatial(
                chunk_latent, tile_size, overlap,
            )
            chunks.append(decoded_chunk)
            t_start = t_end

        return torch.cat(chunks, dim=2)

    def _decode_tiled_3d_spatial(
        self,
        latents: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """Spatially tiled decode for a single temporal chunk.

        For each spatial tile position, slices ``(B, C, T, tile_h, tile_w)``
        from the latents, decodes, and blends tiles spatially.

        Args:
            latents: ``(B, C, T, H, W)`` tensor (possibly a temporal
                sub-chunk of the full video).
            tile_size: Spatial tile size in latent space.
            overlap: Spatial overlap in latent space.

        Returns:
            ``(B, 3, T_out, H_px, W_px)`` tensor.
        """
        b, c, t, h, w = latents.shape
        stride = tile_size - overlap

        tiles: list[torch.Tensor] = []
        positions: list[tuple[int, int, int, int]] = []  # (y, x, th, tw)

        y = 0
        while y < h:
            x = 0
            th = min(tile_size, h - y)
            while x < w:
                tw = min(tile_size, w - x)
                tile_latent = latents[:, :, :, y : y + th, x : x + tw]
                decoded = self._decode_single_3d(tile_latent)
                tiles.append(decoded)
                positions.append((y, x, th, tw))
                x += stride
                if x >= w:
                    break
            y += stride
            if y >= h:
                break

        # Determine pixel-space scale factors from the first tile
        # tiles[0] shape: (B, 3, T_out, H_px_tile, W_px_tile)
        first_tile = tiles[0]
        t_out = first_tile.shape[2]
        scale_h = first_tile.shape[3] // positions[0][2]
        scale_w = first_tile.shape[4] // positions[0][3]

        full_h = h * scale_h
        full_w = w * scale_w
        output = torch.zeros(
            (b, 3, t_out, full_h, full_w),
            dtype=first_tile.dtype,
            device=first_tile.device,
        )
        weight = torch.zeros(
            (1, 1, 1, full_h, full_w),
            dtype=first_tile.dtype,
            device=first_tile.device,
        )

        overlap_px_h = overlap * scale_h
        overlap_px_w = overlap * scale_w

        for tile, (ly, lx, lth, ltw) in zip(tiles, positions):
            py = ly * scale_h
            px_pos = lx * scale_w
            ph = lth * scale_h
            pw = ltw * scale_w

            # Build per-tile spatial weight mask (broadcast over T)
            tile_weight = torch.ones(
                (1, 1, 1, ph, pw), dtype=tile.dtype, device=tile.device,
            )

            # Top edge ramp
            if ly > 0 and overlap_px_h > 0:
                ramp = torch.linspace(
                    0.0, 1.0, overlap_px_h,
                    device=tile.device, dtype=tile.dtype,
                )
                tile_weight[:, :, :, :overlap_px_h, :] *= ramp.view(1, 1, 1, -1, 1)

            # Bottom edge ramp
            if py + ph < full_h and overlap_px_h > 0:
                ramp = torch.linspace(
                    1.0, 0.0, overlap_px_h,
                    device=tile.device, dtype=tile.dtype,
                )
                tile_weight[:, :, :, -overlap_px_h:, :] *= ramp.view(1, 1, 1, -1, 1)

            # Left edge ramp
            if lx > 0 and overlap_px_w > 0:
                ramp = torch.linspace(
                    0.0, 1.0, overlap_px_w,
                    device=tile.device, dtype=tile.dtype,
                )
                tile_weight[:, :, :, :, :overlap_px_w] *= ramp.view(1, 1, 1, 1, -1)

            # Right edge ramp
            if px_pos + pw < full_w and overlap_px_w > 0:
                ramp = torch.linspace(
                    1.0, 0.0, overlap_px_w,
                    device=tile.device, dtype=tile.dtype,
                )
                tile_weight[:, :, :, :, -overlap_px_w:] *= ramp.view(1, 1, 1, 1, -1)

            output[:, :, :, py : py + ph, px_pos : px_pos + pw] += tile * tile_weight
            weight[:, :, :, py : py + ph, px_pos : px_pos + pw] += tile_weight

        weight = weight.clamp(min=1e-8)
        return output / weight

    # ------------------------------------------------------------------
    # Internal helpers -- multi-pass
    # ------------------------------------------------------------------

    def _decode_tiled_multipass(
        self,
        latents: torch.Tensor,
        tile_size: int,
        overlap: int,
        num_passes: int = 3,
        temporal_chunk: int = 0,
    ) -> torch.Tensor:
        """Multi-pass tiled decode with offset averaging.

        Produces multiple tiled decodes from different grid alignments
        and averages the results to reduce tiling artefacts.

        Pass layout:
        - Pass 1: Standard grid (no offset).
        - Pass 2: Grid offset by ``tile_size // 2`` horizontally.
        - Pass 3: Grid offset by ``tile_size // 2`` vertically.

        Args:
            latents: ``(B, C, H, W)`` or ``(B, C, T, H, W)`` tensor.
            tile_size: Spatial tile size in latent space.
            overlap: Spatial overlap in latent space.
            num_passes: Number of offset passes (1--3).
            temporal_chunk: For 3D latents, temporal chunk size (0 = full).

        Returns:
            Averaged decoded pixel tensor.
        """
        is_3d = latents.ndim == 5
        half = tile_size // 2

        offsets: list[tuple[int, int]] = [(0, 0)]
        if num_passes >= 2:
            offsets.append((0, half))  # horizontal offset
        if num_passes >= 3:
            offsets.append((half, 0))  # vertical offset

        results: list[torch.Tensor] = []

        for dy, dx in offsets:
            if dy > 0 or dx > 0:
                # Pad latents in spatial dimensions to accommodate offset.
                # For 5-D: (B, C, T, H, W) pad order is (W_left, W_right, H_top, H_bottom)
                # For 4-D: (B, C, H, W) same pad order
                padded = F.pad(latents, (dx, 0, dy, 0), mode="reflect")
            else:
                padded = latents

            if is_3d:
                decoded = self._decode_tiled_3d(
                    padded, tile_size, overlap, temporal_chunk,
                )
            else:
                decoded = self._decode_tiled(padded, tile_size, overlap)

            if dy > 0 or dx > 0:
                # Determine the pixel-space scale factors so we can crop
                # the padding back out of the decoded result.
                if is_3d:
                    padded_h = padded.shape[3]
                    padded_w = padded.shape[4]
                    dec_h = decoded.shape[3]
                    dec_w = decoded.shape[4]
                else:
                    padded_h = padded.shape[2]
                    padded_w = padded.shape[3]
                    dec_h = decoded.shape[2]
                    dec_w = decoded.shape[3]

                scale_h = dec_h // padded_h
                scale_w = dec_w // padded_w
                crop_y = dy * scale_h
                crop_x = dx * scale_w

                if is_3d:
                    decoded = decoded[:, :, :, crop_y:, crop_x:]
                else:
                    decoded = decoded[:, :, crop_y:, crop_x:]

            results.append(decoded)

        return torch.stack(results).mean(dim=0)


# ------------------------------------------------------------------
# Module-level blending helper (2D)
# ------------------------------------------------------------------


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
