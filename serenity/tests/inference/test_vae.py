"""Tests for VAE encoder and decoder with tiling support."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from serenity.inference.vae.decoder import VAEDecoder, _blend_tiles
from serenity.inference.vae.encoder import VAEEncoder


# ---------------------------------------------------------------------------
# Mock VAE model
# ---------------------------------------------------------------------------


class _MockVAEDecode(nn.Module):
    """Minimal VAE decoder mock: returns a fixed-shape tensor."""

    def __init__(self, scale: int = 8) -> None:
        super().__init__()
        self.scale = scale
        # Need at least one parameter so .to() works
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = latents.shape
        # Return pixel tensor normalised to [-1, 1] (decoder will shift)
        return torch.zeros(b, 3, h * self.scale, w * self.scale, dtype=latents.dtype, device=latents.device)


class _MockVAEEncode(nn.Module):
    """Minimal VAE encoder mock: returns a fixed-shape latent."""

    def __init__(self, scale: int = 8, latent_channels: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.latent_channels = latent_channels
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, image: torch.Tensor) -> MagicMock:
        b, _c, h, w = image.shape
        latent = torch.randn(
            b, self.latent_channels, h // self.scale, w // self.scale,
            dtype=image.dtype, device=image.device,
        )
        result = MagicMock()
        result.latent_dist.sample.return_value = latent
        return result


# ---------------------------------------------------------------------------
# VAEDecoder tests
# ---------------------------------------------------------------------------


class TestVAEDecoder:
    """Tests for the VAE decoder."""

    def test_is_loaded_false_initially(self) -> None:
        dec = VAEDecoder()
        assert dec.is_loaded is False

    def test_is_loaded_after_load(self) -> None:
        dec = VAEDecoder()
        dec.load(_MockVAEDecode())
        assert dec.is_loaded is True

    def test_unload(self) -> None:
        dec = VAEDecoder()
        dec.load(_MockVAEDecode())
        dec.unload()
        assert dec.is_loaded is False

    def test_decode_raises_when_not_loaded(self) -> None:
        dec = VAEDecoder()
        with pytest.raises(RuntimeError, match="not loaded"):
            dec.decode(torch.randn(1, 4, 64, 64))

    def test_decode_standard_shape(self) -> None:
        dec = VAEDecoder(dtype=torch.float32)
        dec.load(_MockVAEDecode(scale=8))
        latents = torch.randn(1, 4, 64, 64)
        pixels = dec.decode(latents)
        assert pixels.shape == (1, 3, 512, 512)

    def test_decode_output_range(self) -> None:
        dec = VAEDecoder(dtype=torch.float32)
        dec.load(_MockVAEDecode(scale=8))
        latents = torch.randn(1, 4, 32, 32)
        pixels = dec.decode(latents)
        assert pixels.min() >= 0.0
        assert pixels.max() <= 1.0

    def test_decode_tiled_shape(self) -> None:
        """Tiled decode should produce the same spatial dimensions."""
        dec = VAEDecoder(dtype=torch.float32)
        dec.load(_MockVAEDecode(scale=8))
        latents = torch.randn(1, 4, 64, 64)
        pixels = dec.decode(latents, tiling=True, tile_size=32, overlap=8)
        assert pixels.shape == (1, 3, 512, 512)

    def test_decode_tiled_output_range(self) -> None:
        dec = VAEDecoder(dtype=torch.float32)
        dec.load(_MockVAEDecode(scale=8))
        latents = torch.randn(1, 4, 64, 64)
        pixels = dec.decode(latents, tiling=True, tile_size=32, overlap=8)
        assert pixels.min() >= 0.0
        assert pixels.max() <= 1.0

    def test_custom_scaling_factor(self) -> None:
        dec = VAEDecoder(dtype=torch.float32, scaling_factor=0.3611)
        dec.load(_MockVAEDecode(scale=8))
        assert dec.scaling_factor == pytest.approx(0.3611)

    def test_batch_decode(self) -> None:
        dec = VAEDecoder(dtype=torch.float32)
        dec.load(_MockVAEDecode(scale=8))
        latents = torch.randn(2, 4, 32, 32)
        pixels = dec.decode(latents)
        assert pixels.shape == (2, 3, 256, 256)


# ---------------------------------------------------------------------------
# _blend_tiles tests
# ---------------------------------------------------------------------------


class TestBlendTiles:
    """Tests for the tile blending helper."""

    def test_single_tile_passthrough(self) -> None:
        """A single tile covering the full image should be returned as-is."""
        tile = torch.ones(1, 3, 64, 64)
        positions = [(0, 0, 8, 8)]  # latent coords
        full_shape = (1, 3, 64, 64)
        result = _blend_tiles([tile], positions, full_shape, overlap=0, scale_h=8, scale_w=8)
        assert result.shape == full_shape
        # Single tile, no overlap -> output should equal input
        assert torch.allclose(result, tile, atol=1e-6)

    def test_two_tiles_blend_produces_valid_output(self) -> None:
        """Two overlapping tiles should blend smoothly."""
        tile_a = torch.ones(1, 3, 64, 64)
        tile_b = torch.ones(1, 3, 64, 64) * 0.5
        # tile_a at latent (0,0), tile_b at latent (0,4) with 4 latent overlap
        positions = [(0, 0, 8, 8), (0, 4, 8, 8)]
        full_shape = (1, 3, 64, 96)  # 12 latent width * 8
        result = _blend_tiles(
            [tile_a, tile_b], positions, full_shape, overlap=4, scale_h=8, scale_w=8,
        )
        assert result.shape == full_shape
        # No NaN or Inf
        assert torch.isfinite(result).all()

    def test_blend_gradient_smoothness(self) -> None:
        """Overlap regions should produce values between the two tiles."""
        tile_a = torch.ones(1, 3, 32, 32) * 1.0
        tile_b = torch.ones(1, 3, 32, 32) * 0.0
        # 2 latent overlap -> 16 pixel overlap
        positions = [(0, 0, 4, 4), (0, 2, 4, 4)]
        full_shape = (1, 3, 32, 48)  # 6 latent * 8
        result = _blend_tiles(
            [tile_a, tile_b], positions, full_shape, overlap=2, scale_h=8, scale_w=8,
        )
        # In the overlap region the blended values should be between 0 and 1
        overlap_region = result[:, :, :, 16:32]
        assert overlap_region.min() >= -1e-6
        assert overlap_region.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# VAEEncoder tests
# ---------------------------------------------------------------------------


class TestVAEEncoder:
    """Tests for the VAE encoder."""

    def test_is_loaded_false_initially(self) -> None:
        enc = VAEEncoder()
        assert enc.is_loaded is False

    def test_is_loaded_after_load(self) -> None:
        enc = VAEEncoder()
        enc.load(_MockVAEEncode())
        assert enc.is_loaded is True

    def test_unload(self) -> None:
        enc = VAEEncoder()
        enc.load(_MockVAEEncode())
        enc.unload()
        assert enc.is_loaded is False

    def test_encode_raises_when_not_loaded(self) -> None:
        enc = VAEEncoder()
        with pytest.raises(RuntimeError, match="not loaded"):
            enc.encode(torch.randn(1, 3, 512, 512))

    def test_encode_standard_shape(self) -> None:
        enc = VAEEncoder(dtype=torch.float32)
        enc.load(_MockVAEEncode(scale=8, latent_channels=4))
        image = torch.rand(1, 3, 512, 512)
        latents = enc.encode(image)
        assert latents.shape == (1, 4, 64, 64)

    def test_encode_batch(self) -> None:
        enc = VAEEncoder(dtype=torch.float32)
        enc.load(_MockVAEEncode(scale=8, latent_channels=4))
        image = torch.rand(2, 3, 256, 256)
        latents = enc.encode(image)
        assert latents.shape == (2, 4, 32, 32)

    def test_encode_tiled_shape(self) -> None:
        enc = VAEEncoder(dtype=torch.float32)
        enc.load(_MockVAEEncode(scale=8, latent_channels=4))
        image = torch.rand(1, 3, 512, 512)
        latents = enc.encode(image, tiling=True, tile_size=256, overlap=64)
        assert latents.shape == (1, 4, 64, 64)

    def test_custom_scaling_factor(self) -> None:
        enc = VAEEncoder(dtype=torch.float32, scaling_factor=0.3611)
        assert enc.scaling_factor == pytest.approx(0.3611)
