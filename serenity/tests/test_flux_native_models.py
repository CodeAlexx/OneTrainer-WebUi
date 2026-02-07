"""Unit tests for native Flux model helpers."""

from __future__ import annotations

from serenity.models.flux1 import Flux1Model
from serenity.models.flux2 import Flux2Model

import torch


def test_flux1_pack_unpack_round_trip():
    model = Flux1Model()
    latents = torch.randn(2, 16, 32, 32)

    packed, image_ids = model.pack_latents(latents)
    unpacked = model.unpack_latents(packed, 32, 32)

    assert packed.shape == (2, 256, 64)
    assert image_ids.shape == (256, 3)
    assert unpacked.shape == latents.shape
    assert torch.allclose(unpacked, latents, atol=1e-6)


def test_flux2_patchify_round_trip():
    latents = torch.randn(1, 32, 64, 64)
    patchified = Flux2Model.patchify_latents(latents)
    restored = Flux2Model.unpatchify_latents(patchified)

    assert patchified.shape == (1, 128, 32, 32)
    assert restored.shape == latents.shape
    assert torch.allclose(restored, latents, atol=1e-6)


def test_flux2_pack_unpack_round_trip():
    model = Flux2Model()
    latents = torch.randn(2, 128, 32, 32)

    packed, image_ids = model.pack_latents(latents)
    unpacked = model.unpack_latents(packed, 32, 32)

    assert packed.shape == (2, 1024, 128)
    assert image_ids.shape == (2, 1024, 4)
    assert unpacked.shape == latents.shape
    assert torch.allclose(unpacked, latents, atol=1e-6)


def test_flux2_text_ids_shape():
    prompt_embeds = torch.randn(2, 77, 4096)
    text_ids = Flux2Model.prepare_text_ids(prompt_embeds)
    assert text_ids.shape == (2, 77, 4)
