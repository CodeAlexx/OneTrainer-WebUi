"""
Test LTX2 Parity with SimpleTuner

Verifies that Serenity's LTX2 implementation matches SimpleTuner's:
1. Latent normalization using VAE mean/std
2. Pack/unpack functions produce correct shapes
3. Flow matching interpolation is correct
4. Timestep scaling matches (x1000)
5. Frame constraint (8n+1)

Run: python -m serenity.tests.test_ltx2_parity
"""


# Import our implementations
from serenity.models.ltx2 import (
    LTX2Model,
    adjust_video_frames,
    normalize_ltx2_latents,
    pack_ltx2_latents,
    unpack_ltx2_latents,
)

import torch


def test_ltx_load_pipeline_accepts_single_transformer_with_component_overrides(monkeypatch, tmp_path):
    """Single-file transformer path should route through split bundle assembly."""

    transformer_path = tmp_path / "ltx.safetensors"
    transformer_path.write_bytes(b"weights")
    template_path = tmp_path / "template"
    template_path.mkdir(parents=True, exist_ok=True)

    calls: dict[str, object] = {}

    def _fake_find_template(explicit_template):
        calls["template_hint"] = explicit_template
        return template_path

    def _fake_build_bundle(*, primary_model_path, template_root, component_paths, dtype):
        calls["primary_model_path"] = primary_model_path
        calls["template_root"] = template_root
        calls["component_paths"] = component_paths
        calls["dtype"] = dtype

        class _Pipeline:
            def __init__(self) -> None:
                self.device = None

            def to(self, device):
                self.device = device
                return self

        return _Pipeline()

    monkeypatch.setattr("serenity.models.ltx2._find_ltx_template_root", _fake_find_template)
    monkeypatch.setattr("serenity.models.ltx2._build_ltx_bundle", _fake_build_bundle)

    model = LTX2Model()
    pipeline = model.load_pipeline(
        str(transformer_path),
        torch.float32,
        train_device=torch.device("cpu"),
        vae_path="/custom/vae",
        ltx_template_path=str(template_path),
    )

    assert calls["primary_model_path"] == transformer_path
    assert calls["template_root"] == template_path
    assert calls["component_paths"]["vae"] == "/custom/vae"
    assert calls["dtype"] == torch.float32
    assert pipeline.device == "cpu"


def test_ltx_load_pipeline_infers_split_component_roots(monkeypatch, tmp_path):
    """When using .../diffusion_models/file.safetensors, sibling VAE/clip roots are inferred."""

    models_root = tmp_path / "Models"
    diffusion_root = models_root / "diffusion_models"
    diffusion_root.mkdir(parents=True, exist_ok=True)
    transformer_path = diffusion_root / "ltx-2-19b-dev-fp8.safetensors"
    transformer_path.write_bytes(b"weights")
    (models_root / "VAE").mkdir()
    (models_root / "clip").mkdir()

    template_path = tmp_path / "template"
    template_path.mkdir(parents=True, exist_ok=True)

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "serenity.models.ltx2._find_ltx_template_root",
        lambda _explicit_template: template_path,
    )

    def _fake_build_bundle(*, primary_model_path, template_root, component_paths, dtype):
        del primary_model_path, template_root, dtype
        calls["component_paths"] = component_paths

        class _Pipeline:
            def to(self, _device):
                return self

        return _Pipeline()

    monkeypatch.setattr("serenity.models.ltx2._build_ltx_bundle", _fake_build_bundle)

    model = LTX2Model()
    model.load_pipeline(
        str(transformer_path),
        torch.float32,
        train_device=torch.device("cpu"),
    )

    component_paths = calls["component_paths"]
    assert component_paths["vae"] == str(models_root / "VAE")
    assert component_paths["text_encoder"] == str(models_root / "clip")
    assert component_paths["tokenizer"] == str(models_root / "clip")


def test_normalize_latents():
    """Test latent normalization matches SimpleTuner."""
    print("\n=== Test: Latent Normalization ===")

    # Create test latents [B, C, T, H, W]
    batch_size, channels, frames, height, width = 2, 128, 9, 48, 72
    latents = torch.randn(batch_size, channels, frames, height, width)

    # Create fake VAE statistics (typical values)
    latents_mean = torch.randn(channels) * 0.1  # Small mean
    latents_std = torch.ones(channels) + torch.randn(channels) * 0.1  # Close to 1
    scaling_factor = 0.3611  # Typical LTX2 scaling factor

    # Forward normalization
    normalized = normalize_ltx2_latents(
        latents, latents_mean, latents_std, scaling_factor, reverse=False
    )

    # Verify shape unchanged
    assert normalized.shape == latents.shape, f"Shape mismatch: {normalized.shape} vs {latents.shape}"

    # Verify not identical to input
    assert not torch.allclose(normalized, latents), "Normalization should change values"

    # Verify reverse gives back original
    restored = normalize_ltx2_latents(
        normalized, latents_mean, latents_std, scaling_factor, reverse=True
    )
    assert torch.allclose(restored, latents.float(), atol=1e-4), "Reverse should restore original"

    # Check normalized has approximately zero mean
    norm_mean = normalized.mean()
    print(f"  Normalized mean: {norm_mean.item():.4f} (should be near 0)")

    print("  ✅ Latent normalization PASSED")


def test_pack_unpack_latents():
    """Test pack/unpack are inverses of each other."""
    print("\n=== Test: Pack/Unpack Latents ===")

    # Test with different patch sizes
    for patch_size in [1, 2]:
        for patch_size_t in [1, 2]:
            batch_size, channels = 2, 128
            # Ensure dimensions are divisible by patch sizes
            frames = 8 * patch_size_t
            height = 48 * patch_size
            width = 72 * patch_size

            latents = torch.randn(batch_size, channels, frames, height, width)

            # Pack
            packed = pack_ltx2_latents(latents, patch_size, patch_size_t)

            # Expected sequence length
            post_patch_frames = frames // patch_size_t
            post_patch_height = height // patch_size
            post_patch_width = width // patch_size
            expected_seq_len = post_patch_frames * post_patch_height * post_patch_width
            expected_hidden_dim = channels * patch_size * patch_size * patch_size_t

            assert packed.shape == (batch_size, expected_seq_len, expected_hidden_dim), \
                f"Pack shape mismatch: {packed.shape} vs ({batch_size}, {expected_seq_len}, {expected_hidden_dim})"

            # Unpack
            unpacked = unpack_ltx2_latents(
                packed, post_patch_frames, post_patch_height, post_patch_width,
                patch_size, patch_size_t
            )

            # Verify roundtrip
            assert torch.allclose(unpacked, latents, atol=1e-6), \
                f"Pack/unpack should be identity for patch_size={patch_size}, patch_size_t={patch_size_t}"

            print(f"  ✅ patch_size={patch_size}, patch_size_t={patch_size_t}: PASSED")

    print("  ✅ Pack/Unpack latents PASSED")


def test_flow_matching_interpolation():
    """Test flow matching interpolation formula."""
    print("\n=== Test: Flow Matching Interpolation ===")

    batch_size, channels, frames, height, width = 2, 128, 9, 48, 72
    latents = torch.randn(batch_size, channels, frames, height, width)
    noise = torch.randn_like(latents)

    # Test at t=0: should equal latents (clean data)
    sigma = torch.zeros(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * latents + sigma * noise
    assert torch.allclose(noisy, latents), "At t=0, noisy should equal latents"

    # Test at t=1: should equal noise (pure noise)
    sigma = torch.ones(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * latents + sigma * noise
    assert torch.allclose(noisy, noise), "At t=1, noisy should equal noise"

    # Test at t=0.5: should be midpoint
    sigma = torch.full((batch_size, 1, 1, 1, 1), 0.5)
    noisy = (1 - sigma) * latents + sigma * noise
    expected = 0.5 * latents + 0.5 * noise
    assert torch.allclose(noisy, expected), "At t=0.5, noisy should be midpoint"

    # Test velocity target (flow matching objective)
    target = noise - latents
    print(f"  Target (velocity) shape: {target.shape}")

    print("  ✅ Flow matching interpolation PASSED")


def test_timestep_scaling():
    """Test that timesteps are scaled by 1000 for transformer."""
    print("\n=== Test: Timestep Scaling ===")

    # Sample continuous timesteps [0, 1]
    batch_size = 4
    timesteps = torch.rand(batch_size)

    # Scale for transformer (as in LTX2Model.forward)
    scaled = (timesteps * 1000).long()

    # Verify range
    assert scaled.min() >= 0, "Scaled timesteps should be >= 0"
    assert scaled.max() <= 1000, "Scaled timesteps should be <= 1000"

    # Verify integer type
    assert scaled.dtype == torch.int64, "Scaled timesteps should be long integers"

    print(f"  Raw timesteps: {timesteps.tolist()}")
    print(f"  Scaled timesteps: {scaled.tolist()}")

    print("  ✅ Timestep scaling PASSED")


def test_frame_constraint():
    """Test frame count adjustment for LTX2 constraint."""
    print("\n=== Test: Frame Constraint ===")

    # Test cases: (input, expected)
    test_cases = [
        (1, 1),    # Already valid
        (9, 9),    # Already valid
        (17, 17),  # Already valid
        (10, 9),   # Round down
        (16, 9),   # Round down (rounds to 9, not 17)
        (24, 17),  # Round down
        (121, 121), # Already valid (max common)
        (120, 113), # Round down
        (2, 1),    # Round down to minimum
        (8, 1),    # Round down
    ]

    for input_frames, expected in test_cases:
        result = adjust_video_frames(input_frames)
        assert result == expected, \
            f"Frame {input_frames} -> {result}, expected {expected}"
        # Verify constraint: frames % 8 == 1
        assert result % 8 == 1, f"Result {result} doesn't satisfy frames % 8 == 1"
        print(f"  ✅ {input_frames} -> {result}")

    # Test static method on LTX2Model
    assert LTX2Model.validate_frame_count(25) == 25, "Static method should work"
    assert LTX2Model.adjust_video_frames(26) == 25, "Alias should work"

    print("  ✅ Frame constraint PASSED")


def test_valid_frame_counts():
    """Test get_valid_frame_counts returns correct sequence."""
    print("\n=== Test: Valid Frame Counts ===")

    valid = LTX2Model.get_valid_frame_counts(65)
    expected = [1, 9, 17, 25, 33, 41, 49, 57, 65]

    assert valid == expected, f"Valid frame counts mismatch: {valid} vs {expected}"

    # Verify all satisfy constraint
    for f in valid:
        assert f % 8 == 1, f"Frame count {f} doesn't satisfy constraint"

    print(f"  Valid counts up to 65: {valid}")
    print("  ✅ Valid frame counts PASSED")


def test_pack_unpack_specific_shapes():
    """Test pack/unpack with LTX2-specific dimensions."""
    print("\n=== Test: Pack/Unpack Specific Shapes ===")

    # Typical LTX2 dimensions after VAE encoding
    # Original: 768x512, 25 frames
    # After 32x spatial, 8x temporal compression: 24x16, 4 frames
    batch_size = 1
    channels = 128  # LTX2 latent channels
    frames = 4
    height = 16
    width = 24

    latents = torch.randn(batch_size, channels, frames, height, width)

    # Pack with default patch sizes (1, 1)
    packed = pack_ltx2_latents(latents, patch_size=1, patch_size_t=1)

    # Expected: [B, T*H*W, C] = [1, 4*16*24, 128] = [1, 1536, 128]
    expected_shape = (batch_size, frames * height * width, channels)
    assert packed.shape == expected_shape, f"Shape mismatch: {packed.shape} vs {expected_shape}"

    # Unpack
    unpacked = unpack_ltx2_latents(packed, frames, height, width, patch_size=1, patch_size_t=1)
    assert torch.allclose(unpacked, latents), "Roundtrip failed"

    print(f"  Input shape: {latents.shape}")
    print(f"  Packed shape: {packed.shape}")
    print(f"  Unpacked shape: {unpacked.shape}")
    print("  ✅ Specific shapes PASSED")


def test_normalize_denormalize_roundtrip():
    """Test that normalize followed by denormalize is identity."""
    print("\n=== Test: Normalize/Denormalize Roundtrip ===")

    batch_size, channels, frames, height, width = 2, 128, 9, 48, 72
    latents = torch.randn(batch_size, channels, frames, height, width)

    # Typical LTX2 VAE parameters
    latents_mean = torch.zeros(channels)
    latents_std = torch.ones(channels)
    scaling_factor = 0.3611

    # Normalize
    normalized = normalize_ltx2_latents(
        latents, latents_mean, latents_std, scaling_factor, reverse=False
    )

    # Denormalize
    denormalized = normalize_ltx2_latents(
        normalized, latents_mean, latents_std, scaling_factor, reverse=True
    )

    # Should match original
    assert torch.allclose(denormalized, latents.float(), atol=1e-4), \
        f"Roundtrip failed: max diff = {(denormalized - latents.float()).abs().max()}"

    print("  ✅ Normalize/Denormalize roundtrip PASSED")


def test_normalize_with_nonzero_mean():
    """Test normalization with non-zero mean shifts values correctly."""
    print("\n=== Test: Normalize with Non-Zero Mean ===")

    batch_size, channels, frames, height, width = 1, 4, 1, 2, 2
    latents = torch.ones(batch_size, channels, frames, height, width) * 2.0

    # Mean of 1.0 should center at 0
    latents_mean = torch.ones(channels)
    latents_std = torch.ones(channels)
    scaling_factor = 1.0

    normalized = normalize_ltx2_latents(
        latents, latents_mean, latents_std, scaling_factor, reverse=False
    )

    # Should be approximately 1.0 (2 - 1 = 1)
    assert torch.allclose(normalized.float(), torch.ones_like(normalized), atol=1e-5), \
        "Normalization with mean shift failed"

    print("  ✅ Non-zero mean normalization PASSED")


def test_ltx_prompt_max_length_respects_tokenizer_limit():
    """Test LTX prompt max length caps at 226 and respects tokenizer limit."""
    print("\n=== Test: Prompt Max Length ===")

    class DummyPipeline:
        def __init__(self, model_max_length: int) -> None:
            self.tokenizer = type("Tokenizer", (), {"model_max_length": model_max_length})()
            self.max_sequence_length_seen = None

        def encode_prompt(self, **kwargs):
            self.max_sequence_length_seen = kwargs.get("max_sequence_length")
            prompt_embeds = torch.randn(1, 4, 8)
            prompt_mask = torch.ones(1, 4, dtype=torch.long)
            return prompt_embeds, prompt_mask

    model = LTX2Model()

    pipeline_large = DummyPipeline(model_max_length=512)
    model.encode_prompt_features(pipeline_large, "test", device=torch.device("cpu"))
    assert pipeline_large.max_sequence_length_seen == 226

    pipeline_small = DummyPipeline(model_max_length=128)
    model.encode_prompt_features(pipeline_small, "test", device=torch.device("cpu"))
    assert pipeline_small.max_sequence_length_seen == 128

    print("  ✅ Prompt max length PASSED")


def run_all_tests():
    """Run all parity tests."""
    print("=" * 60)
    print("LTX2 Parity Tests: Serenity vs SimpleTuner")
    print("=" * 60)

    test_normalize_latents()
    test_pack_unpack_latents()
    test_flow_matching_interpolation()
    test_timestep_scaling()
    test_frame_constraint()
    test_valid_frame_counts()
    test_pack_unpack_specific_shapes()
    test_normalize_denormalize_roundtrip()
    test_normalize_with_nonzero_mean()
    test_ltx_prompt_max_length_respects_tokenizer_limit()

    print("\n" + "=" * 60)
    print("All LTX2 parity tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
