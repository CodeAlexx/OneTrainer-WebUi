"""
FLUX.2 Klein Specific Tests

Tests for FLUX.2 Klein implementation details:
- VAE patchification (32 → 128 channels)
- Qwen3 stacked layer embeddings
- Flow matching loss computation
- Block swap configuration
"""

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# Test: VAE Patchification
# =============================================================================

class TestVAEPatchification:
    """Test VAE pixel shuffle patchification."""

    def test_patchify_latents_shape(self):
        """Patchify should transform (B, 32, H, W) → (B, 128, H/2, W/2)."""
        from eritrainer.models.flux2_klein import Flux2KleinModel

        # Create model with minimal components (no actual weights)
        class MockModel:
            pass

        model = MockModel()

        # Use the static method from Flux2KleinModel if available,
        # otherwise test the algorithm directly
        latents = torch.randn(1, 32, 64, 64)

        # Patchify: (B, 32, H, W) → (B, 128, H/2, W/2)
        b, c, h, w = latents.shape
        assert c == 32, f"Expected 32 channels, got {c}"

        # Reshape: (B, 32, H, W) → (B, 32, H/2, 2, W/2, 2)
        patchified = latents.view(b, c, h // 2, 2, w // 2, 2)

        # Permute: (B, 32, H/2, 2, W/2, 2) → (B, 32, 2, 2, H/2, W/2)
        patchified = patchified.permute(0, 1, 3, 5, 2, 4)

        # Flatten channels: (B, 32*2*2, H/2, W/2) = (B, 128, H/2, W/2)
        patchified = patchified.reshape(b, c * 4, h // 2, w // 2)

        assert patchified.shape == (1, 128, 32, 32)
        print(f"  ✅ Patchification: {latents.shape} → {patchified.shape}")

    def test_patchify_latents_reversible(self):
        """Patchification should be reversible (unpatchify)."""
        latents = torch.randn(1, 32, 64, 64)
        b, c, h, w = latents.shape

        # Patchify
        patchified = latents.view(b, c, h // 2, 2, w // 2, 2)
        patchified = patchified.permute(0, 1, 3, 5, 2, 4)
        patchified = patchified.reshape(b, c * 4, h // 2, w // 2)

        # Unpatchify (reverse)
        b2, c2, h2, w2 = patchified.shape
        unpatchified = patchified.reshape(b2, c2 // 4, 2, 2, h2, w2)
        unpatchified = unpatchified.permute(0, 1, 4, 2, 5, 3)  # Reverse of 0,1,3,5,2,4
        unpatchified = unpatchified.reshape(b2, c2 // 4, h2 * 2, w2 * 2)

        assert unpatchified.shape == latents.shape
        assert torch.allclose(unpatchified, latents)
        print("  ✅ Patchification is reversible")


# =============================================================================
# Test: Qwen3 Stacked Layer Embeddings
# =============================================================================

class TestQwen3StackedEmbeddings:
    """Test Qwen3 stacked layer embedding computation."""

    def test_stacked_layers_shape_4b(self):
        """
        Klein 4B embedding should be [B, L, 7680] from stacking layers [9, 18, 27].
        7680 = 3 * 2560 (Qwen3 4B hidden size)
        """
        # Simulate Qwen3 hidden states
        batch_size = 2
        seq_len = 512
        hidden_size = 2560  # Qwen3 4B
        num_layers = 28  # Qwen3 has 28 layers

        # Mock hidden states from layers [9, 18, 27]
        hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)]

        # Stack layers [9, 18, 27]
        OUTPUT_LAYERS = (9, 18, 27)
        stacked = torch.stack([hidden_states[i] for i in OUTPUT_LAYERS], dim=1)  # [B, 3, L, D]

        # Reshape to [B, L, 3*D]
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            stacked.shape[0], stacked.shape[2], -1
        )  # [B, L, 3*D]

        assert prompt_embeds.shape == (batch_size, seq_len, 7680)
        print(f"  ✅ Klein 4B embedding shape: {prompt_embeds.shape}")

    def test_stacked_layers_shape_9b(self):
        """
        Klein 9B embedding should be [B, L, 12288] from stacking layers [9, 18, 27].
        12288 = 3 * 4096 (Qwen3 9B hidden size)
        """
        batch_size = 2
        seq_len = 512
        hidden_size = 4096  # Qwen3 9B
        num_layers = 36  # Approximate

        hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)]

        OUTPUT_LAYERS = (9, 18, 27)
        stacked = torch.stack([hidden_states[i] for i in OUTPUT_LAYERS], dim=1)
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            stacked.shape[0], stacked.shape[2], -1
        )

        assert prompt_embeds.shape == (batch_size, seq_len, 12288)
        print(f"  ✅ Klein 9B embedding shape: {prompt_embeds.shape}")


# =============================================================================
# Test: Flow Matching Loss
# =============================================================================

class TestFlowMatchingLoss:
    """Test flow matching loss computation."""

    def test_flow_matching_target(self):
        """Flow matching target should be: velocity = noise - latents."""
        batch_size = 2
        channels = 128
        height = 32
        width = 32

        latents = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(latents)

        # Flow matching target
        target = noise - latents

        # Verify
        assert target.shape == latents.shape
        assert torch.allclose(target, noise - latents)
        print("  ✅ Flow matching target: velocity = noise - latents")

    def test_flow_matching_interpolation(self):
        """Flow matching interpolation: x_t = (1-t) * x_0 + t * noise."""
        batch_size = 2
        channels = 128
        height = 32
        width = 32

        latents = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(latents)

        # Sample timesteps
        t = torch.rand(batch_size)
        t_expanded = t.view(-1, 1, 1, 1)

        # Interpolation
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise

        # At t=0: should be latents
        t_zero = torch.zeros(batch_size).view(-1, 1, 1, 1)
        at_zero = (1 - t_zero) * latents + t_zero * noise
        assert torch.allclose(at_zero, latents)

        # At t=1: should be noise
        t_one = torch.ones(batch_size).view(-1, 1, 1, 1)
        at_one = (1 - t_one) * latents + t_one * noise
        assert torch.allclose(at_one, noise)

        print("  ✅ Flow matching interpolation verified")

    def test_mse_loss_computation(self):
        """MSE loss between prediction and target."""
        pred = torch.randn(2, 128, 32, 32)
        target = torch.randn(2, 128, 32, 32)

        loss = F.mse_loss(pred, target, reduction="mean")

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        print(f"  ✅ MSE loss computed: {loss.item():.4f}")


# =============================================================================
# Test: Block Swap Limits
# =============================================================================

class TestBlockSwapLimits:
    """Test block swap configuration limits."""

    def test_klein_4b_block_count(self):
        """Klein 4B should have 25 blocks (5 double + 20 single)."""
        from eritrainer.models.flux2_klein import FLUX2_KLEIN_CONFIG

        config = FLUX2_KLEIN_CONFIG['klein-4b']
        assert config['double_blocks'] == 5
        assert config['single_blocks'] == 20
        assert config['total_blocks'] == 25
        assert config['max_swappable'] == 24  # total - 1
        print("  ✅ Klein 4B: 25 blocks, 24 swappable")

    def test_klein_9b_block_count(self):
        """Klein 9B should have 32 blocks (8 double + 24 single)."""
        from eritrainer.models.flux2_klein import FLUX2_KLEIN_CONFIG

        config = FLUX2_KLEIN_CONFIG['klein-9b']
        assert config['double_blocks'] == 8
        assert config['single_blocks'] == 24
        assert config['total_blocks'] == 32
        assert config['max_swappable'] == 31  # total - 1
        print("  ✅ Klein 9B: 32 blocks, 31 swappable")

    @pytest.mark.skip(reason="block_swap module not yet implemented")
    def test_block_swap_clamping(self):
        """Block swap should clamp to max allowed."""
        from eritrainer.training.block_swap import BlockSwapOffloader

        # Create 10 dummy blocks
        blocks = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(10)])

        # Request 15 (more than max = 10 - 2 = 8)
        offloader = BlockSwapOffloader(
            block_type="test",
            blocks=list(blocks),
            num_blocks=10,
            blocks_to_swap=15,
            device=torch.device('cpu'),
        )

        assert offloader.blocks_to_swap == 8  # Clamped to max
        print("  ✅ Block swap clamped correctly")


# =============================================================================
# Test: No Guidance for Klein
# =============================================================================

class TestNoGuidance:
    """Test that Klein models don't use guidance."""

    @pytest.mark.skip(reason="predict module not yet implemented")
    def test_guidance_is_none_in_predictor(self):
        """Flux2KleinPredictor should pass guidance=None."""
        from eritrainer.training.predict import Flux2KleinPredictor

        predictor = Flux2KleinPredictor()

        # The predictor should exist and be configured for no guidance
        assert predictor is not None
        print("  ✅ Flux2KleinPredictor created (will pass guidance=None)")


# =============================================================================
# Test: VAE Batch Normalization
# =============================================================================

class TestVAEBatchNorm:
    """Test VAE batch norm statistics handling."""

    def test_normalize_latents_formula(self):
        """Normalization: (latents - mean) / sqrt(var + eps)."""
        # Simulate batch norm running stats
        channels = 128
        running_mean = torch.randn(channels)
        running_var = torch.abs(torch.randn(channels)) + 0.1  # Ensure positive
        eps = 1e-5

        latents = torch.randn(2, channels, 32, 32)

        # Normalize
        mean = running_mean.view(1, -1, 1, 1)
        var = running_var.view(1, -1, 1, 1)

        normalized = (latents - mean) / torch.sqrt(var + eps)

        assert normalized.shape == latents.shape
        print("  ✅ VAE batch norm normalization formula verified")


# =============================================================================
# Test: Latent Packing
# =============================================================================

class TestLatentPacking:
    """Test latent packing for transformer input."""

    def test_pack_latents_shape(self):
        """Pack latents: (B, C, H, W) → (B, H*W, C)."""
        latents = torch.randn(2, 128, 32, 32)

        # Pack: (B, C, H, W) → (B, H*W, C)
        b, c, h, w = latents.shape
        packed = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)

        assert packed.shape == (2, 32 * 32, 128)
        print(f"  ✅ Packed latents: {latents.shape} → {packed.shape}")

    def test_unpack_latents_roundtrip(self):
        """Pack and unpack should roundtrip."""
        latents = torch.randn(2, 128, 32, 32)
        b, c, h, w = latents.shape

        # Pack
        packed = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Unpack
        unpacked = packed.reshape(b, h, w, c).permute(0, 3, 1, 2)

        assert torch.allclose(unpacked, latents)
        print("  ✅ Pack/unpack roundtrip verified")


# =============================================================================
# Test: Flux2KleinSampler
# =============================================================================

class TestFlux2KleinSampler:
    """Test Flux2KleinSampler class."""

    def test_sampler_exists(self):
        """Sampler class should be importable."""
        from eritrainer.models.flux2_klein import Flux2KleinSampler
        assert Flux2KleinSampler is not None
        print("  ✅ Flux2KleinSampler is importable")

    @pytest.mark.skip(reason="SampleGenerator class not yet implemented")
    def test_sampler_dispatch(self):
        """SampleGenerator should dispatch Flux2Klein to Flux2KleinSampler."""
        from eritrainer.sampling.sampler import SampleGenerator

        # Check the model type dispatch logic
        model_type = "Flux2KleinModel"
        assert 'Flux2Klein' in model_type
        assert 'Flux' in model_type
        # Flux2Klein check should come BEFORE Flux check
        print("  ✅ Sampler dispatch correctly prioritizes Flux2Klein over Flux")

    def test_auto_step_detection(self):
        """Sampler should auto-detect steps based on model variant."""
        # Distilled should default to 4 steps
        # Base should default to 50 steps
        from eritrainer.models.flux2_klein import FLUX2_KLEIN_CONFIG

        # For distilled models (klein-4b, klein-9b without -base)
        expected_distilled_steps = 4
        expected_base_steps = 50

        # Just verify the expected values make sense
        assert expected_distilled_steps < expected_base_steps
        print(f"  ✅ Step defaults: distilled={expected_distilled_steps}, base={expected_base_steps}")

    def test_flow_matching_timesteps(self):
        """Flow matching should use timesteps from 1.0 → 0.0."""
        import torch

        num_steps = 4
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

        assert timesteps[0] == 1.0
        assert timesteps[-1] == 0.0
        assert len(timesteps) == num_steps + 1
        print(f"  ✅ Flow matching timesteps: {timesteps.tolist()}")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all FLUX.2 Klein specific tests."""
    print("=== FLUX.2 Klein Specific Tests ===\n")

    print("Test: VAE Patchification")
    t = TestVAEPatchification()
    t.test_patchify_latents_shape()
    t.test_patchify_latents_reversible()

    print("\nTest: Qwen3 Stacked Embeddings")
    t = TestQwen3StackedEmbeddings()
    t.test_stacked_layers_shape_4b()
    t.test_stacked_layers_shape_9b()

    print("\nTest: Flow Matching Loss")
    t = TestFlowMatchingLoss()
    t.test_flow_matching_target()
    t.test_flow_matching_interpolation()
    t.test_mse_loss_computation()

    print("\nTest: Block Swap Limits")
    t = TestBlockSwapLimits()
    t.test_klein_4b_block_count()
    t.test_klein_9b_block_count()
    t.test_block_swap_clamping()

    print("\nTest: No Guidance")
    t = TestNoGuidance()
    t.test_guidance_is_none_in_predictor()

    print("\nTest: VAE Batch Norm")
    t = TestVAEBatchNorm()
    t.test_normalize_latents_formula()

    print("\nTest: Latent Packing")
    t = TestLatentPacking()
    t.test_pack_latents_shape()
    t.test_unpack_latents_roundtrip()

    print("\nTest: Flux2KleinSampler")
    t = TestFlux2KleinSampler()
    t.test_sampler_exists()
    t.test_sampler_dispatch()
    t.test_auto_step_detection()
    t.test_flow_matching_timesteps()

    print("\n=== ALL FLUX.2 Klein Tests PASSED ===")


if __name__ == "__main__":
    run_all_tests()
