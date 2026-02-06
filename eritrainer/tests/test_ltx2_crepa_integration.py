"""
Integration Test: LTX2 + CREPA

Demonstrates how to use LTX2 training with CREPA regularization.
This test uses mock components to avoid loading actual models.

Run: python -m eritrainer.tests.test_ltx2_crepa_integration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Import LTX2 components
from eritrainer.models.ltx2 import (
    LTX2Model,
    normalize_ltx2_latents,
    pack_ltx2_latents,
    unpack_ltx2_latents,
    adjust_video_frames,
)

# Import CREPA components
from eritrainer.training.crepa import (
    CrepaConfig,
    CrepaRegularizer,
)


# =============================================================================
# Mock Components
# =============================================================================


class MockTransformer(nn.Module):
    """Mock LTX2 transformer for testing."""

    def __init__(self, hidden_size: int = 1024, latent_dim: int = 128, num_layers: int = 28):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        # Input projection: latent_dim -> hidden_size
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        # Simple transformer simulation
        self.proj = nn.Linear(hidden_size, hidden_size)
        # Output projection: hidden_size -> latent_dim
        self.output_proj = nn.Linear(hidden_size, latent_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: int = 9,
        height: int = 48,
        width: int = 72,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        """Forward pass returning sample and optionally hidden states."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        input_dim = hidden_states.shape[2]

        # Project input to hidden size
        h = self.input_proj(hidden_states.float())  # [B, seq, hidden]

        # Simple forward pass
        h = self.proj(h)

        # Project back to latent dim for output
        output = self.output_proj(h).to(hidden_states.dtype)  # [B, seq, latent_dim]

        # Create mock hidden states for CREPA (in hidden_size space)
        crepa_hidden = None
        if output_hidden_states:
            # [B, seq_len, hidden] -> [B, T, H*W, hidden]
            crepa_hidden = h.view(batch_size, num_frames, -1, self.hidden_size)

        @dataclass
        class Output:
            sample: torch.Tensor
            hidden_states: Optional[torch.Tensor] = None

        if output_hidden_states:
            return Output(sample=output, hidden_states=crepa_hidden)
        return Output(sample=output)


# =============================================================================
# Integration Tests
# =============================================================================


def test_ltx2_crepa_training_loop():
    """Test a mock training loop with LTX2 + CREPA."""
    print("\n=== Integration Test: LTX2 + CREPA Training Loop ===")

    device = torch.device("cpu")
    dtype = torch.float32
    hidden_size = 1024
    batch_size = 2
    num_frames = 9  # Satisfies frames % 8 == 1
    height = 48
    width = 72
    latent_channels = 128

    # Create mock transformer
    transformer = MockTransformer(hidden_size=hidden_size, latent_dim=latent_channels).to(device)

    # Set up normalization parameters (as if from VAE)
    latents_mean = torch.zeros(latent_channels)
    latents_std = torch.ones(latent_channels)
    scaling_factor = 0.3611

    # Create CREPA regularizer
    crepa_config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=14,  # Middle layer
        use_backbone_features=True,  # Use transformer features (no encoder)
        adjacent_distance=1,
        warmup_steps=10,
        scheduler_type="constant",
    )

    crepa = CrepaRegularizer(
        crepa_config,
        device=device,
        hidden_size=hidden_size,
        max_train_steps=100,
    )

    # Attach CREPA projector to transformer
    crepa.attach_to_model(transformer)

    # Mock training data
    latents = torch.randn(batch_size, latent_channels, num_frames, height, width, device=device, dtype=dtype)
    text_embeddings = torch.randn(batch_size, 77, hidden_size, device=device, dtype=dtype)

    print(f"  Latents shape: {latents.shape}")
    print(f"  Text embeddings shape: {text_embeddings.shape}")

    # =========================================================================
    # Training Step
    # =========================================================================

    # 1. Normalize latents (using standalone function)
    scaled_latents = normalize_ltx2_latents(
        latents, latents_mean, latents_std, scaling_factor, reverse=False
    )
    print(f"  Scaled latents shape: {scaled_latents.shape}")

    # 2. Sample timesteps [0, 1]
    timesteps = torch.rand(batch_size, device=device, dtype=dtype)

    # 3. Sample noise
    noise = torch.randn_like(scaled_latents)

    # 4. Flow matching interpolation
    sigma = timesteps.view(-1, 1, 1, 1, 1)
    noisy_latents = (1 - sigma) * scaled_latents + sigma * noise

    # 5. Pack latents (using standalone function)
    packed_noisy = pack_ltx2_latents(noisy_latents, patch_size=1, patch_size_t=1)
    print(f"  Packed latents shape: {packed_noisy.shape}")

    # 6. Forward through transformer (with hidden states for CREPA)
    output = transformer(
        hidden_states=packed_noisy,
        encoder_hidden_states=text_embeddings,
        timestep=timesteps * 1000,
        num_frames=num_frames,
        height=height,
        width=width,
        output_hidden_states=crepa.wants_hidden_states(),
        return_dict=True,
    )

    # 7. Unpack prediction (using standalone function)
    packed_pred = output.sample
    prediction = unpack_ltx2_latents(packed_pred, num_frames, height, width, patch_size=1, patch_size_t=1)
    print(f"  Prediction shape: {prediction.shape}")

    # 8. Compute MSE loss (velocity target)
    target = noise - scaled_latents
    mse_loss = torch.nn.functional.mse_loss(prediction, target)
    print(f"  MSE Loss: {mse_loss.item():.4f}")

    # 9. Compute CREPA loss
    crepa_loss, crepa_log = crepa.compute_loss(
        hidden_states=output.hidden_states,
        frame_features=output.hidden_states,  # In backbone mode, same as hidden states
        step=50,
    )

    if crepa_loss is not None:
        print(f"  CREPA Loss: {crepa_loss.item():.4f}")
        print(f"  CREPA Similarity: {crepa_log['crepa/similarity']:.4f}")
        print(f"  CREPA Weight: {crepa_log['crepa/weight']:.4f}")

        # Total loss
        total_loss = mse_loss + crepa_loss
        print(f"  Total Loss: {total_loss.item():.4f}")

        # Backward pass (verify gradients flow)
        total_loss.backward()

        # Check CREPA projector has gradients
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in crepa.projector.parameters()
        )
        assert projector_has_grad, "CREPA projector should have gradients"
        print("  ✅ Gradients flow through CREPA projector")
    else:
        print("  CREPA loss is None (cutoff active)")

    print("\n  ✅ LTX2 + CREPA integration test PASSED")


def test_ltx2_frame_constraint():
    """Test frame count adjustment."""
    print("\n=== Test: LTX2 Frame Constraint ===")

    # Valid frame counts (frames % 8 == 1)
    valid = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]

    for n in valid:
        adjusted = adjust_video_frames(n)
        assert adjusted == n, f"Frame {n} should remain unchanged"
        assert adjusted % 8 == 1, f"Frame {adjusted} doesn't satisfy constraint"

    # Invalid frame counts (should be adjusted)
    invalid = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    for n in invalid:
        adjusted = adjust_video_frames(n)
        assert adjusted % 8 == 1, f"Adjusted {adjusted} doesn't satisfy constraint"
        assert adjusted <= n, f"Adjusted {adjusted} should be <= {n}"

    # Also test via static method
    assert LTX2Model.validate_frame_count(25) == 25
    assert LTX2Model.adjust_video_frames(26) == 25

    print("  ✅ Frame constraint PASSED")


def test_ltx2_pack_unpack_identity():
    """Test pack/unpack are inverses using standalone functions."""
    print("\n=== Test: LTX2 Pack/Unpack Identity ===")

    batch_size = 2
    channels = 128
    num_frames = 9
    height = 48
    width = 72

    latents = torch.randn(batch_size, channels, num_frames, height, width)

    # Pack using standalone function
    packed = pack_ltx2_latents(latents, patch_size=1, patch_size_t=1)
    expected_seq_len = num_frames * height * width
    assert packed.shape == (batch_size, expected_seq_len, channels), f"Wrong packed shape: {packed.shape}"

    # Unpack using standalone function
    unpacked = unpack_ltx2_latents(packed, num_frames, height, width, patch_size=1, patch_size_t=1)
    assert unpacked.shape == latents.shape, f"Wrong unpacked shape: {unpacked.shape}"
    assert torch.allclose(unpacked, latents, atol=1e-6), "Pack/unpack not identity"

    print("  ✅ Pack/Unpack identity PASSED")


def test_ltx2_scale_unscale_identity():
    """Test scale/unscale are inverses using standalone functions."""
    print("\n=== Test: LTX2 Scale/Unscale Identity ===")

    channels = 128

    # Set up normalization parameters
    latents_mean = torch.randn(channels) * 0.1
    latents_std = torch.ones(channels) + torch.randn(channels) * 0.1
    scaling_factor = 0.3611

    latents = torch.randn(2, channels, 9, 48, 72)

    # Scale using standalone function
    scaled = normalize_ltx2_latents(
        latents, latents_mean, latents_std, scaling_factor, reverse=False
    )

    # Unscale using standalone function
    unscaled = normalize_ltx2_latents(
        scaled, latents_mean, latents_std, scaling_factor, reverse=True
    )

    # Should match original (compare as float since normalize converts to float)
    assert torch.allclose(unscaled, latents.float(), atol=1e-4), "Scale/unscale not identity"

    print("  ✅ Scale/Unscale identity PASSED")


def test_flow_matching_formulation():
    """Test flow matching formulas are correct."""
    print("\n=== Test: Flow Matching Formulation ===")

    batch_size = 2
    channels = 128
    num_frames = 9
    height = 48
    width = 72

    latents = torch.randn(batch_size, channels, num_frames, height, width)
    noise = torch.randn_like(latents)

    # Test at t=0: should equal latents (clean data)
    sigma = torch.zeros(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * latents + sigma * noise
    assert torch.allclose(noisy, latents), "At t=0, noisy should equal latents"

    # Test at t=1: should equal noise (pure noise)
    sigma = torch.ones(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * latents + sigma * noise
    assert torch.allclose(noisy, noise), "At t=1, noisy should equal noise"

    # Test velocity target
    target = noise - latents
    assert target.shape == latents.shape, "Target shape should match latents"

    print("  ✅ Flow matching formulation PASSED")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("LTX2 + CREPA Integration Tests")
    print("=" * 60)

    test_ltx2_frame_constraint()
    test_ltx2_pack_unpack_identity()
    test_ltx2_scale_unscale_identity()
    test_flow_matching_formulation()
    test_ltx2_crepa_training_loop()

    print("\n" + "=" * 60)
    print("All integration tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
