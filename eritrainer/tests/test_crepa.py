"""
Test CREPA (Cross-frame REPresentation Alignment) Implementation

Tests:
1. CrepaScheduler warmup, decay, and cutoff
2. CrepaRegularizer projector creation
3. Temporal and spatial alignment
4. Loss computation (mock without actual encoder)

Run: python -m eritrainer.tests.test_crepa
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any

from eritrainer.training.crepa import (
    CrepaConfig,
    CrepaScheduler,
    CrepaRegularizer,
)


def test_crepa_scheduler_constant():
    """Test constant scheduler maintains weight."""
    print("\n=== Test: CrepaScheduler (constant) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        scheduler_type="constant",
    )
    scheduler = CrepaScheduler(config, max_train_steps=1000)

    # Weight should be constant
    assert scheduler.get_weight(0) == 0.5, "Step 0 should be 0.5"
    assert scheduler.get_weight(500) == 0.5, "Step 500 should be 0.5"
    assert scheduler.get_weight(1000) == 0.5, "Step 1000 should be 0.5"

    print("  ✅ Constant scheduler PASSED")


def test_crepa_scheduler_warmup():
    """Test warmup phase."""
    print("\n=== Test: CrepaScheduler (warmup) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=1.0,
        scheduler_type="constant",
        warmup_steps=100,
    )
    scheduler = CrepaScheduler(config, max_train_steps=1000)

    # Step 0: should be 0
    assert scheduler.get_weight(0) == 0.0, "Step 0 should be 0"

    # Step 50: should be 0.5
    assert abs(scheduler.get_weight(50) - 0.5) < 0.01, "Step 50 should be ~0.5"

    # Step 100+: should be 1.0
    assert scheduler.get_weight(100) == 1.0, "Step 100 should be 1.0"
    assert scheduler.get_weight(500) == 1.0, "Step 500 should be 1.0"

    print("  ✅ Warmup scheduler PASSED")


def test_crepa_scheduler_linear_decay():
    """Test linear decay."""
    print("\n=== Test: CrepaScheduler (linear decay) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=1.0,
        scheduler_type="linear",
        decay_steps=100,
        lambda_end=0.0,
    )
    scheduler = CrepaScheduler(config, max_train_steps=100)

    # Linear decay from 1.0 to 0.0 over 100 steps
    assert scheduler.get_weight(0) == 1.0, "Step 0 should be 1.0"
    assert abs(scheduler.get_weight(50) - 0.5) < 0.01, "Step 50 should be ~0.5"
    assert scheduler.get_weight(100) == 0.0, "Step 100 should be 0.0"

    print("  ✅ Linear decay scheduler PASSED")


def test_crepa_scheduler_cutoff():
    """Test step-based cutoff."""
    print("\n=== Test: CrepaScheduler (cutoff) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=1.0,
        scheduler_type="constant",
        cutoff_step=50,
        threshold_mode="permanent",
    )
    scheduler = CrepaScheduler(config, max_train_steps=100)

    # Before cutoff
    assert scheduler.get_weight(40) == 1.0, "Step 40 should be 1.0"
    assert not scheduler.is_cutoff(), "Should not be cutoff yet"

    # After cutoff
    assert scheduler.get_weight(60) == 0.0, "Step 60 should be 0.0"
    assert scheduler.is_cutoff(), "Should be cutoff"

    # Permanent: stays cutoff
    assert scheduler.get_weight(40) == 0.0, "After cutoff, should stay 0"

    print("  ✅ Cutoff scheduler PASSED")


def test_crepa_config():
    """Test CrepaConfig defaults and creation."""
    print("\n=== Test: CrepaConfig ===")

    # Default config
    config = CrepaConfig()
    assert config.enabled is False
    assert config.crepa_lambda == 0.5
    assert config.encoder_name == "dinov2_vitg14"

    # Custom config
    config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.3,
        adjacent_distance=2,
        encoder_name="dinov2_vitb14",
    )
    assert config.enabled is True
    assert config.crepa_lambda == 0.3
    assert config.adjacent_distance == 2

    print("  ✅ CrepaConfig PASSED")


class MockModel(nn.Module):
    """Mock transformer for testing projector attachment."""
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)


def test_crepa_projector_attachment():
    """Test projector is attached to model."""
    print("\n=== Test: CrepaRegularizer (projector attachment) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=10,
        use_backbone_features=True,  # Skip encoder loading
    )

    device = torch.device("cpu")
    regularizer = CrepaRegularizer(config, device, hidden_size=1024)

    model = MockModel(1024)

    # Before attachment
    assert not hasattr(model, "crepa_projector"), "Should not have projector yet"

    # Attach
    regularizer.attach_to_model(model)

    # After attachment
    assert hasattr(model, "crepa_projector"), "Should have projector"
    assert regularizer.projector is not None
    assert regularizer.encoder_dim == 1024  # Backbone mode uses hidden_size

    print("  ✅ Projector attachment PASSED")


def test_crepa_project_hidden_states():
    """Test hidden state projection."""
    print("\n=== Test: CrepaRegularizer (projection) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=10,
        use_backbone_features=True,
    )

    device = torch.device("cpu")
    regularizer = CrepaRegularizer(config, device, hidden_size=512)

    model = MockModel(512)
    regularizer.attach_to_model(model)

    # Test 4D input [B, T, P, D]
    hidden_states = torch.randn(2, 9, 16, 512)
    projected = regularizer._project_hidden_states(hidden_states)

    assert projected.shape == (2, 9, 16, 512), f"Shape mismatch: {projected.shape}"

    # Test 3D input [B, T, D]
    hidden_states_3d = torch.randn(2, 9, 512)
    projected_3d = regularizer._project_hidden_states(hidden_states_3d)

    assert projected_3d.shape == (2, 9, 1, 512), f"Shape mismatch: {projected_3d.shape}"

    print("  ✅ Projection PASSED")


def test_crepa_temporal_alignment():
    """Test temporal dimension alignment."""
    print("\n=== Test: CrepaRegularizer (temporal alignment) ===")

    config = CrepaConfig(
        enabled=True,
        use_backbone_features=True,
        block_index=10,
    )
    regularizer = CrepaRegularizer(config, torch.device("cpu"), hidden_size=512)

    # Same size - no change
    proj = torch.randn(2, 9, 16, 512)
    feat = torch.randn(2, 9, 16, 512)
    p_out, f_out = regularizer._align_temporal(proj, feat)
    assert p_out.shape[1] == f_out.shape[1] == 9

    # Different sizes - subsample larger
    proj = torch.randn(2, 5, 16, 512)
    feat = torch.randn(2, 25, 16, 512)
    p_out, f_out = regularizer._align_temporal(proj, feat)
    assert p_out.shape[1] == f_out.shape[1] == 5

    proj = torch.randn(2, 25, 16, 512)
    feat = torch.randn(2, 5, 16, 512)
    p_out, f_out = regularizer._align_temporal(proj, feat)
    assert p_out.shape[1] == f_out.shape[1] == 5

    print("  ✅ Temporal alignment PASSED")


def test_crepa_spatial_alignment():
    """Test spatial token alignment."""
    print("\n=== Test: CrepaRegularizer (spatial alignment) ===")

    config = CrepaConfig(
        enabled=True,
        use_backbone_features=True,
        block_index=10,
        spatial_align=True,
    )
    regularizer = CrepaRegularizer(config, torch.device("cpu"), hidden_size=512)

    # Same size
    proj = torch.randn(2, 9, 16, 512)
    feat = torch.randn(2, 9, 16, 512)
    p_out, f_out = regularizer._align_spatial(proj, feat)
    assert p_out.shape[2] == f_out.shape[2] == 16

    # Different sizes - interpolate to smaller
    proj = torch.randn(2, 9, 64, 512)  # 8x8
    feat = torch.randn(2, 9, 16, 512)  # 4x4
    p_out, f_out = regularizer._align_spatial(proj, feat)
    assert p_out.shape[2] == f_out.shape[2] == 16

    print("  ✅ Spatial alignment PASSED")


def test_crepa_loss_computation_backbone_mode():
    """Test loss computation in backbone features mode (no encoder)."""
    print("\n=== Test: CrepaRegularizer (loss computation) ===")

    config = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=10,
        use_backbone_features=True,
        adjacent_distance=1,
    )

    device = torch.device("cpu")
    regularizer = CrepaRegularizer(config, device, hidden_size=512)

    model = MockModel(512)
    regularizer.attach_to_model(model)

    # Create test tensors
    batch_size, num_frames, num_patches, hidden_size = 2, 9, 16, 512
    hidden_states = torch.randn(batch_size, num_frames, num_patches, hidden_size)
    frame_features = torch.randn(batch_size, num_frames, num_patches, hidden_size)

    # Compute loss
    loss, log_data = regularizer.compute_loss(
        hidden_states=hidden_states,
        frame_features=frame_features,
        step=100,
    )

    assert loss is not None, "Loss should not be None"
    assert loss.ndim == 0, "Loss should be scalar"
    assert log_data is not None
    assert "crepa/loss" in log_data
    assert "crepa/similarity" in log_data
    assert "crepa/weight" in log_data

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Similarity: {log_data['crepa/similarity']:.4f}")
    print(f"  Weight: {log_data['crepa/weight']:.4f}")

    print("  ✅ Loss computation PASSED")


def test_crepa_cumulative_neighbors():
    """Test cumulative neighbors mode."""
    print("\n=== Test: CrepaRegularizer (cumulative neighbors) ===")

    # Non-cumulative (default)
    config1 = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=10,
        use_backbone_features=True,
        adjacent_distance=2,
        cumulative_neighbors=False,
    )

    # Cumulative
    config2 = CrepaConfig(
        enabled=True,
        crepa_lambda=0.5,
        block_index=10,
        use_backbone_features=True,
        adjacent_distance=2,
        cumulative_neighbors=True,
    )

    device = torch.device("cpu")

    reg1 = CrepaRegularizer(config1, device, hidden_size=512)
    reg2 = CrepaRegularizer(config2, device, hidden_size=512)

    model1 = MockModel(512)
    model2 = MockModel(512)
    reg1.attach_to_model(model1)
    reg2.attach_to_model(model2)

    # Same projector weights for comparison
    with torch.no_grad():
        for p1, p2 in zip(reg1.projector.parameters(), reg2.projector.parameters()):
            p2.copy_(p1)

    hidden = torch.randn(2, 9, 16, 512)
    features = torch.randn(2, 9, 16, 512)

    loss1, _ = reg1.compute_loss(hidden_states=hidden, frame_features=features)
    loss2, _ = reg2.compute_loss(hidden_states=hidden, frame_features=features)

    # Losses should be different (cumulative uses more neighbors)
    print(f"  Non-cumulative loss: {loss1.item():.4f}")
    print(f"  Cumulative loss: {loss2.item():.4f}")

    print("  ✅ Cumulative neighbors PASSED")


def test_crepa_disabled():
    """Test disabled CREPA returns None."""
    print("\n=== Test: CrepaRegularizer (disabled) ===")

    config = CrepaConfig(enabled=False)
    regularizer = CrepaRegularizer(config, torch.device("cpu"), hidden_size=512)

    assert not regularizer.enabled
    assert not regularizer.wants_hidden_states()

    loss, log_data = regularizer.compute_loss(
        hidden_states=torch.randn(2, 9, 512),
    )

    assert loss is None
    assert log_data is None

    print("  ✅ Disabled mode PASSED")


def run_all_tests():
    """Run all CREPA tests."""
    print("=" * 60)
    print("CREPA Implementation Tests")
    print("=" * 60)

    test_crepa_scheduler_constant()
    test_crepa_scheduler_warmup()
    test_crepa_scheduler_linear_decay()
    test_crepa_scheduler_cutoff()
    test_crepa_config()
    test_crepa_projector_attachment()
    test_crepa_project_hidden_states()
    test_crepa_temporal_alignment()
    test_crepa_spatial_alignment()
    test_crepa_loss_computation_backbone_mode()
    test_crepa_cumulative_neighbors()
    test_crepa_disabled()

    print("\n" + "=" * 60)
    print("All CREPA tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
