"""Tests for Phase 5b pipeline and training modules.

Covers:
- pipeline/buckets.py: BucketManager and BucketBatchSampler
- pipeline/masks.py: mask generation and resizing utilities
- training/optimizers.py: OptimizerType enum and create_optimizer factory
"""

from __future__ import annotations

import pytest
import torch

from serenity.pipeline.bucket import Bucket
from serenity.pipeline.buckets import BucketBatchSampler, BucketManager
from serenity.pipeline.masks import (
    generate_full_mask,
    generate_random_mask,
    resize_mask_to_latent,
)
from serenity.training.optimizers import OptimizerType, create_optimizer


# ---------------------------------------------------------------------------
# Module 1: pipeline/buckets.py
# ---------------------------------------------------------------------------


def test_bucket_manager_assigns_to_correct_ar_bucket():
    """Samples should land in the bucket with the closest aspect ratio."""
    landscape = Bucket(width=768, height=512)
    square = Bucket(width=512, height=512)
    portrait = Bucket(width=512, height=768)
    mgr = BucketManager(buckets=[landscape, square, portrait])

    # Wide image -> landscape bucket
    mgr.assign_sample(0, width=800, height=500)
    assert 0 in landscape.samples
    assert 0 not in square.samples

    # Square-ish image -> square bucket
    mgr.assign_sample(1, width=520, height=510)
    assert 1 in square.samples

    # Tall image -> portrait bucket
    mgr.assign_sample(2, width=400, height=700)
    assert 2 in portrait.samples


def test_bucket_batch_sampler_drop_last_drops_partial():
    """With drop_last=True, partial batches must be excluded."""
    bucket = Bucket(width=512, height=512, batch_size=4)
    # Assign 7 samples -> should produce 1 full batch (4), drop remainder (3)
    for i in range(7):
        bucket.samples.append(i)

    mgr = BucketManager(buckets=[bucket])
    sampler = BucketBatchSampler(mgr, batch_size=4, shuffle=False, drop_last=True)

    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 4
    assert len(sampler) == 1


def test_bucket_manager_empty_buckets():
    """Operations on empty bucket list should not crash."""
    mgr = BucketManager(buckets=[])

    # select_bucket returns None when no buckets
    assert mgr.select_bucket(512, 512) is None

    # assign_sample returns None when no buckets
    assert mgr.assign_sample(0, 512, 512) is None

    # get_nonempty_buckets is empty
    assert mgr.get_nonempty_buckets() == []

    # total_samples is zero
    assert mgr.total_samples() == 0

    # clear on empty list doesn't crash
    mgr.clear()


def test_bucket_manager_get_nonempty_filters_correctly():
    """Only buckets with assigned samples appear in nonempty list."""
    b1 = Bucket(width=512, height=512)
    b2 = Bucket(width=768, height=512)
    b3 = Bucket(width=512, height=768)
    mgr = BucketManager(buckets=[b1, b2, b3])

    mgr.assign_sample(0, 512, 512)  # -> b1 (square)
    mgr.assign_sample(1, 512, 500)  # -> b1 (close to square)

    nonempty = mgr.get_nonempty_buckets()
    assert len(nonempty) == 1
    assert nonempty[0] is b1
    assert mgr.total_samples() == 2


def test_bucket_manager_clear_removes_all_assignments():
    """clear() should empty all sample lists."""
    b1 = Bucket(width=512, height=512)
    mgr = BucketManager(buckets=[b1])
    mgr.assign_sample(0, 512, 512)
    mgr.assign_sample(1, 500, 500)
    assert mgr.total_samples() == 2

    mgr.clear()
    assert mgr.total_samples() == 0
    assert mgr.get_nonempty_buckets() == []


def test_bucket_batch_sampler_no_drop_last_keeps_partial():
    """With drop_last=False, partial batches are retained."""
    bucket = Bucket(width=512, height=512, batch_size=3)
    for i in range(5):
        bucket.samples.append(i)

    mgr = BucketManager(buckets=[bucket])
    sampler = BucketBatchSampler(mgr, batch_size=3, shuffle=False, drop_last=False)

    batches = list(sampler)
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2
    assert len(sampler) == 2


def test_bucket_batch_sampler_empty_manager():
    """Iterating over sampler with no samples yields nothing."""
    mgr = BucketManager(buckets=[Bucket(width=512, height=512)])
    sampler = BucketBatchSampler(mgr, batch_size=2, shuffle=False, drop_last=False)

    batches = list(sampler)
    assert batches == []
    assert len(sampler) == 0


def test_bucket_manager_from_config():
    """from_config creates sensible auto-generated buckets."""
    mgr = BucketManager.from_config(
        base_resolution=512, min_dim=256, max_dim=1024, quantize=64,
    )
    assert len(mgr.buckets) > 0
    # All buckets should have valid positive dimensions
    for b in mgr.buckets:
        assert b.width >= 256
        assert b.height >= 256
        assert b.width <= 1024
        assert b.height <= 1024


# ---------------------------------------------------------------------------
# Module 2: pipeline/masks.py
# ---------------------------------------------------------------------------


def test_generate_full_mask_shape_and_value():
    """Full mask should be [1, H, W] filled with the requested value."""
    mask = generate_full_mask(64, 128, value=0.75)
    assert mask.shape == (1, 64, 128)
    assert mask.dtype == torch.float32
    assert torch.allclose(mask, torch.tensor(0.75))


def test_generate_random_mask_binary_values():
    """Random mask should contain only 0.0 and 1.0."""
    mask = generate_random_mask(32, 32)
    assert mask.shape == (1, 32, 32)
    unique = torch.unique(mask)
    for v in unique:
        assert v.item() in (0.0, 1.0)


def test_resize_mask_to_latent_downscales():
    """resize_mask_to_latent with 1/8 scale should shrink dims by 8x."""
    mask = generate_full_mask(64, 128, value=1.0)
    latent = resize_mask_to_latent(mask, scale_factor=0.125)
    assert latent.shape == (1, 8, 16)


def test_generate_full_mask_zero_value():
    """Full mask with value=0.0 should be all zeros."""
    mask = generate_full_mask(16, 16, value=0.0)
    assert mask.sum().item() == 0.0


def test_generate_random_mask_threshold_extremes():
    """Threshold=0.0 should give all-zero mask, threshold=1.0 all-one."""
    mask_zero = generate_random_mask(16, 16, threshold=0.0)
    assert mask_zero.sum().item() == 0.0

    mask_one = generate_random_mask(16, 16, threshold=1.0)
    assert mask_one.sum().item() == 16 * 16


def test_resize_mask_to_latent_2d_input():
    """resize_mask_to_latent should handle 2D [H, W] tensors."""
    mask_2d = torch.ones(64, 128)
    result = resize_mask_to_latent(mask_2d, scale_factor=0.25)
    assert result.shape == (16, 32)


def test_resize_mask_to_latent_preserves_values():
    """A uniform mask should remain uniform after nearest-mode resize."""
    mask = generate_full_mask(64, 64, value=0.5)
    latent = resize_mask_to_latent(mask, scale_factor=0.5)
    assert torch.allclose(latent, torch.tensor(0.5))


# ---------------------------------------------------------------------------
# Module 3: training/optimizers.py
# ---------------------------------------------------------------------------


def test_create_optimizer_adamw_returns_correct_type():
    """ADAMW should produce a torch.optim.AdamW instance."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = create_optimizer(OptimizerType.ADAMW, params, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_optimizer_type_is_schedule_free():
    """Schedule-free variants must report is_schedule_free=True."""
    assert OptimizerType.SCHEDULE_FREE_ADAMW.is_schedule_free
    assert OptimizerType.SCHEDULE_FREE_SGD.is_schedule_free
    assert OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE.is_schedule_free
    # Standard types should be False
    assert not OptimizerType.ADAMW.is_schedule_free
    assert not OptimizerType.SGD.is_schedule_free


def test_optimizer_type_is_adaptive():
    """Adaptive optimizers must report is_adaptive=True."""
    assert OptimizerType.PRODIGY.is_adaptive
    assert OptimizerType.DADAPT_ADAM.is_adaptive
    assert not OptimizerType.ADAMW.is_adaptive
    assert not OptimizerType.SGD.is_adaptive


def test_create_optimizer_sgd_returns_correct_type():
    """SGD should produce a torch.optim.SGD instance."""
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    opt = create_optimizer(OptimizerType.SGD, params, lr=0.01, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)


def test_create_optimizer_adam_returns_correct_type():
    """ADAM should produce a torch.optim.Adam instance."""
    params = [torch.nn.Parameter(torch.randn(3))]
    opt = create_optimizer(OptimizerType.ADAM, params, lr=1e-4)
    assert isinstance(opt, torch.optim.Adam)


def test_create_optimizer_from_string():
    """String optimizer type should work the same as enum."""
    params = [torch.nn.Parameter(torch.randn(2))]
    opt = create_optimizer("adamw", params, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_create_optimizer_invalid_string_raises():
    """An unrecognized string should raise ValueError."""
    params = [torch.nn.Parameter(torch.randn(2))]
    with pytest.raises(ValueError):
        create_optimizer("nonexistent_optimizer_xyz", params, lr=1e-3)


def test_optimizer_type_requires_optional_dep():
    """Built-in types should not require optional deps; others should."""
    assert not OptimizerType.ADAM.requires_optional_dep
    assert not OptimizerType.ADAMW.requires_optional_dep
    assert not OptimizerType.SGD.requires_optional_dep
    assert OptimizerType.ADAM_8BIT.requires_optional_dep
    assert OptimizerType.PRODIGY.requires_optional_dep
    assert OptimizerType.SCHEDULE_FREE_ADAMW.requires_optional_dep


def test_create_optimizer_unknown_kwargs_ignored():
    """Extra unknown kwargs should be silently dropped, not raise."""
    params = [torch.nn.Parameter(torch.randn(2))]
    opt = create_optimizer(
        OptimizerType.ADAMW, params, lr=1e-3,
        totally_fake_param=42, another_fake=True,
    )
    assert isinstance(opt, torch.optim.AdamW)
