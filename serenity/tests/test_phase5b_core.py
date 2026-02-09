"""Tests for core callbacks, concept config, and augmentation pipeline."""

from __future__ import annotations

import torch
import pytest

from serenity.core.callbacks import TrainCallbacks
from serenity.core.concept_config import ConceptConfig, ConceptImageConfig, ConceptTextConfig
from serenity.pipeline.augmentations import (
    apply_random_flip,
    apply_random_saturation,
    apply_random_brightness,
    apply_random_contrast,
    apply_crop_jitter,
    apply_augmentations,
    compose_from_config,
    AugmentationResult,
)


# ---------------------------------------------------------------------------
# Module 1: core/callbacks.py
# ---------------------------------------------------------------------------


def test_callbacks_register_and_dispatch_on_step():
    """Register on_step handler and verify it receives correct args."""
    received = []
    cb = TrainCallbacks()
    cb.register_on_step(lambda step, loss: received.append((step, loss)))

    cb.on_step(42, 0.125)

    assert received == [(42, 0.125)]


def test_callbacks_multiple_handlers_dispatch_order():
    """Multiple handlers fire in registration order."""
    order = []
    cb = TrainCallbacks()
    cb.register_on_train_start(lambda: order.append("first"))
    cb.register_on_train_start(lambda: order.append("second"))

    cb.on_train_start()

    assert order == ["first", "second"]


def test_callbacks_exception_suppressed():
    """A raising handler does not crash dispatch or block later handlers."""
    results = []
    cb = TrainCallbacks()
    cb.register_on_epoch_start(lambda e: (_ for _ in ()).throw(ValueError("boom")))
    cb.register_on_epoch_start(lambda e: results.append(e))

    # Should not raise
    cb.on_epoch_start(5)

    assert results == [5]


def test_callbacks_handler_count_and_clear():
    """handler_count reflects total across all events; clear() resets to 0."""
    cb = TrainCallbacks()
    cb.register_on_step(lambda s, l: None)
    cb.register_on_log(lambda s, m: None)
    cb.register_on_checkpoint(lambda s, p: None)

    assert cb.handler_count == 3

    cb.clear()

    assert cb.handler_count == 0


def test_callbacks_all_eight_events_dispatch():
    """Integration: all 8 event types can register and dispatch."""
    cb = TrainCallbacks()
    fired = set()

    cb.register_on_train_start(lambda: fired.add("train_start"))
    cb.register_on_train_end(lambda: fired.add("train_end"))
    cb.register_on_epoch_start(lambda e: fired.add("epoch_start"))
    cb.register_on_epoch_end(lambda e: fired.add("epoch_end"))
    cb.register_on_step(lambda s, l: fired.add("step"))
    cb.register_on_sample(lambda s, imgs: fired.add("sample"))
    cb.register_on_checkpoint(lambda s, p: fired.add("checkpoint"))
    cb.register_on_log(lambda s, m: fired.add("log"))

    cb.on_train_start()
    cb.on_train_end()
    cb.on_epoch_start(1)
    cb.on_epoch_end(1)
    cb.on_step(1, 0.5)
    cb.on_sample(1, ["img"])
    cb.on_checkpoint(1, "/tmp/ckpt")
    cb.on_log(1, {"loss": 0.5})

    assert fired == {
        "train_start", "train_end",
        "epoch_start", "epoch_end",
        "step", "sample", "checkpoint", "log",
    }


# ---------------------------------------------------------------------------
# Module 2: core/concept_config.py
# ---------------------------------------------------------------------------


def test_concept_config_from_dict_nested():
    """from_dict() converts nested image/text dicts into dataclass instances."""
    data = {
        "name": "faces",
        "path": "/data/faces",
        "image": {"enable_random_flip": True, "random_brightness_max_strength": 0.2},
        "text": {"prompt_source": "file", "tag_delimiter": ";"},
    }
    cfg = ConceptConfig.from_dict(data)

    assert cfg.name == "faces"
    assert isinstance(cfg.image, ConceptImageConfig)
    assert cfg.image.enable_random_flip is True
    assert cfg.image.random_brightness_max_strength == 0.2
    assert isinstance(cfg.text, ConceptTextConfig)
    assert cfg.text.prompt_source == "file"
    assert cfg.text.tag_delimiter == ";"


def test_concept_config_default_seed_unique():
    """Each ConceptConfig gets a unique random seed by default."""
    seeds = {ConceptConfig().seed for _ in range(50)}
    # With 2^31 range, 50 instances should all be unique
    assert len(seeds) == 50


def test_concept_config_post_init_dict_to_dataclass():
    """__post_init__ converts raw dict fields to dataclass instances."""
    cfg = ConceptConfig(
        name="style",
        image={"enable_fixed_flip": True},  # type: ignore[arg-type]
        text={"enable_tag_shuffling": True},  # type: ignore[arg-type]
    )

    assert isinstance(cfg.image, ConceptImageConfig)
    assert cfg.image.enable_fixed_flip is True
    assert isinstance(cfg.text, ConceptTextConfig)
    assert cfg.text.enable_tag_shuffling is True


def test_concept_config_from_dict_empty_sub_configs():
    """from_dict() with empty/missing sub-config dicts uses defaults."""
    cfg = ConceptConfig.from_dict({"name": "minimal"})

    assert cfg.name == "minimal"
    assert isinstance(cfg.image, ConceptImageConfig)
    assert cfg.image.enable_crop_jitter is True  # default
    assert isinstance(cfg.text, ConceptTextConfig)
    assert cfg.text.prompt_source == "sample"  # default


# ---------------------------------------------------------------------------
# Module 3: pipeline/augmentations.py
# ---------------------------------------------------------------------------


def test_apply_random_flip_fixed_mode():
    """Fixed flip reverses the width dimension of a [C,H,W] tensor."""
    tensor = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1,2,3]
    flipped, was_flipped = apply_random_flip(tensor, enable_fixed=True)

    assert was_flipped is True
    expected = torch.tensor([[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]])
    assert torch.equal(flipped, expected)


def test_apply_random_flip_disabled_returns_original():
    """With both enables False, tensor is returned unchanged."""
    tensor = torch.rand(3, 4, 4)
    result, was_flipped = apply_random_flip(tensor, enable_random=False, enable_fixed=False)

    assert was_flipped is False
    assert torch.equal(result, tensor)


def test_apply_random_saturation_single_channel_unchanged():
    """Saturation on a single-channel tensor returns input unchanged."""
    tensor = torch.rand(1, 8, 8)
    result = apply_random_saturation(tensor, enable_random=True, max_strength=0.5)

    assert torch.equal(result, tensor)


def test_apply_random_saturation_3channel_modifies():
    """Saturation on a 3-channel tensor with fixed mode changes values."""
    tensor = torch.rand(3, 8, 8)
    result = apply_random_saturation(tensor, enable_fixed=True, max_strength=0.5)

    # With non-zero strength, result should differ from input
    assert not torch.equal(result, tensor)
    assert result.shape == tensor.shape


def test_compose_from_config_partial_enables():
    """compose_from_config includes only enabled augmentations."""
    config = ConceptImageConfig(
        enable_fixed_flip=True,
        enable_random_brightness=True,
        random_brightness_max_strength=0.1,
        # Everything else disabled by default
    )
    chain = compose_from_config(config)
    tensor = torch.rand(3, 16, 16)
    result = chain(tensor)

    # Result should differ (flip + brightness applied)
    assert result.shape == tensor.shape
    # Verify flip happened: last dim should be reversed
    # Apply just flip to compare
    flipped_only, _ = apply_random_flip(tensor, enable_fixed=True)
    # The chain applies flip then brightness, so result != original
    assert not torch.equal(result, tensor)


def test_compose_from_config_nothing_enabled():
    """Config with all augmentations disabled produces identity transform."""
    config = ConceptImageConfig(
        enable_crop_jitter=False,
        enable_random_flip=False,
        enable_fixed_flip=False,
    )
    chain = compose_from_config(config)
    tensor = torch.rand(3, 16, 16)
    result = chain(tensor)

    assert torch.equal(result, tensor)


def test_apply_augmentations_integration():
    """apply_augmentations returns AugmentationResult with correct metadata."""
    config = ConceptImageConfig(
        enable_fixed_flip=True,
        enable_random_brightness=True,
        random_brightness_max_strength=0.05,
    )
    tensor = torch.rand(3, 16, 16)
    result = apply_augmentations(tensor, config)

    assert isinstance(result, AugmentationResult)
    assert result.was_flipped is True
    assert result.tensor.shape == tensor.shape
    # Flip + brightness means result differs from input
    assert not torch.equal(result.tensor, tensor)


def test_apply_crop_jitter_center_crop():
    """Disabled jitter performs center crop."""
    tensor = torch.arange(32.0).reshape(1, 4, 8)  # [1, 4, 8]
    cropped, (oy, ox) = apply_crop_jitter(tensor, crop_h=2, crop_w=4, enable=False)

    assert cropped.shape == (1, 2, 4)
    assert oy == 1  # (4 - 2) // 2
    assert ox == 2  # (8 - 4) // 2
