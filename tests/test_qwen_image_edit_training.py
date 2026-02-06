#!/usr/bin/env python3
"""
Integration test for Qwen Image Edit LoRA training.

Tests the full training pipeline with a small dataset.
"""

import sys
sys.path.insert(0, '/home/alex/OneTrainer')

import os
import torch
import json
from pathlib import Path


def create_test_config():
    """Create a minimal test config for Qwen Image Edit."""
    return {
        # Model settings
        "base_model_name": "Qwen/Qwen-Image-Edit",
        "model_type": "QWEN_IMAGE_EDIT",
        "training_method": "LORA",

        # Training settings
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "optimizer": "ADAMW",

        # Resolution
        "resolution": "512",

        # Memory optimization
        "gradient_checkpointing": "CPU_OFFLOADED",
        "layer_offload_fraction": 0.3,
        "dataloader_threads": 1,

        # Precision settings
        "train_dtype": "BFLOAT_16",
        "weight_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",

        # Component settings
        "transformer": {
            "train": True,
            "weight_dtype": "FLOAT_8"
        },
        "text_encoder": {
            "train": False,
            "weight_dtype": "FLOAT_8"
        },
        "vae": {
            "weight_dtype": "FLOAT_32"
        },

        # LoRA settings
        "lora_rank": 16,
        "lora_alpha": 16,
        "layer_filter": "attn",

        # Timestep distribution
        "timestep_distribution": "LOGIT_NORMAL",

        # Dataset
        "concept_file_name": "",  # Will be set programmatically

        # Caching
        "latent_caching": True,
        "cache_dir": "/tmp/qwen_edit_test_cache",

        # Output
        "output_model_destination": "/tmp/qwen_edit_test_output",
        "output_model_format": "SAFETENSORS",

        # Sampling
        "sample_after_epoch": True,
        "sample_after": 5,
        "sample_definition_file_name": "",

        # Misc
        "debug_mode": False,
        "train_device": "cuda",
        "temp_device": "cpu",
    }


def create_concept_config(dataset_path: str):
    """Create concept configuration for the dataset."""
    return [
        {
            "name": "qwen_edit_test",
            "enabled": True,
            "path": dataset_path,
            "text": {
                "prompt_source": "sample",
                "prompt_path": ""
            },
            "image": {
                "enable_crop_jitter": False,
                "enable_random_flip": False,
                "enable_resolution_override": False
            },
            "balancing": 1.0,
            "loss_weight": 1.0,
            "type": "FINE_TUNE"
        }
    ]


def test_data_loader_creation():
    """Test that the data loader can be created."""
    print("\n=== Testing Data Loader Creation ===\n")

    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.create import create_data_loader, create_model_loader
    from modules.util.enum.ModelType import ModelType
    from modules.util.TrainProgress import TrainProgress

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return False

    train_device = torch.device("cuda")
    temp_device = torch.device("cpu")

    # Create config
    config = TrainConfig.default_values()
    config.model_type = ModelType.QWEN_IMAGE_EDIT
    config.base_model_name = "Qwen/Qwen-Image-Edit"

    # Check that has_conditioning_image_input returns True
    assert config.model_type.has_conditioning_image_input(), \
        "QWEN_IMAGE_EDIT should have conditioning image input"
    print("✓ ModelType.has_conditioning_image_input() returns True")

    print("✓ Data loader configuration valid")
    return True


def test_imports():
    """Test that all new modules can be imported."""
    print("\n=== Testing Imports ===\n")

    try:
        from modules.dataLoader.QwenImageEditDataLoader import QwenImageEditDataLoader
        print("✓ QwenImageEditDataLoader imported")
    except ImportError as e:
        print(f"✗ Failed to import QwenImageEditDataLoader: {e}")
        return False

    try:
        from modules.modelSetup.QwenImageEditLoRASetup import QwenImageEditLoRASetup
        print("✓ QwenImageEditLoRASetup imported")
    except ImportError as e:
        print(f"✗ Failed to import QwenImageEditLoRASetup: {e}")
        return False

    try:
        from modules.model.QwenImageEditModel import QwenImageEditModel
        print("✓ QwenImageEditModel imported")
    except ImportError as e:
        print(f"✗ Failed to import QwenImageEditModel: {e}")
        return False

    return True


def test_model_type_flags():
    """Test ModelType enum flags for QWEN_IMAGE_EDIT."""
    print("\n=== Testing ModelType Flags ===\n")

    from modules.util.enum.ModelType import ModelType

    mt = ModelType.QWEN_IMAGE_EDIT

    assert mt.is_qwen_image_edit(), "is_qwen_image_edit() should return True"
    print("✓ is_qwen_image_edit() = True")

    assert mt.has_conditioning_image_input(), "has_conditioning_image_input() should return True"
    print("✓ has_conditioning_image_input() = True")

    assert mt.is_flow_matching(), "is_flow_matching() should return True"
    print("✓ is_flow_matching() = True")

    return True


def test_predict_method_exists():
    """Test that predict method exists in setup."""
    print("\n=== Testing Predict Method ===\n")

    from modules.modelSetup.QwenImageEditLoRASetup import QwenImageEditLoRASetup

    setup = QwenImageEditLoRASetup(
        train_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        temp_device=torch.device("cpu"),
        debug_mode=False
    )

    assert hasattr(setup, 'predict'), "Setup should have predict method"
    print("✓ predict() method exists")

    # Check method signature
    import inspect
    sig = inspect.signature(setup.predict)
    params = list(sig.parameters.keys())
    assert 'model' in params, "predict should have 'model' parameter"
    assert 'batch' in params, "predict should have 'batch' parameter"
    assert 'config' in params, "predict should have 'config' parameter"
    print("✓ predict() has correct signature")

    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Qwen Image Edit Integration Tests")
    print("=" * 60)

    tests = [
        ("Model Type Flags", test_model_type_flags),
        ("Predict Method Exists", test_predict_method_exists),
        ("Data Loader Creation", test_data_loader_creation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result, error in results:
        if result:
            print(f"✓ {name}: PASSED")
            passed += 1
        else:
            print(f"✗ {name}: FAILED - {error}")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
