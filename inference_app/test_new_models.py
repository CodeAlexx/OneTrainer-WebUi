#!/usr/bin/env python3
"""
Test script for new model loaders: Kandinsky 5, OmniGen2, Lumina 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_kandinsky5():
    """Test Kandinsky 5 loading."""
    print("\n" + "="*60)
    print("Testing Kandinsky 5 Video loading...")
    print("="*60)

    try:
        from backend.app import InferenceEngine, LoadModelRequest, ModelType

        engine = InferenceEngine()

        # Test video model
        request = LoadModelRequest(
            model_path="/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors",
            model_type=ModelType.KANDINSKY_5_VIDEO,
            precision="bf16"
        )

        result = engine.load_model(request)
        print(f"Load result: {result}")

        if result.get("success"):
            print("Kandinsky 5 Video loaded successfully")
            return True
        else:
            print(f"Failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_omnigen2():
    """Test OmniGen2 loading."""
    print("\n" + "="*60)
    print("Testing OmniGen2 loading...")
    print("="*60)

    try:
        from backend.app import InferenceEngine, LoadModelRequest, ModelType

        engine = InferenceEngine()

        request = LoadModelRequest(
            model_path="BAAI/OmniGen2",  # HuggingFace model ID
            model_type=ModelType.OMNIGEN_2,
            precision="bf16"
        )

        result = engine.load_model(request)
        print(f"Load result: {result}")

        if result.get("success"):
            print("OmniGen2 loaded successfully")
            return True
        else:
            print(f"Failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lumina2():
    """Test Lumina 2 loading."""
    print("\n" + "="*60)
    print("Testing Lumina 2 loading...")
    print("="*60)

    try:
        from backend.app import InferenceEngine, LoadModelRequest, ModelType

        engine = InferenceEngine()

        request = LoadModelRequest(
            model_path="Alpha-VLLM/Lumina-Image-2.0",  # HuggingFace model ID
            model_type=ModelType.LUMINA_2,
            precision="bf16"
        )

        result = engine.load_model(request)
        print(f"Load result: {result}")

        if result.get("success"):
            print("Lumina 2 loaded successfully")
            return True
        else:
            print(f"Failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing New Model Loaders")
    print("="*60)

    results = {}

    # Test each model
    results["Kandinsky 5 Video"] = test_kandinsky5()
    results["OmniGen2"] = test_omnigen2()
    results["Lumina 2"] = test_lumina2()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, success in results.items():
        status = "" if success else ""
        print(f"{status} {name}: {'PASSED' if success else 'FAILED'}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
