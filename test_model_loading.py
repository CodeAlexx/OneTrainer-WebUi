#!/usr/bin/env python3
"""
Test script to diagnose model loading issues in inference_service.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test each model type
TEST_MODELS = [
    ("FLUX_DEV_1", "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors"),
    ("STABLE_DIFFUSION_XL_10_BASE", "/home/alex/SwarmUI/Models/diffusion_models/lustifySDXLNSFW_ggwpV7.safetensors"),
    ("CHROMA_1", "lodestones/Chroma1-HD"),
    ("Z_IMAGE", "/home/alex/SwarmUI/Models/diffusion_models/z_image_de_turbo_v1_bf16.safetensors"),
]

from web_ui.backend.services.inference_service import InferenceService

service = InferenceService()

for model_type, model_path in TEST_MODELS:
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    try:
        result = service.load_model(model_path, model_type)
        print(f"Result: {result}")
        
        if result.get('success'):
            print("SUCCESS - Model loaded!")
            # Unload to free memory
            service.unload_model()
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
    except Exception as e:
        import traceback
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
    
    # Clean up
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "="*60)
print("Testing complete")
