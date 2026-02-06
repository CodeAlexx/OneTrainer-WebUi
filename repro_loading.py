
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Mock torch if needed or just let it fail naturally
import torch
print(f"Torch version: {torch.__version__}")

try:
    from web_ui.backend.services.inference_service import InferenceService
    
    service = InferenceService()
    
    # Test Flux loading (safetensors)
    flux_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors"
    model_type = "FLUX_DEV_1"
    
    print(f"Attempting to load {model_type} from {flux_path}")
    result = service.load_model(flux_path, model_type)
    print(f"Result: {result}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
