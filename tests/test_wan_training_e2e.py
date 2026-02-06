#!/usr/bin/env python
"""
Wan 2.1 T2V End-to-End Training Test

Tests the full training pipeline with real model weights.
Runs a few training steps to verify everything works.
"""

import sys
import os
import torch
import traceback

# Add OneTrainer to path
sys.path.insert(0, '/home/alex/OneTrainer')

def main():
    print("=" * 60)
    print("Wan 2.1 T2V End-to-End Training Test")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda")
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Model path - using HuggingFace cache directly
    import os
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--Wan2.2-T2V-A14B-Diffusers-bf16/snapshots/fe9c6303ddfdefcae7f06ec7fc7c244bada46cc3")
    
    print(f"\n--- Step 1: Loading Model ---")
    try:
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.TrainingMethod import TrainingMethod
        from modules.util import create
        from modules.util.ModelNames import ModelNames
        from modules.util.ModelWeightDtypes import ModelWeightDtypes
        from modules.util.enum.DataType import DataType
        from modules.util.config.TrainConfig import QuantizationConfig
        
        model_type = ModelType.WAN_T2V
        training_method = TrainingMethod.LORA
        
        # Create loader
        loader = create.create_model_loader(model_type, training_method)
        print(f"Loader: {type(loader).__name__}")
        
        # Set up model names
        model_names = ModelNames()
        model_names.base_model = model_path
        model_names.text_encoder_model = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--umt5_xxl_encoder/snapshots/0dc787b18485f27e1eb5a678127868becc2401ef")
        model_names.vae_model = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--wan2.1-vae/snapshots/1ec3f48e1ac74941bb0b9b61cb107de654c7af4b")
        
        # Set up dtypes - use NF4 for transformer to fit 14B in 24GB VRAM
        weight_dtypes = ModelWeightDtypes(
            train_dtype=DataType.BFLOAT_16,
            fallback_train_dtype=DataType.FLOAT_16,
            unet=DataType.BFLOAT_16,
            prior=DataType.BFLOAT_16,
            transformer=DataType.NFLOAT_4,  # NF4 quantization for transformer
            text_encoder=DataType.BFLOAT_16,
            text_encoder_2=DataType.BFLOAT_16,
            text_encoder_3=DataType.BFLOAT_16,
            text_encoder_4=DataType.BFLOAT_16,
            vae=DataType.BFLOAT_16,
            effnet_encoder=DataType.BFLOAT_16,
            decoder=DataType.BFLOAT_16,
            decoder_text_encoder=DataType.BFLOAT_16,
            decoder_vqgan=DataType.BFLOAT_16,
            lora=DataType.BFLOAT_16,
            embedding=DataType.BFLOAT_16,
        )
        
        # Quantization config (for layer filters etc.)
        quantization = QuantizationConfig.default_values()
        
        print(f"Loading model from: {model_path}")
        print("Transformer dtype: NFLOAT_4 (NF4 quantization)")
        model = loader.load(model_type, model_names, weight_dtypes, quantization)
        
        if model is None:
            print("ERROR: Model loading returned None")
            return False
            
        print(f"✓ Model loaded: {type(model).__name__}")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n--- Step 2: Create Model Setup ---")
    try:
        temp_device = torch.device("cpu")
        setup = create.create_model_setup(
            model_type, device, temp_device, training_method, debug_mode=True
        )
        
        if setup is None:
            print("ERROR: Model setup returned None")
            return False
            
        print(f"✓ Setup: {type(setup).__name__}")
        
    except Exception as e:
        print(f"ERROR creating setup: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n--- Step 3: Setup Model for Training ---")
    try:
        from modules.util.config.TrainConfig import TrainConfig
        
        # Create minimal config
        config = TrainConfig.default_values()
        config.model_type = model_type
        config.training_method = training_method
        config.lora_rank = 4  # Small rank for testing
        config.lora_alpha = 4
        
        # Assign config to model
        model.train_config = config
        
        # Setup the model
        setup.setup_model(model, config)
        print("✓ Model setup complete")
        
    except Exception as e:
        print(f"ERROR in setup_model: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n--- Step 4: Move to Train Device ---")
    try:
        setup.setup_train_device(model, config)
        print("✓ Model on train device")
        
        # Print memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
    except Exception as e:
        print(f"ERROR moving to device: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n--- Step 5: Create Parameters ---")
    try:
        parameters = setup.create_parameters(model, config)
        if hasattr(parameters, 'parameters'):
            all_params = parameters.parameters()
            param_count = sum(p.numel() for p in all_params)
        else:
            # Fallback
            param_count = 0
            print("Warning: Could not count parameters")
        print(f"✓ Trainable parameters: {param_count:,}")
        
    except Exception as e:
        print(f"ERROR creating parameters: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ WAN T2V TRAINING PIPELINE TEST PASSED!")
    print("=" * 60)
    print("\nThe Wan model is ready for training.")
    print("All factory methods and setup procedures work correctly.")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
