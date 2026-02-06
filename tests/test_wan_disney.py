#!/usr/bin/env python3
"""Quick 10-step test of Wan 2.2 LoRA training on Disney dataset."""

import sys
import os
import torch
import traceback
from pathlib import Path
import time

# Add modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.util.config.TrainConfig import TrainConfig, QuantizationConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util import create
from modules.modelSetup.WanLoRASetup import WanLoRASetup
from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
from modules.util.TrainProgress import TrainProgress

def run_training_test():
    print(f"\n============================================================")
    print(f"Running Wan 2.2 LoRA Training: Quick 10-Step Test")
    print(f"============================================================")

    train_device = torch.device('cuda')
    temp_device = torch.device('cpu') 
    
    # Config
    config = TrainConfig.default_values()
    config.model_type = ModelType.WAN_T2V
    config.training_method = TrainingMethod.LORA
    config.learning_rate = 1e-4
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.mixed_precision = DataType.BFLOAT_16
    config.network_dim = 16
    config.network_alpha = 16
    
    config.epochs = 1
    
    # Cache setup - REQUIRED for 24GB VRAM with dual transformer
    config.cache_dir = os.path.abspath("tests/cache_wan_disney_test")
    config.clean_cache = True  # Fresh cache for test
    config.latent_caching = True  # Cache latents and text embeddings, keeps T5 on CPU
    
    # Disable text encoder training - REQUIRED for 24GB VRAM
    # This keeps T5 on CPU after caching is complete
    config.text_encoder.train = False
    
    # Enable gradient checkpointing - REQUIRED for 24GB VRAM
    from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
    config.gradient_checkpointing = GradientCheckpointingMethod.ON
    
    # Dataset - Use Disney videos
    concept = ConceptConfig.default_values()
    concept.name = "disney_videos"
    concept.path = "/home/alex/disney/videos"  # Updated path
    concept.enabled = True
    from modules.util.enum.ConceptType import ConceptType
    concept.type = ConceptType.STANDARD 
    
    concept.image.enable_resolution_override = True
    concept.image.resolution_override = "512"
    
    # Single frame training - REQUIRED for 24GB VRAM
    # Video training with multiple frames uses too much memory
    config.frames = "1"
    
    config.concepts = [concept]

    # Model Paths - Wan 2.2
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--Wan2.2-T2V-A14B-Diffusers-bf16/snapshots/fe9c6303ddfdefcae7f06ec7fc7c244bada46cc3")
    t5_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--umt5_xxl_encoder/snapshots/0dc787b18485f27e1eb5a678127868becc2401ef/text_encoder")
    vae_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--wan2.1-vae/snapshots/1ec3f48e1ac74941bb0b9b61cb107de654c7af4b")
    
    model_names = ModelNames(model_path)
    model_names.text_encoder_model = t5_path
    model_names.vae_model = vae_path

    # Weights - Use NF4 quantization for 14B transformer
    weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
    weight_dtypes.transformer = DataType.NFLOAT_4  # NF4 quantization for VRAM
    weight_dtypes.text_encoder = DataType.BFLOAT_16
    weight_dtypes.vae = DataType.BFLOAT_16
    
    quantization = QuantizationConfig.default_values() 

    model = None
    optimizer = None
    loader = None
    setup = None

    try:
        # 1. Load Model
        print("Loading Wan 2.2 model...")
        loader = create.create_model_loader(config.model_type, config.training_method)
        model = loader.load(config.model_type, model_names, weight_dtypes, quantization)
        model.train_config = config
        print("✓ Model loaded!")
        
        # 2. Setup
        print("Setting up LoRA training...")
        setup = create.create_model_setup(
            config.model_type, train_device, temp_device, config.training_method, debug_mode=True
        )
        setup.setup_optimizations(model, config)
        setup.setup_train_device(model, config)
        setup.setup_model(model, config)
        
        # NOTE: Don't call model.to() - setup_train_device handles device placement
        # For dual transformer with low_vram, only one transformer is on GPU at a time
        print("✓ LoRA setup complete!")

        # 3. Data Loader
        print("Creating data loader for Disney videos...")
        progress = TrainProgress()
        loader = WanBaseDataLoader(train_device, temp_device, config, model, progress)
        data_loader = loader.get_data_loader()
        
        print("Starting next epoch (caching)...")
        loader.get_data_set().start_next_epoch()
        print("✓ Data loader ready!")
        
        # After caching: ensure transformer_2 is on GPU for training
        # Use transformer_to() to also move LoRA modules
        if model.is_dual_transformer:
            print("Moving transformer and LoRA to GPU for training...")
            model.transformer_to(train_device)
            torch.cuda.empty_cache()
        
        # 4. Optimizer
        print("Creating optimizer...")
        params = setup.create_parameters(model, config)
        
        import bitsandbytes as bnb
        if hasattr(params, 'parameters'):
            param_list = params.parameters()
        else:
            param_list = []
            
        optimizer = bnb.optim.AdamW8bit(param_list, lr=config.learning_rate)
        print("✓ Optimizer ready!")
        
        # 5. Training Loop - 10 steps
        print("\n" + "="*60)
        print("Starting 10-step training test...")
        print("="*60 + "\n")
        
        data_iter = iter(data_loader)
        
        start_time = time.time()
        
        for i in range(10):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart epoch
                print(f"Epoch finished at step {i}. Restarting iterator.")
                loader.get_data_set().start_next_epoch()
                data_iter = iter(data_loader)
                batch = next(data_iter)
                
            optimizer.zero_grad()
            
            # Use caption from batch or default
            if 'prompt' not in batch or not batch['prompt'] or not batch['prompt'][0]:
                batch['prompt'] = ["a disney cartoon video clip"]

            # Forward 
            model_output_data = setup.predict(model, batch, config, progress)
            
            # Loss
            loss = setup.calculate_loss(model, batch, model_output_data, config)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            elapsed = time.time() - start_time
            print(f"Step {i+1}/10: Loss = {loss.item():.4f} (Time: {elapsed:.2f}s)")
            
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"✓ Training test COMPLETE!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time/step: {total_time/10:.2f}s")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training FAILED with exception: {e}")
        traceback.print_exc()
        raise e

    finally:
        print("Cleaning up...")
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if loader is not None:
            del loader
        if setup is not None:
            del setup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_training_test()
