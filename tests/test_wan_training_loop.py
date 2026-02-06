
import sys
import os
import torch
import traceback
from pathlib import Path

# Add modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.util.config.TrainConfig import TrainConfig, QuantizationConfig
from modules.util.config.ConceptConfig import ConceptConfig, ConceptImageConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.Optimizer import Optimizer
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util import create
from modules.modelSetup.WanLoRASetup import WanLoRASetup
from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
from modules.util.TrainProgress import TrainProgress

def run_training_test(lr_name, learning_rate):
    print(f"\n============================================================")
    print(f"Running Wan LoRA Training Test: {lr_name} (LR={learning_rate})")
    print(f"============================================================")

    train_device = torch.device('cuda')
    # Use CPU for temp device to avoid OOM during caching (move models to CPU when caching)
    temp_device = torch.device('cpu') 
    
    # Config
    config = TrainConfig.default_values()
    config.model_type = ModelType.WAN_T2V
    config.training_method = TrainingMethod.LORA
    config.learning_rate = learning_rate
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.mixed_precision = DataType.BFLOAT_16
    config.network_dim = 16
    config.network_alpha = 16
    
    config.epochs = 1
    
    # Cache setup 
    config.cache_dir = os.path.abspath("tests/cache_wan_test")
    # Clean cache False to reuse previous run if possible
    config.clean_cache = False 
    
    # Dataset
    concept = ConceptConfig.default_values()
    concept.name = "disney_test"
    concept.path = os.path.abspath("datasets/Disney-VideoGeneration-Dataset/videos")
    concept.enabled = True
    from modules.util.enum.ConceptType import ConceptType
    concept.type = ConceptType.STANDARD 
    
    concept.image.enable_resolution_override = True
    concept.image.resolution_override = "512"
    
    config.concepts = [concept]

    # Model Paths
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--Wan2.2-T2V-A14B-Diffusers-bf16/snapshots/fe9c6303ddfdefcae7f06ec7fc7c244bada46cc3")
    t5_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--umt5_xxl_encoder/snapshots/0dc787b18485f27e1eb5a678127868becc2401ef/text_encoder")
    vae_path = os.path.expanduser("~/.cache/huggingface/hub/models--ai-toolkit--wan2.1-vae/snapshots/1ec3f48e1ac74941bb0b9b61cb107de654c7af4b")
    
    model_names = ModelNames(model_path)
    model_names.text_encoder_model = t5_path
    model_names.vae_model = vae_path

    # Weights 
    weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
    weight_dtypes.transformer = DataType.NFLOAT_4 
    weight_dtypes.text_encoder = DataType.BFLOAT_16
    weight_dtypes.vae = DataType.BFLOAT_16
    
    quantization = QuantizationConfig.default_values() 

    model = None
    optimizer = None
    loader = None
    setup = None

    try:
        # 1. Load Model
        print("Loading model...")
        loader = create.create_model_loader(config.model_type, config.training_method)
        model = loader.load(config.model_type, model_names, weight_dtypes, quantization)
        model.train_config = config
        
        # 2. Setup
        print("Setting up model...")
        setup = create.create_model_setup(
            config.model_type, train_device, temp_device, config.training_method, debug_mode=True
        )
        setup.setup_optimizations(model, config)
        setup.setup_train_device(model, config)
        setup.setup_model(model, config)
        
        model.to(train_device)

        # 3. Data Loader
        print("Creating data loader...")
        progress = TrainProgress()
        loader = WanBaseDataLoader(train_device, temp_device, config, model, progress)
        data_loader = loader.get_data_loader()
        
        print("Starting next epoch (caching)...")
        loader.get_data_set().start_next_epoch()
        
        # 4. Optimizer
        print("Creating optimizer...")
        params = setup.create_parameters(model, config)
        
        import bitsandbytes as bnb
        if hasattr(params, 'parameters'):
            param_list = params.parameters()
        else:
            param_list = []
            
        optimizer = bnb.optim.AdamW8bit(param_list, lr=learning_rate)
        
        # 5. Training Loop
        print("Starting training loop (5 steps)...")
        
        data_iter = iter(data_loader)
        
        for i in range(5):
            try:
                batch = next(data_iter)
            except StopIteration:
                print("StopIteration reached early.")
                break
                
            optimizer.zero_grad()
            
            # FORCE PROMPT if missing/empty
            if 'prompt' not in batch or not batch['prompt'] or batch['prompt'] == ['']:
                # print(f"DEBUG: Injecting dummy prompt for step {i}")
                batch['prompt'] = ["a video of a disney character"]

            # Forward 
            model_output_data = setup.predict(model, batch, config, progress)
            
            # Loss
            loss = setup.calculate_loss(model, batch, model_output_data, config)
            
            print(f"Step {i+1}: Loss = {loss.item()}")
            
            loss.backward()
            optimizer.step()
            
        print(f"Test {lr_name} COMPLETE")
        
    except Exception as e:
        print(f"Test {lr_name} FAILED with exception: {e}")
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
    try:
        run_training_test("HIGH", 1e-4) 
    except Exception as e:
        print(f"HIGH Test Failed (Outer): {e}")
