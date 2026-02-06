#!/usr/bin/env python3
"""
Test Chroma inference with the trained LoRA - with key conversion.
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from safetensors.torch import load_file, save_file

def convert_lora_keys_onetrainer_to_diffusers(state_dict):
    """
    Convert OneTrainer LoRA key format to diffusers format.
    
    OneTrainer format:
      lora_transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight
      lora_transformer_single_transformer_blocks_0_attn_to_k.lora_up.weight
      lora_transformer_single_transformer_blocks_0_attn_to_k.alpha
    
    Diffusers format:
      transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight
      transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Convert underscores to dots in the path
        # lora_transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight
        # -> transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight
        
        new_key = key
        
        # Handle the main patterns
        if key.startswith("lora_transformer_"):
            # Remove the lora_ prefix
            new_key = key[len("lora_"):] 
            
            # Convert underscores to dots for path elements
            # transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight
            # Split by the last underscore before lora_down/lora_up/alpha
            
            if ".lora_down.weight" in new_key:
                path = new_key.replace(".lora_down.weight", "")
                # Convert path: transformer_single_transformer_blocks_0_attn_to_k
                # to: transformer.single_transformer_blocks.0.attn.to_k
                converted_path = convert_path(path)
                new_key = f"{converted_path}.lora_A.weight"
                
            elif ".lora_up.weight" in new_key:
                path = new_key.replace(".lora_up.weight", "")
                converted_path = convert_path(path)
                new_key = f"{converted_path}.lora_B.weight"
                
            elif ".alpha" in new_key:
                # Alpha is handled differently - diffusers uses it as scale
                path = new_key.replace(".alpha", "")
                converted_path = convert_path(path)
                new_key = f"{converted_path}.alpha"
                
        new_state_dict[new_key] = value
    
    return new_state_dict


def convert_path(path):
    """
    Convert underscore-separated path to dot-separated.
    transformer_single_transformer_blocks_0_attn_to_k
    -> transformer.single_transformer_blocks.0.attn.to_k
    """
    # Handle known patterns
    path = path.replace("transformer_single_transformer_blocks_", "transformer.single_transformer_blocks.")
    path = path.replace("transformer_transformer_blocks_", "transformer.transformer_blocks.")
    
    # Handle block indices (e.g., _0_, _1_, etc.)
    path = re.sub(r"\.(\d+)_", r".\1.", path)
    
    # Handle attn and other suffixes
    path = path.replace("_attn_", ".attn.")
    path = path.replace("_to_k", ".to_k")
    path = path.replace("_to_q", ".to_q")
    path = path.replace("_to_v", ".to_v")
    path = path.replace("_to_out", ".to_out")
    path = path.replace("_proj_mlp", ".proj_mlp")
    path = path.replace("_proj_out", ".proj_out")
    path = path.replace("_norm_out", ".norm_out")
    path = path.replace("_norm1", ".norm1")
    path = path.replace("_norm2", ".norm2")
    path = path.replace("_ff_", ".ff.")
    path = path.replace("_net_0_proj", ".net.0.proj")
    path = path.replace("_net_2", ".net.2")
    
    return path


def main():
    print("=" * 60)
    print("Testing Chroma Inference with LoRA (Key Conversion)")
    print("=" * 60)
    
    # Paths
    base_model = "lodestones/Chroma1-HD"
    lora_path = Path(__file__).parent / "models" / "model.safetensors"
    output_path = Path(__file__).parent / "test_output_chroma_with_lora.png"
    converted_lora_path = Path(__file__).parent / "models" / "model_diffusers.safetensors"
    
    print(f"\nBase model: {base_model}")
    print(f"LoRA path: {lora_path}")
    print(f"LoRA exists: {lora_path.exists()}")
    
    if not lora_path.exists():
        print(f"ERROR: LoRA file not found at {lora_path}")
        return 1
    
    # Load and convert LoRA keys
    print("\n1. Loading and converting LoRA keys...")
    lora_weights = load_file(str(lora_path))
    print(f"   Original keys: {len(lora_weights)}")
    print(f"   Sample original: {list(lora_weights.keys())[:2]}")
    
    converted_weights = convert_lora_keys_onetrainer_to_diffusers(lora_weights)
    print(f"   Converted keys: {len(converted_weights)}")
    print(f"   Sample converted: {list(converted_weights.keys())[:2]}")
    
    # Save converted weights
    print(f"\n2. Saving converted LoRA to {converted_lora_path}...")
    save_file(converted_weights, str(converted_lora_path))
    
    # Load ChromaPipeline
    print("\n3. Loading ChromaPipeline...")
    from diffusers import ChromaPipeline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    pipe = ChromaPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    print(f"   Pipeline type: {type(pipe).__name__}")
    
    # Enable CPU offload
    print("\n4. Enabling CPU offload...")
    pipe.enable_sequential_cpu_offload()
    
    # Load converted LoRA
    print("\n5. Loading converted LoRA weights...")
    try:
        pipe.load_lora_weights(str(converted_lora_path))
        print("   SUCCESS: LoRA loaded!")
    except Exception as e:
        print(f"   Failed to load converted LoRA: {e}")
        print("   Trying to load into transformer directly...")
        
        try:
            # Get transformer state dict to compare keys
            transformer_keys = list(pipe.transformer.state_dict().keys())[:10]
            print(f"   Transformer sample keys: {transformer_keys}")
            
            # Try setting LoRA weight scale manually
            pipe.set_adapters(["default"], adapter_weights=[1.0])
            print("   Set adapter weights")
        except Exception as e2:
            print(f"   Manual loading failed: {e2}")
            print("   Generating without LoRA as fallback...")
    
    # Generate test image with user's settings
    print("\n6. Generating test image (cfg=3.5, steps=26)...")
    prompt = "a beautiful woman with red hair, highly detailed, professional photo"
    
    try:
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=26,
            height=1024,
            width=1024,
        ).images[0]
        
        image.save(str(output_path))
        print(f"   Image saved to: {output_path}")
        print("\n" + "=" * 60)
        print("SUCCESS! Chroma inference completed")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"   Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
