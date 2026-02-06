import torch
import torch.nn as nn
from diffusers import Transformer2DModel

def inspect_model():
    # Attempt to create a Z-Image like transformer
    # 6B model, but we can use a small config for inspection
    try:
        from diffusers import ZImageTransformer2DModel
        print("Found ZImageTransformer2DModel")
        # Just creating a default one for structure check
        model = ZImageTransformer2DModel()
    except ImportError:
        print("ZImageTransformer2DModel not found, trying generic Transformer2DModel")
        model = Transformer2DModel()

    print("\nModel structure:")
    for name, child in model.named_children():
        print(f"Child: {name} ({type(child)})")
        if name == 'layers' or name == 'transformer_blocks':
            block = child[0]
            print(f"\nInspecting first block: {type(block)}")
            print("Parameters:")
            for p_name, p in block.named_parameters():
                print(f"  P: {p_name} {p.shape} {p.device}")
            print("Buffers:")
            for b_name, b in block.named_buffers():
                print(f"  B: {b_name} {b.shape} {b.device}")
            break

if __name__ == "__main__":
    inspect_model()
