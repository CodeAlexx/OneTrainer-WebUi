
import os
from .vae2_1 import Wan2_1_VAE
from .vae2_2 import Wan2_2_VAE

def WanVAE(vae_pth, device='cpu', dtype=None, **kwargs):
    """
    Factory function to return the correct VAE model based on the checkpoint filename or configuration.
    """
    # Simple heuristic based on filename
    filename = os.path.basename(vae_pth)
    
    if "Wan2.1" in filename:
        return Wan2_1_VAE(vae_pth=vae_pth, device=device, dtype=dtype, **kwargs)
    elif "Wan2.2" in filename:
        return Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype, **kwargs)
    else:
        # Fallback or default?
        print(f"Warning: Could not determine VAE version from filename '{filename}'. Defaulting to Wan2.1")
        return Wan2_1_VAE(vae_pth=vae_pth, device=device, dtype=dtype, **kwargs)
