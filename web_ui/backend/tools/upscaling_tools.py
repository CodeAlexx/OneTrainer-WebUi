"""
Upscaling Tools - Real-ESRGAN
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).parent / "models"


class RealESRGANUpscaler:
    """Real-ESRGAN upscaler."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.upsampler = None
        self.model_name = "RealESRGAN_x4plus"

    def load(self, model_name: str = None, scale: int = 4):
        """Load Real-ESRGAN model."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model_name = model_name or self.model_name
            model_path = MODELS_DIR / f"{model_name}.pth"

            # Model architecture
            if 'x4plus' in model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4
            elif 'x2plus' in model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2
                )
                netscale = 2
            else:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4

            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                dni_weight=None,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                gpu_id=0
            )
            print(f"✅ Real-ESRGAN loaded ({model_name})")
            return True
        except Exception as e:
            print(f"❌ Failed to load Real-ESRGAN: {e}")
            return False

    def upscale(self, image: Image.Image, scale: float = 4.0) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.

        Args:
            image: PIL Image
            scale: Output scale factor

        Returns:
            Upscaled PIL Image
        """
        if self.upsampler is None:
            self.load()

        # Convert to numpy BGR
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]

        # Upscale
        output, _ = self.upsampler.enhance(img_np, outscale=scale)

        # Convert back to PIL RGB
        output_rgb = output[:, :, ::-1]
        return Image.fromarray(output_rgb)


class LanczosUpscaler:
    """Simple Lanczos upscaler (no model needed)."""

    def upscale(self, image: Image.Image, scale: float = 2.0) -> Image.Image:
        """Upscale using Lanczos interpolation."""
        new_size = (int(image.width * scale), int(image.height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)


# Singleton instance
_esrgan_instance = None


def get_esrgan() -> RealESRGANUpscaler:
    """Get or create Real-ESRGAN instance."""
    global _esrgan_instance
    if _esrgan_instance is None:
        _esrgan_instance = RealESRGANUpscaler()
    return _esrgan_instance


def upscale_image(image: Image.Image, method: str = "esrgan", scale: float = 4.0) -> Image.Image:
    """
    Upscale image using specified method.

    Args:
        image: PIL Image
        method: 'esrgan' or 'lanczos'
        scale: Scale factor

    Returns:
        Upscaled PIL Image
    """
    if method == "esrgan":
        return get_esrgan().upscale(image, scale)
    else:
        return LanczosUpscaler().upscale(image, scale)
