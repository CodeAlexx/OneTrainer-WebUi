"""
OneTrainer Inference App - Tools
================================

A collection of image processing tools for the inference app.

Categories:
- Segmentation: SAM2
- Face: GFPGAN, CodeFormer, InsightFace
- Upscaling: Real-ESRGAN
- Background: REMBG
"""

from pathlib import Path

TOOLS_DIR = Path(__file__).parent
MODELS_DIR = TOOLS_DIR / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)
