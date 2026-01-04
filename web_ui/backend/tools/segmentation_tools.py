"""
Segmentation Tools - SAM2
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

MODELS_DIR = Path(__file__).parent / "models"


class SAM2Segmenter:
    """Segment Anything Model 2 wrapper."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path or str(MODELS_DIR / "sam2_hiera_large.pt")
        self.predictor = None

    def load(self):
        """Load SAM2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # SAM2 config
            model_cfg = "sam2_hiera_l.yaml"

            sam2_model = build_sam2(model_cfg, self.model_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            return False

    def segment_point(self, image: Image.Image, points: List[Tuple[int, int]],
                      labels: List[int] = None) -> np.ndarray:
        """
        Segment image based on point prompts.

        Args:
            image: PIL Image
            points: List of (x, y) coordinates
            labels: List of labels (1 for foreground, 0 for background)

        Returns:
            Binary mask as numpy array
        """
        if self.predictor is None:
            self.load()

        # Convert to numpy
        image_np = np.array(image.convert("RGB"))

        # Set image
        self.predictor.set_image(image_np)

        # Prepare points
        point_coords = np.array(points)
        point_labels = np.array(labels) if labels else np.ones(len(points))

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def segment_box(self, image: Image.Image, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Segment image based on bounding box.

        Args:
            image: PIL Image
            box: (x1, y1, x2, y2) bounding box

        Returns:
            Binary mask as numpy array
        """
        if self.predictor is None:
            self.load()

        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)

        masks, scores, logits = self.predictor.predict(
            box=np.array(box),
            multimask_output=True
        )

        best_idx = np.argmax(scores)
        return masks[best_idx]

    def auto_segment(self, image: Image.Image) -> List[np.ndarray]:
        """
        Automatically segment all objects in image.

        Returns:
            List of binary masks
        """
        if self.predictor is None:
            self.load()

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            image_np = np.array(image.convert("RGB"))

            mask_generator = SAM2AutomaticMaskGenerator(self.predictor.model)
            masks = mask_generator.generate(image_np)

            return [m['segmentation'] for m in masks]
        except Exception as e:
            print(f"Auto-segment failed: {e}")
            return []


# Singleton instance
_sam2_instance = None

def get_sam2() -> SAM2Segmenter:
    """Get or create SAM2 instance."""
    global _sam2_instance
    if _sam2_instance is None:
        _sam2_instance = SAM2Segmenter()
    return _sam2_instance
