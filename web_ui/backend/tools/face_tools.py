"""
Face Tools - GFPGAN, CodeFormer, InsightFace
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple

MODELS_DIR = Path(__file__).parent / "models"


class FaceRestorer:
    """Face restoration using GFPGAN or CodeFormer."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.gfpgan = None
        self.codeformer = None

    def load_gfpgan(self, model_path: str = None):
        """Load GFPGAN model."""
        try:
            from gfpgan import GFPGANer

            model_path = model_path or str(MODELS_DIR / "GFPGANv1.4.pth")

            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✅ GFPGAN loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load GFPGAN: {e}")
            return False

    def restore_gfpgan(self, image: Image.Image, upscale: int = 2) -> Image.Image:
        """
        Restore faces using GFPGAN.

        Args:
            image: PIL Image
            upscale: Upscale factor (1, 2, 4)

        Returns:
            Restored PIL Image
        """
        if self.gfpgan is None:
            self.load_gfpgan()

        # Convert to numpy BGR
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]

        # Restore
        _, _, output = self.gfpgan.enhance(
            img_np,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )

        # Convert back to PIL RGB
        output_rgb = output[:, :, ::-1]
        return Image.fromarray(output_rgb)

    def load_codeformer(self):
        """Load CodeFormer model."""
        try:
            import sys
            codeformer_path = Path(__file__).parent / "face" / "CodeFormer"
            if codeformer_path.exists():
                sys.path.insert(0, str(codeformer_path))

            # CodeFormer loading would go here
            # For now, use basicsr's implementation
            print("⚠️ CodeFormer requires additional setup")
            return False
        except Exception as e:
            print(f"❌ Failed to load CodeFormer: {e}")
            return False


class FaceDetector:
    """Face detection using InsightFace."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.app = None

    def load(self):
        """Load InsightFace model."""
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ InsightFace loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load InsightFace: {e}")
            return False

    def detect_faces(self, image: Image.Image) -> List[dict]:
        """
        Detect faces in image.

        Returns:
            List of face dicts with bbox, landmarks, embedding
        """
        if self.app is None:
            self.load()

        img_np = np.array(image.convert("RGB"))
        faces = self.app.get(img_np)

        results = []
        for face in faces:
            results.append({
                'bbox': face.bbox.tolist(),
                'landmarks': face.kps.tolist() if face.kps is not None else None,
                'embedding': face.embedding.tolist() if face.embedding is not None else None,
                'age': face.age if hasattr(face, 'age') else None,
                'gender': face.gender if hasattr(face, 'gender') else None,
            })

        return results

    def get_face_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get face embedding from image."""
        faces = self.detect_faces(image)
        if faces:
            return np.array(faces[0]['embedding'])
        return None


# Singleton instances
_face_restorer = None
_face_detector = None


def get_face_restorer() -> FaceRestorer:
    """Get or create FaceRestorer instance."""
    global _face_restorer
    if _face_restorer is None:
        _face_restorer = FaceRestorer()
    return _face_restorer


def get_face_detector() -> FaceDetector:
    """Get or create FaceDetector instance."""
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector()
    return _face_detector
