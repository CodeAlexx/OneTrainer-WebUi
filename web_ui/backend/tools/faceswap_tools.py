"""
Face Swap Tools - Using InsightFace inswapper
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

MODELS_DIR = Path(__file__).parent / "models"


class FaceSwapper:
    """Face swapping using InsightFace inswapper."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.swapper = None
        self.face_analyser = None

    def load(self, model_path: str = None):
        """Load face swap model."""
        try:
            from insightface.app import FaceAnalysis
            import insightface

            # Load face analyser
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

            # Load swapper model
            model_path = model_path or str(MODELS_DIR / "inswapper_128.onnx")
            
            # Try to find model in common locations
            if not Path(model_path).exists():
                # Check insightface models dir
                import insightface
                insightface_root = Path(insightface.__file__).parent
                alt_path = insightface_root / "models" / "inswapper_128.onnx"
                if alt_path.exists():
                    model_path = str(alt_path)
                else:
                    # Try to download
                    print("⚠️ inswapper_128.onnx not found. Please download from:")
                    print("   https://huggingface.co/deepinsight/inswapper/tree/main")
                    return False

            self.swapper = insightface.model_zoo.get_model(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("✅ FaceSwapper loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load FaceSwapper: {e}")
            return False

    def get_faces(self, image: Image.Image):
        """Detect faces in image."""
        if self.face_analyser is None:
            self.load()
        
        img_np = np.array(image.convert("RGB"))
        faces = self.face_analyser.get(img_np)
        return faces, img_np

    def swap_face(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
    ) -> Image.Image:
        """
        Swap face from source to target image.

        Args:
            source_image: Image containing the face to use
            target_image: Image to swap face onto
            source_face_idx: Which face in source (if multiple)
            target_face_idx: Which face in target to replace

        Returns:
            PIL Image with face swapped
        """
        if self.swapper is None:
            self.load()

        # Get faces
        source_faces, source_np = self.get_faces(source_image)
        target_faces, target_np = self.get_faces(target_image)

        if not source_faces:
            print("No face found in source image")
            return target_image
        if not target_faces:
            print("No face found in target image")
            return target_image

        # Get specific faces
        source_face = source_faces[min(source_face_idx, len(source_faces) - 1)]
        target_face = target_faces[min(target_face_idx, len(target_faces) - 1)]

        # Swap
        result = self.swapper.get(target_np, target_face, source_face, paste_back=True)

        return Image.fromarray(result)

    def swap_all_faces(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
    ) -> Image.Image:
        """Swap all faces in target with source face."""
        if self.swapper is None:
            self.load()

        source_faces, source_np = self.get_faces(source_image)
        target_faces, target_np = self.get_faces(target_image)

        if not source_faces:
            return target_image

        source_face = source_faces[0]
        result = target_np.copy()

        for target_face in target_faces:
            result = self.swapper.get(result, target_face, source_face, paste_back=True)

        return Image.fromarray(result)


# Singleton instance
_faceswap_instance = None


def get_faceswapper() -> FaceSwapper:
    """Get or create FaceSwapper instance."""
    global _faceswap_instance
    if _faceswap_instance is None:
        _faceswap_instance = FaceSwapper()
    return _faceswap_instance


def swap_face(source: Image.Image, target: Image.Image, **kwargs) -> Image.Image:
    """Convenience function for face swap."""
    return get_faceswapper().swap_face(source, target, **kwargs)
