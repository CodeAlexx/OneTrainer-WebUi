"""
ControlNet Preprocessor Plugin

Provides depth, canny, and pose estimation preprocessors for ControlNet conditioning.
"""

from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

from ..plugin_base import (
    PreprocessorPlugin,
    PluginInfo,
    PluginType,
    PluginUIElement,
)


class ControlNetPreprocessor(PreprocessorPlugin):
    """
    ControlNet preprocessor plugin.

    Supports multiple preprocessor types for ControlNet conditioning.
    """

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._preprocessor = None
        self._current_type = None

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="ControlNet Preprocessor",
            version="1.0.0",
            description="Preprocessors for ControlNet conditioning (depth, canny, pose, etc.)",
            author="OneTrainer",
            plugin_type=PluginType.PREPROCESSOR,
            supported_models=["STABLE_DIFFUSION_15", "STABLE_DIFFUSION_XL_10_BASE", "FLUX_DEV_1"],
            requires_gpu=True,
            dependencies=["controlnet-aux"],
        )

    @property
    def ui_elements(self) -> List[PluginUIElement]:
        return [
            PluginUIElement(
                id="preprocessor_type",
                label="Preprocessor Type",
                type="dropdown",
                default="canny",
                options={
                    "choices": ["canny", "depth_midas", "depth_zoe", "openpose", "lineart", "hed", "scribble"],
                },
            ),
            PluginUIElement(
                id="control_strength",
                label="Control Strength",
                type="slider",
                default=1.0,
                options={"min": 0.0, "max": 2.0, "step": 0.05},
            ),
            PluginUIElement(
                id="control_start",
                label="Control Start",
                type="slider",
                default=0.0,
                options={"min": 0.0, "max": 1.0, "step": 0.05},
            ),
            PluginUIElement(
                id="control_end",
                label="Control End",
                type="slider",
                default=1.0,
                options={"min": 0.0, "max": 1.0, "step": 0.05},
            ),
            PluginUIElement(
                id="control_image",
                label="Control Image",
                type="image",
                default=None,
            ),
        ]

    def load(self) -> bool:
        """Load the preprocessor models."""
        try:
            # Lazy load controlnet_aux when needed
            self._loaded = True
            return True
        except ImportError:
            print("ControlNet preprocessors require 'controlnet-aux' package")
            return False

    def _get_preprocessor(self, preprocessor_type: str):
        """Get or create preprocessor instance."""
        if self._current_type == preprocessor_type and self._preprocessor is not None:
            return self._preprocessor

        try:
            from controlnet_aux import (
                CannyDetector,
                MidasDetector,
                ZoeDetector,
                OpenposeDetector,
                LineartDetector,
                HEDdetector,
            )

            preprocessor_map = {
                "canny": CannyDetector,
                "depth_midas": MidasDetector,
                "depth_zoe": ZoeDetector,
                "openpose": OpenposeDetector,
                "lineart": LineartDetector,
                "hed": HEDdetector,
            }

            if preprocessor_type in preprocessor_map:
                self._preprocessor = preprocessor_map[preprocessor_type]()
                self._current_type = preprocessor_type
                return self._preprocessor

        except ImportError:
            pass

        return None

    def preprocess(
        self,
        image: Image.Image,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preprocess an image for ControlNet conditioning.

        Args:
            image: Input image
            settings: Plugin settings from UI

        Returns:
            Dict containing conditioning data
        """
        preprocessor_type = settings.get("preprocessor_type", "canny")
        control_strength = settings.get("control_strength", 1.0)
        control_start = settings.get("control_start", 0.0)
        control_end = settings.get("control_end", 1.0)

        # Get the preprocessor
        preprocessor = self._get_preprocessor(preprocessor_type)

        if preprocessor is None:
            # Fallback: just return the image as-is
            return {
                "control_image": image,
                "controlnet_conditioning_scale": control_strength,
                "control_guidance_start": control_start,
                "control_guidance_end": control_end,
            }

        # Run preprocessing
        if preprocessor_type == "canny":
            processed = preprocessor(image, low_threshold=100, high_threshold=200)
        elif preprocessor_type == "scribble":
            # Simple edge detection fallback
            import cv2
            import numpy as np
            img_array = np.array(image.convert("L"))
            edges = cv2.Canny(img_array, 50, 150)
            processed = Image.fromarray(edges)
        else:
            processed = preprocessor(image)

        return {
            "control_image": processed,
            "controlnet_conditioning_scale": control_strength,
            "control_guidance_start": control_start,
            "control_guidance_end": control_end,
        }

    def get_conditioning_kwargs(
        self,
        preprocessed: Dict[str, Any],
        model_type: str,
    ) -> Dict[str, Any]:
        """Convert to model-specific kwargs."""
        # For diffusers ControlNet pipeline
        return {
            "image": preprocessed.get("control_image"),
            "controlnet_conditioning_scale": preprocessed.get("controlnet_conditioning_scale", 1.0),
            "control_guidance_start": preprocessed.get("control_guidance_start", 0.0),
            "control_guidance_end": preprocessed.get("control_guidance_end", 1.0),
        }
