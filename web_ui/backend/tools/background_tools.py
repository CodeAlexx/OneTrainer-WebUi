"""
Background Tools - REMBG
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple


class BackgroundRemover:
    """Background removal using rembg."""

    def __init__(self):
        self.session = None

    def load(self, model_name: str = "u2net"):
        """
        Load rembg model.

        Available models:
        - u2net (default, general purpose)
        - u2netp (lightweight)
        - u2net_human_seg (for humans)
        - silueta (fast)
        - isnet-general-use
        - isnet-anime
        """
        try:
            from rembg import new_session
            self.session = new_session(model_name)
            print(f"✅ REMBG loaded ({model_name})")
            return True
        except Exception as e:
            print(f"❌ Failed to load REMBG: {e}")
            return False

    def remove_background(self, image: Image.Image,
                          alpha_matting: bool = False,
                          alpha_matting_foreground_threshold: int = 240,
                          alpha_matting_background_threshold: int = 10) -> Image.Image:
        """
        Remove background from image.

        Args:
            image: PIL Image
            alpha_matting: Use alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold (0-255)
            alpha_matting_background_threshold: Background threshold (0-255)

        Returns:
            PIL Image with transparent background (RGBA)
        """
        try:
            from rembg import remove

            if self.session is None:
                self.load()

            result = remove(
                image,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold
            )
            return result
        except Exception as e:
            print(f"Background removal failed: {e}")
            return image

    def get_mask(self, image: Image.Image) -> Image.Image:
        """
        Get background mask from image.

        Returns:
            Grayscale mask (white = foreground, black = background)
        """
        result = self.remove_background(image)
        if result.mode == 'RGBA':
            return result.split()[3]  # Alpha channel
        return result.convert('L')

    def replace_background(self, image: Image.Image,
                           background: Image.Image = None,
                           color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Replace background with solid color or another image.

        Args:
            image: PIL Image (foreground)
            background: Optional background image
            color: Background color if no image provided

        Returns:
            PIL Image with replaced background
        """
        # Remove background
        fg = self.remove_background(image)

        if fg.mode != 'RGBA':
            return fg

        # Create background
        if background is not None:
            bg = background.resize(fg.size).convert('RGBA')
        else:
            bg = Image.new('RGBA', fg.size, (*color, 255))

        # Composite
        return Image.alpha_composite(bg, fg).convert('RGB')


# Singleton instance
_rembg_instance = None


def get_rembg() -> BackgroundRemover:
    """Get or create BackgroundRemover instance."""
    global _rembg_instance
    if _rembg_instance is None:
        _rembg_instance = BackgroundRemover()
    return _rembg_instance


def remove_background(image: Image.Image, **kwargs) -> Image.Image:
    """Convenience function to remove background."""
    return get_rembg().remove_background(image, **kwargs)
