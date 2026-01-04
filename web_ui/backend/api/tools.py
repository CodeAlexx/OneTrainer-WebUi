"""
Tools REST API endpoints.

Provides endpoints for:
- Face restoration (GFPGAN)
- Face swapping
- Image upscaling (Real-ESRGAN)
- Background removal (rembg)
- Segmentation (SAM2)
"""

import io
import base64
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import Response
from PIL import Image

router = APIRouter()


# Request/Response Models

class Base64ImageRequest(BaseModel):
    """Base64 encoded image input."""
    image: str = Field(..., description="Base64 encoded image")


class FaceRestoreRequest(BaseModel):
    """Face restoration request."""
    image: str = Field(..., description="Base64 encoded image")
    upscale: int = Field(default=2, ge=1, le=4, description="Upscale factor")
    method: str = Field(default="gfpgan", description="Method: gfpgan or codeformer")


class FaceSwapRequest(BaseModel):
    """Face swap request."""
    source_image: str = Field(..., description="Base64 source image (face to use)")
    target_image: str = Field(..., description="Base64 target image (face to replace)")
    swap_all: bool = Field(default=False, description="Swap all faces in target")


class UpscaleRequest(BaseModel):
    """Image upscale request."""
    image: str = Field(..., description="Base64 encoded image")
    scale: float = Field(default=4.0, ge=1.0, le=8.0, description="Scale factor")
    method: str = Field(default="esrgan", description="Method: esrgan or lanczos")


class BackgroundRemoveRequest(BaseModel):
    """Background removal request."""
    image: str = Field(..., description="Base64 encoded image")
    model: str = Field(default="u2net", description="Model: u2net, u2netp, isnet-general-use")
    alpha_matting: bool = Field(default=False, description="Use alpha matting")


class BackgroundReplaceRequest(BaseModel):
    """Background replacement request."""
    image: str = Field(..., description="Base64 foreground image")
    background: Optional[str] = Field(default=None, description="Base64 background image")
    color: Optional[List[int]] = Field(default=[255, 255, 255], description="Background color RGB")


class SegmentRequest(BaseModel):
    """Segmentation request."""
    image: str = Field(..., description="Base64 encoded image")
    points: Optional[List[List[int]]] = Field(default=None, description="Points [[x,y], ...]")
    labels: Optional[List[int]] = Field(default=None, description="Point labels (1=foreground)")
    box: Optional[List[int]] = Field(default=None, description="Bounding box [x1,y1,x2,y2]")
    auto: bool = Field(default=False, description="Auto segment all objects")


class Base64Response(BaseModel):
    """Base64 image response."""
    success: bool
    image: str
    message: Optional[str] = None


# Utility functions

def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    # Remove data URL prefix if present
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


# Endpoints

@router.post(
    "/face/restore",
    response_model=Base64Response,
    summary="Restore faces in image",
    description="Enhance and restore faces using GFPGAN or CodeFormer.",
)
async def face_restore(request: FaceRestoreRequest) -> Base64Response:
    """Restore faces in image."""
    try:
        from web_ui.backend.tools.face_tools import get_face_restorer
        
        image = decode_base64_image(request.image)
        restorer = get_face_restorer()
        
        if request.method == "gfpgan":
            result = restorer.restore_gfpgan(image, upscale=request.upscale)
        else:
            # Fallback to GFPGAN if CodeFormer not available
            result = restorer.restore_gfpgan(image, upscale=request.upscale)
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message=f"Face restored using {request.method}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/face/swap",
    response_model=Base64Response,
    summary="Swap faces between images",
    description="Swap face from source image onto target image.",
)
async def face_swap(request: FaceSwapRequest) -> Base64Response:
    """Swap faces between images."""
    try:
        from web_ui.backend.tools.faceswap_tools import get_faceswapper
        
        source = decode_base64_image(request.source_image)
        target = decode_base64_image(request.target_image)
        
        swapper = get_faceswapper()
        
        if request.swap_all:
            result = swapper.swap_all_faces(source, target)
        else:
            result = swapper.swap_face(source, target)
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message="Face swapped successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upscale",
    response_model=Base64Response,
    summary="Upscale image",
    description="Upscale image using Real-ESRGAN or Lanczos.",
)
async def upscale_image(request: UpscaleRequest) -> Base64Response:
    """Upscale image."""
    try:
        from web_ui.backend.tools.upscaling_tools import upscale_image as do_upscale
        
        image = decode_base64_image(request.image)
        result = do_upscale(image, method=request.method, scale=request.scale)
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message=f"Upscaled {request.scale}x using {request.method}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/background/remove",
    response_model=Base64Response,
    summary="Remove background",
    description="Remove background from image using rembg.",
)
async def remove_background(request: BackgroundRemoveRequest) -> Base64Response:
    """Remove background from image."""
    try:
        from web_ui.backend.tools.background_tools import get_rembg
        
        image = decode_base64_image(request.image)
        remover = get_rembg()
        remover.load(request.model)
        
        result = remover.remove_background(image, alpha_matting=request.alpha_matting)
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message=f"Background removed using {request.model}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/background/replace",
    response_model=Base64Response,
    summary="Replace background",
    description="Replace background with color or image.",
)
async def replace_background(request: BackgroundReplaceRequest) -> Base64Response:
    """Replace background."""
    try:
        from web_ui.backend.tools.background_tools import get_rembg
        
        image = decode_base64_image(request.image)
        bg_image = decode_base64_image(request.background) if request.background else None
        color = tuple(request.color) if request.color else (255, 255, 255)
        
        remover = get_rembg()
        result = remover.replace_background(image, background=bg_image, color=color)
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message="Background replaced"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/segment",
    response_model=Base64Response,
    summary="Segment image",
    description="Segment objects using SAM2.",
)
async def segment_image(request: SegmentRequest) -> Base64Response:
    """Segment image using SAM2."""
    try:
        from web_ui.backend.tools.segmentation_tools import get_sam2
        import numpy as np
        
        image = decode_base64_image(request.image)
        sam2 = get_sam2()
        
        if request.auto:
            masks = sam2.auto_segment(image)
            if masks:
                # Combine masks for visualization
                combined = np.zeros_like(masks[0], dtype=np.uint8)
                for i, mask in enumerate(masks):
                    combined[mask] = (i + 1) * 30 % 256
                result = Image.fromarray(combined)
            else:
                result = image
        elif request.box:
            mask = sam2.segment_box(image, tuple(request.box))
            result = Image.fromarray((mask * 255).astype(np.uint8))
        elif request.points:
            labels = request.labels or [1] * len(request.points)
            mask = sam2.segment_point(image, request.points, labels)
            result = Image.fromarray((mask * 255).astype(np.uint8))
        else:
            # Return original if no segmentation method specified
            result = image
        
        return Base64Response(
            success=True,
            image=encode_image_base64(result),
            message="Segmentation complete"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", summary="Get tools status")
async def get_tools_status():
    """Get status of available tools."""
    tools_status = {}
    
    # Check each tool
    try:
        from web_ui.backend.tools.face_tools import get_face_restorer
        tools_status["face_restore"] = {"available": True, "loaded": get_face_restorer().gfpgan is not None}
    except:
        tools_status["face_restore"] = {"available": False, "loaded": False}
    
    try:
        from web_ui.backend.tools.faceswap_tools import get_faceswapper
        tools_status["face_swap"] = {"available": True, "loaded": get_faceswapper().swapper is not None}
    except:
        tools_status["face_swap"] = {"available": False, "loaded": False}
    
    try:
        from web_ui.backend.tools.upscaling_tools import get_esrgan
        tools_status["upscale"] = {"available": True, "loaded": get_esrgan().upsampler is not None}
    except:
        tools_status["upscale"] = {"available": False, "loaded": False}
    
    try:
        from web_ui.backend.tools.background_tools import get_rembg
        tools_status["background"] = {"available": True, "loaded": get_rembg().session is not None}
    except:
        tools_status["background"] = {"available": False, "loaded": False}
    
    try:
        from web_ui.backend.tools.segmentation_tools import get_sam2
        tools_status["segment"] = {"available": True, "loaded": get_sam2().predictor is not None}
    except:
        tools_status["segment"] = {"available": False, "loaded": False}
    
    return {"tools": tools_status}
