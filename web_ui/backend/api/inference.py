"""
Inference REST API endpoints.

Provides endpoints for:
- Loading/unloading models
- Generating images
- Managing gallery
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path

from web_ui.backend.models import CommandResponse
from web_ui.backend.services.inference_service import (
    get_inference_service,
    GenerationRequest,
)

router = APIRouter()


# Request/Response Models

class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_path: str = Field(..., description="Path to base model or checkpoint")
    model_type: str = Field(..., description="Model type (e.g., FLUX_DEV_1, SDXL_1_0)")
    lora_paths: Optional[List[str]] = Field(default=None, description="Optional LoRA paths to apply")


class GenerateRequest(BaseModel):
    """Request to generate images."""
    model_config = {"populate_by_name": True}
    
    prompt: str = Field(..., description="Text prompt for generation")
    negative_prompt: str = Field(default="", description="Negative prompt")
    width: int = Field(default=1024, ge=256, le=4096, description="Image width")
    height: int = Field(default=1024, ge=256, le=4096, description="Image height")
    steps: int = Field(default=20, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=30.0, alias="cfg_scale", description="CFG scale")
    seed: int = Field(default=-1, description="Seed (-1 for random)")
    batch_size: int = Field(default=1, ge=1, le=4, description="Batch size")
    # Model selection (for auto-load on generate)
    model_path: Optional[str] = Field(default=None, description="Path to model for auto-load")
    model_type: Optional[str] = Field(default=None, description="Model type for auto-load")
    lora_paths: Optional[List[str]] = Field(default=None, description="LoRA paths to apply")
    # Generation mode
    mode: str = Field(default="txt2img", description="Generation mode: txt2img, img2img, inpainting, edit, video")
    # img2img / inpainting inputs
    init_image_path: str = Field(default="", description="Path to init image for img2img/inpainting")
    mask_image_path: str = Field(default="", description="Path to mask image for inpainting")
    strength: float = Field(default=0.75, ge=0.0, le=1.0, description="Denoising strength for img2img")
    # Video generation
    num_frames: int = Field(default=16, ge=1, le=128, description="Number of frames for video")
    fps: int = Field(default=8, ge=1, le=60, description="Frames per second for video")
    # Edit mode
    edit_instruction: str = Field(default="", description="Edit instruction for edit models")
    # Multi-image input
    reference_images: Optional[List[str]] = Field(default=None, description="Reference image paths for FLUX 2")


class InferenceStateResponse(BaseModel):
    """Current inference state."""
    model_loaded: bool
    model_path: Optional[str]
    model_type: Optional[str]
    lora_paths: List[str]
    is_generating: bool
    generation_progress: int


class GeneratedImageResponse(BaseModel):
    """Generated image info."""
    id: str
    path: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    created_at: str


class GalleryResponse(BaseModel):
    """Gallery of generated images."""
    images: List[GeneratedImageResponse]
    count: int


# Endpoints

@router.get(
    "/status",
    response_model=InferenceStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Get inference status",
    description="Get current inference state including loaded model and generation progress.",
)
async def get_status() -> InferenceStateResponse:
    """Get current inference state."""
    service = get_inference_service()
    state = service.get_state()
    return InferenceStateResponse(**state)


@router.post(
    "/load",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Load model",
    description="Load a base model with optional LoRA adapters.",
)
async def load_model(request: LoadModelRequest) -> CommandResponse:
    """Load a model for inference."""
    service = get_inference_service()
    result = service.load_model(
        model_path=request.model_path,
        model_type=request.model_type,
        lora_paths=request.lora_paths,
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to load model")
        )
    
    return CommandResponse(
        success=True,
        message=result.get("message", "Model loaded")
    )


@router.post(
    "/unload",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Unload model",
    description="Unload the current model and free memory.",
)
async def unload_model() -> CommandResponse:
    """Unload the current model."""
    service = get_inference_service()
    result = service.unload_model()
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to unload model")
        )
    
    return CommandResponse(
        success=True,
        message=result.get("message", "Model unloaded")
    )


@router.post(
    "/generate",
    status_code=status.HTTP_200_OK,
    summary="Generate image",
    description="Generate an image with the loaded model.",
)
async def generate_image(request: GenerateRequest):
    """Generate an image."""
    service = get_inference_service()

    # Debug logging
    print(f"[DEBUG] Generate request - model_path: {request.model_path}, model_type: {request.model_type}")

    # Auto-load model if not loaded and model_path/model_type provided
    # Auto-load model if not loaded OR if loaded model is different
    state = service.get_state()
    print(f"[DEBUG] Current state: model_loaded={state['model_loaded']}, model_path={state['model_path']}")
    should_load = False
    
    if request.model_path and request.model_type:
        if not state['model_loaded']:
            should_load = True
        elif state['model_path'] != request.model_path or state['model_type'] != request.model_type:
            should_load = True
            
    if should_load:
        load_result = service.load_model(
            model_path=request.model_path,
            model_type=request.model_type,
            lora_paths=request.lora_paths,
        )
        if not load_result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load model: {load_result.get('error', 'Unknown error')}"
            )

    gen_request = GenerationRequest(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        batch_size=request.batch_size,
        mode=request.mode,
        init_image_path=request.init_image_path,
        mask_image_path=request.mask_image_path,
        strength=request.strength,
        num_frames=request.num_frames,
        fps=request.fps,
        edit_instruction=request.edit_instruction,
        reference_images=request.reference_images,
    )

    result = service.generate(gen_request)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Generation failed")
        )

    return {
        "success": True,
        "image": result.get("image"),
    }


@router.post(
    "/cancel",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel generation",
    description="Cancel the current image generation.",
)
async def cancel_generation() -> CommandResponse:
    """Cancel current generation."""
    service = get_inference_service()
    result = service.cancel_generation()
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to cancel")
        )
    
    return CommandResponse(
        success=True,
        message=result.get("message", "Cancelled")
    )


@router.get(
    "/gallery",
    response_model=GalleryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get gallery",
    description="Get list of generated images.",
)
async def get_gallery(limit: int = 50) -> GalleryResponse:
    """Get generated images gallery."""
    service = get_inference_service()
    images = service.get_gallery(limit)
    
    return GalleryResponse(
        images=[GeneratedImageResponse(**img) for img in images],
        count=len(images)
    )


@router.get(
    "/gallery/{image_id}",
    status_code=status.HTTP_200_OK,
    summary="Get image",
    description="Get a specific generated image file.",
)
async def get_image(image_id: str):
    """Get a generated image file."""
    service = get_inference_service()
    images = service.get_gallery(1000)
    
    for img in images:
        if img["id"] == image_id:
            path = Path(img["path"])
            if path.exists():
                return FileResponse(path, media_type="image/png")
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image file not found"
                )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Image '{image_id}' not found"
    )


@router.delete(
    "/gallery/{image_id}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete image",
    description="Delete a generated image from gallery.",
)
async def delete_image(image_id: str) -> CommandResponse:
    """Delete an image from gallery."""
    service = get_inference_service()
    
    if not service.delete_image(image_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{image_id}' not found"
        )
    
    return CommandResponse(
        success=True,
        message=f"Image '{image_id}' deleted"
    )


@router.delete(
    "/gallery",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear gallery",
    description="Clear all generated images from gallery.",
)
async def clear_gallery() -> CommandResponse:
    """Clear the gallery."""
    service = get_inference_service()
    service.clear_gallery()
    
    return CommandResponse(
        success=True,
        message="Gallery cleared"
    )
