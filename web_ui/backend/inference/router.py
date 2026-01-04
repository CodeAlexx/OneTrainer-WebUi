"""
Inference API Router

Exposes inference endpoints at /api/inference/* for the new InferenceView.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import json

router = APIRouter()

# Lazy-load the engine to avoid startup time
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from .engine import InferenceEngine
        _engine = InferenceEngine()
    return _engine


@router.get("/status")
async def get_status():
    """Get current inference status."""
    engine = get_engine()
    
    # Get GPU info
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        gpu_memory_used = torch.cuda.memory_allocated(0) // (1024 * 1024)
        gpu_memory_free = gpu_memory_total - gpu_memory_used
    else:
        gpu_name = "CPU"
        gpu_memory_total = 0
        gpu_memory_used = 0
        gpu_memory_free = 0
    
    return {
        "gpu_name": gpu_name,
        "gpu_memory_total": gpu_memory_total,
        "gpu_memory_used": gpu_memory_used,
        "gpu_memory_free": gpu_memory_free,
        "model_info": {
            "loaded": engine.pipeline is not None,
            "model_path": engine.model_path,
            "model_type": engine.model_type.value if engine.model_type else None,
            "precision": engine.precision,
            "loras": [{"path": l.path, "weight": l.weight, "enabled": l.enabled} for l in engine.loras],
        },
        "is_generating": engine.is_generating,
        "progress": engine.progress,
        "current_step": engine.current_step,
        "total_steps": engine.total_steps,
    }


@router.get("/gallery")
async def get_gallery(limit: int = 50):
    """Get generated images gallery."""
    engine = get_engine()
    images = engine.gallery[:limit] if hasattr(engine, 'gallery') else []
    return {"images": [img.model_dump() if hasattr(img, 'model_dump') else img for img in images]}


@router.get("/gallery/{image_id}")
async def get_gallery_image(image_id: str):
    """Get a specific gallery image."""
    engine = get_engine()
    for img in engine.gallery:
        if (hasattr(img, 'id') and img.id == image_id) or (isinstance(img, dict) and img.get('id') == image_id):
            path = img.path if hasattr(img, 'path') else img.get('path')
            if path:
                return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.post("/generate")
async def generate(request: dict):
    """Generate images."""
    engine = get_engine()
    
    # Import the request model
    from .engine import GenerateRequest, GenerationMode
    
    try:
        # Convert mode string to enum if needed
        if 'mode' in request and isinstance(request['mode'], str):
            request['mode'] = GenerationMode(request['mode'])
        
        gen_request = GenerateRequest(**request)
        result = await engine.generate_async(gen_request)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/cancel")
async def cancel_generation():
    """Cancel current generation."""
    engine = get_engine()
    engine.should_cancel = True
    return {"success": True, "message": "Cancellation requested"}


@router.post("/model/load")
async def load_model(request: dict):
    """Load a model."""
    engine = get_engine()
    from .engine import LoadModelRequest, ModelType
    
    try:
        # Convert model_type string to enum
        if 'model_type' in request and isinstance(request['model_type'], str):
            request['model_type'] = ModelType(request['model_type'])
        
        load_request = LoadModelRequest(**request)
        result = engine.load_model(load_request)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/unload")
async def unload_model():
    """Unload the current model."""
    engine = get_engine()
    if hasattr(engine, 'unload_model'):
        result = engine.unload_model()
        return result
    else:
        engine.pipeline = None
        engine.model_path = None
        engine.model_type = None
        return {"success": True, "message": "Model unloaded"}
