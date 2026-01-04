import os
import fastapi
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from web_ui.backend.services.caption_service import get_caption_service

router = APIRouter()

# Data Models
class LoadModelRequest(BaseModel):
    model_id: str
    quantization: str = "8-bit" # "None", "8-bit", "4-bit"
    attn_impl: str = "flash_attention_2" # "flash_attention_2", "eager"

class CaptionRequest(BaseModel):
    media_path: str
    prompt: str
    max_tokens: int = 128
    resolution_mode: str = "auto"

class BatchProcessRequest(BaseModel):
    folder_path: str
    prompt: str
    skip_existing: bool = False
    max_tokens: int = 128
    resolution_mode: str = "auto"
    batch_size: int = 1 # Not fully used in generator logic but kept for interface compatibility

# Global state for batch job
current_batch_generator = None
batch_status = {"active": False, "stats": {}, "current_file": None}

@router.get("/state")
async def get_state():
    service = get_caption_service()
    return service.get_state()

@router.post("/load")
async def load_model(req: LoadModelRequest):
    service = get_caption_service()
    try:
        # Run in threadpool to avoid blocking main loop
        result = await asyncio.to_thread(
            service.load_model, 
            req.model_id, 
            req.quantization, 
            req.attn_impl
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unload")
async def unload_model():
    service = get_caption_service()
    service.unload_model()
    return {"status": "unloaded"}

@router.post("/generate")
async def generate_caption(req: CaptionRequest):
    service = get_caption_service()
    try:
        caption = await asyncio.to_thread(
            service.generate_caption,
            req.media_path,
            req.prompt,
            req.max_tokens,
            req.resolution_mode
        )
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/start")
async def start_batch(req: BatchProcessRequest, background_tasks: BackgroundTasks):
    global current_batch_generator, batch_status
    
    if batch_status["active"]:
        raise HTTPException(status_code=400, detail="Batch job already running")

    service = get_caption_service()
    
    # Check model is loaded first
    if not service.model or not service.processor:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")    
    # Initialize generator
    current_batch_generator = service.process_folder(
        req.folder_path,
        req.prompt,
        req.skip_existing,
        req.max_tokens,
        req.resolution_mode
    )
    
    batch_status["active"] = True
    batch_status["stats"] = {"processed": 0, "skipped": 0, "failed": 0}
    batch_status["progress"] = 0
    batch_status["current_file"] = None
    batch_status["last_caption"] = None
    
    # Background task to consume generator (simple loop for now)
    # In a real app with websockets, we'd emit events. 
    # Here we just iterate to keep it running.
    background_tasks.add_task(run_batch_job)
    
    return {"status": "started"}


@router.get("/batch/status")
async def get_batch_status():
    """Returns current batch processing status for UI polling."""
    return {
        "active": batch_status.get("active", False),
        "stats": batch_status.get("stats", {"processed": 0, "skipped": 0, "failed": 0}),
        "current_file": batch_status.get("current_file"),
        "last_caption": batch_status.get("last_caption"),
        "progress": batch_status.get("progress", 0)
    }


@router.post("/batch/stop")
async def stop_batch():
    """Stops the current batch job."""
    global batch_status
    service = get_caption_service()
    service.stop_processing()
    batch_status["active"] = False
    return {"status": "stopped"}

@router.get("/preview")
async def get_image_preview(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
        
    # Resize for thumbnail to save bandwidth
    try:
        from PIL import Image
        import io
        
        # Determine format
        ext = os.path.splitext(path)[-1].lower()
        if ext in ['.mp4', '.mov', '.avi', '.webm']:
             # Video thumbnail not implemented yet, return placeholder or error
             # For now, just return 404 or specific icon, asking user to trust
             raise HTTPException(status_code=400, detail="Video preview not supported yet")

        with Image.open(path) as img:
            img.thumbnail((300, 300))
            buf = io.BytesIO()
            fmt = "JPEG" if ext in ['.jpg', '.jpeg'] else "PNG"
            img.save(buf, format=fmt)
            buf.seek(0)
            return fastapi.responses.StreamingResponse(buf, media_type=f"image/{fmt.lower()}")
    except Exception as e:
        print(f"Preview error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load image")

def _run_batch_sync():
    """Synchronous batch processing that runs in a thread."""
    global current_batch_generator, batch_status
    if not current_batch_generator:
        return
        
    try:
        for update in current_batch_generator:
            # Update global status so polling client can see
            if update["type"] in ["success", "skipped", "error_file"]:
                batch_status["stats"] = update["stats"]
                batch_status["current_file"] = update.get("filename")
                batch_status["last_caption"] = update.get("caption") if update.get("caption") else None
            elif update["type"] == "complete":
                batch_status["active"] = False
                batch_status["current_file"] = None
            elif update["type"] == "aborted":
                batch_status["active"] = False
            
            if "progress" in update:
                batch_status["progress"] = update["progress"]
                
    except Exception as e:
        print(f"Batch job error: {e}")
        import traceback
        traceback.print_exc()
        batch_status["active"] = False


async def run_batch_job():
    """Async wrapper that runs the batch job in a thread pool."""
    await asyncio.to_thread(_run_batch_sync)
