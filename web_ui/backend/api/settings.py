"""Settings API endpoints for model configurations."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json

router = APIRouter(prefix="/settings", tags=["settings"])

MODELS_CONFIG_PATH = Path.home() / ".cache" / "onetrainer" / "models_config.json"


class ModelEntry(BaseModel):
    id: str
    name: str
    path: str
    type: str  # 'safetensors' or 'diffusers'
    category: str  # 'Image' or 'Video'
    modelType: str


class ModelsConfig(BaseModel):
    models: List[ModelEntry]


# Default models configuration
DEFAULT_MODELS = [
    # Image models
    {"id": "1", "name": "FLUX Dev", "path": "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors", "type": "safetensors", "category": "Image", "modelType": "FLUX_DEV_1"},
    {"id": "2", "name": "FLUX Schnell", "path": "/home/alex/SwarmUI/Models/diffusion_models/uncensoredFemalesFLUX4step_nf4Schnell4step.safetensors", "type": "safetensors", "category": "Image", "modelType": "FLUX_SCHNELL"},
    {"id": "3", "name": "SDXL", "path": "/home/alex/SwarmUI/Models/diffusion_models/lustifySDXLNSFW_ggwpV7.safetensors", "type": "safetensors", "category": "Image", "modelType": "STABLE_DIFFUSION_XL_10_BASE"},
    {"id": "4", "name": "SD 3.5", "path": "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors", "type": "safetensors", "category": "Image", "modelType": "STABLE_DIFFUSION_35"},
    {"id": "5", "name": "Z-Image", "path": "/home/alex/SwarmUI/Models/diffusion_models/z_image_de_turbo_v1_bf16.safetensors", "type": "safetensors", "category": "Image", "modelType": "Z_IMAGE"},
    {"id": "6", "name": "Z-Image Turbo", "path": "/home/alex/SwarmUI/Models/diffusion_models/z_image_turbo_bf16.safetensors", "type": "safetensors", "category": "Image", "modelType": "Z_IMAGE_TURBO"},
    {"id": "7", "name": "Qwen Image", "path": "/home/alex/SwarmUI/Models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "type": "safetensors", "category": "Image", "modelType": "QWEN_IMAGE"},
    {"id": "8", "name": "Qwen Edit", "path": "alibaba-pai/OmniGen2-Edit", "type": "diffusers", "category": "Image", "modelType": "QWEN_IMAGE_EDIT"},
    {"id": "9", "name": "Lumina 2", "path": "Alpha-VLLM/Lumina-Image-2.0", "type": "diffusers", "category": "Image", "modelType": "lumina_2"},
    {"id": "10", "name": "OmniGen 2", "path": "BAAI/OmniGen2", "type": "diffusers", "category": "Image", "modelType": "omnigen_2"},
    {"id": "11", "name": "Kandinsky 5 T2I", "path": "kandinskylab/Kandinsky-5.0-T2I-Lite", "type": "diffusers", "category": "Image", "modelType": "KANDINSKY_5"},
    {"id": "12", "name": "Chroma HD", "path": "lodestones/Chroma1-HD", "type": "diffusers", "category": "Image", "modelType": "CHROMA_1"},
    # Video models
    {"id": "13", "name": "Kandinsky 5 T2V Lite", "path": "/home/alex/SwarmUI/Models/diffusion_models/kandinsky5lite_t2v_sft_5s.safetensors", "type": "safetensors", "category": "Video", "modelType": "kandinsky_5_video"},
    {"id": "14", "name": "Kandinsky 5 T2V Pro", "path": "/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors", "type": "safetensors", "category": "Video", "modelType": "kandinsky_5_video_pro"},
    {"id": "15", "name": "Wan 2.2 T2V (High)", "path": "/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", "type": "safetensors", "category": "Video", "modelType": "wan_t2v_high"},
    {"id": "16", "name": "Wan 2.2 T2V (Low)", "path": "/home/alex/SwarmUI/Models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", "type": "safetensors", "category": "Video", "modelType": "wan_t2v_low"},
    {"id": "17", "name": "Wan 2.2 I2V (High)", "path": "/home/alex/SwarmUI/Models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors", "type": "safetensors", "category": "Video", "modelType": "wan_i2v_high"},
    {"id": "18", "name": "Wan 2.2 I2V (Low)", "path": "/home/alex/SwarmUI/Models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors", "type": "safetensors", "category": "Video", "modelType": "wan_i2v_low"},
    {"id": "19", "name": "Wan 2.1 VACE", "path": "/home/alex/SwarmUI/Models/diffusion_models/wan2.1_vace_14B_fp16.safetensors", "type": "safetensors", "category": "Video", "modelType": "wan_vace"},
]


@router.get("/models")
async def get_models():
    """Get configured models."""
    try:
        if MODELS_CONFIG_PATH.exists():
            with open(MODELS_CONFIG_PATH, "r") as f:
                data = json.load(f)
                return {"models": data.get("models", DEFAULT_MODELS)}
    except Exception as e:
        print(f"Error loading models config: {e}")
    
    return {"models": DEFAULT_MODELS}


@router.post("/models")
async def save_models(config: ModelsConfig):
    """Save model configurations."""
    try:
        MODELS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODELS_CONFIG_PATH, "w") as f:
            json.dump({"models": [m.dict() for m in config.models]}, f, indent=2)
        return {"success": True, "message": "Models saved"}
    except Exception as e:
        return {"success": False, "error": str(e)}
