"""
Standalone Inference App - FastAPI Backend

Full-featured inference server supporting:
- Multiple model architectures (FLUX, SDXL, SD3, Z-Image, Qwen, Wan, etc.)
- txt2img, img2img, inpainting, edit, video modes
- LoRA/adapter support
- ControlNet preprocessors
- Real-time progress via WebSocket
"""

import asyncio
import base64
import io
import json
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

# Add OneTrainer to path for model loading
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Configuration & Types
# ============================================================================

class ModelType(str, Enum):
    # FLUX
    FLUX_DEV = "flux_dev"
    FLUX_SCHNELL = "flux_schnell"
    FLUX_2_DEV = "flux_2_dev"
    FLUX_FILL = "flux_fill"
    # Stable Diffusion
    SD_15 = "sd_15"
    SD_21 = "sd_21"
    SDXL = "sdxl"
    SDXL_TURBO = "sdxl_turbo"
    SD_3 = "sd_3"
    SD_35 = "sd_35"
    SD_35_TURBO = "sd_35_turbo"
    # Other
    PIXART_ALPHA = "pixart_alpha"
    PIXART_SIGMA = "pixart_sigma"
    SANA = "sana"
    CHROMA = "chroma"
    HIDREAM = "hidream"
    # Z-Image (Alibaba)
    Z_IMAGE = "z_image"
    Z_IMAGE_TURBO = "z_image_turbo"
    Z_IMAGE_EDIT = "z_image_edit"
    # Qwen
    QWEN_IMAGE = "qwen_image"
    QWEN_IMAGE_EDIT = "qwen_image_edit"
    # Lumina
    LUMINA = "lumina"
    LUMINA_2 = "lumina_2"
    # OmniGen
    OMNIGEN = "omnigen"
    OMNIGEN_2 = "omnigen_2"
    # Video - Wan 2.x
    WAN_T2V = "wan_t2v"
    WAN_I2V = "wan_i2v"
    WAN_VACE = "wan_vace"
    WAN_T2V_HIGH = "wan_t2v_high"  # Wan 2.2 high noise
    WAN_T2V_LOW = "wan_t2v_low"   # Wan 2.2 low noise
    WAN_I2V_HIGH = "wan_i2v_high"  # Wan 2.2 I2V high noise
    WAN_I2V_LOW = "wan_i2v_low"   # Wan 2.2 I2V low noise
    HUNYUAN_VIDEO = "hunyuan_video"
    # Kandinsky
    KANDINSKY_3 = "kandinsky_3"
    KANDINSKY_5 = "kandinsky_5"
    KANDINSKY_5_VIDEO = "kandinsky_5_video"


class GenerationMode(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    EDIT = "edit"
    VIDEO = "video"


class Sampler(str, Enum):
    EULER = "euler"
    EULER_A = "euler_a"
    DPM_2M = "dpm_2m"
    DPM_2M_KARRAS = "dpm_2m_karras"
    DPM_SDE = "dpm_sde"
    DPM_SDE_KARRAS = "dpm_sde_karras"
    DDIM = "ddim"
    UNIPC = "unipc"
    LCM = "lcm"
    FLOW_MATCH = "flow_match"
    HEUN = "heun"
    LMS = "lms"
    PNDM = "pndm"


ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "9:21": (9, 21),
    "3:2": (3, 2),
    "2:3": (2, 3),
}

RESOLUTION_PRESETS = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
    "1920x1080": (1920, 1080),
    "1080x1920": (1080, 1920),
    "1536x1536": (1536, 1536),
    "2048x2048": (2048, 2048),
}


# ============================================================================
# Request/Response Models
# ============================================================================

class LoadModelRequest(BaseModel):
    model_path: str
    model_type: ModelType
    vae_path: Optional[str] = None
    precision: str = "bf16"
    device: str = "cuda"


class LoRAConfig(BaseModel):
    path: str
    weight: float = 1.0
    enabled: bool = True
    is_lycoris: bool = False  # Auto-detected from file or manually set


class GenerateRequest(BaseModel):
    # Basic
    prompt: str
    negative_prompt: str = ""
    mode: GenerationMode = GenerationMode.TXT2IMG

    # Dimensions
    width: int = 1024
    height: int = 1024

    # Generation params
    steps: int = 30
    cfg_scale: float = 7.0
    sampler: Sampler = Sampler.EULER
    seed: int = -1

    # Batch
    batch_size: int = 1
    batch_count: int = 1

    # img2img / inpaint
    init_image: Optional[str] = None  # base64 or path
    mask_image: Optional[str] = None  # base64 or path
    strength: float = 0.75

    # Edit mode
    edit_instruction: Optional[str] = None

    # Video
    num_frames: int = 16
    fps: int = 8

    # LoRAs
    loras: List[LoRAConfig] = []

    # Advanced
    clip_skip: int = 1
    vae_tiling: bool = False
    free_u: bool = False

    # Forge-classic features
    rescale_cfg: float = 0.0  # 0 = disabled, 0.7 recommended for v-pred
    mahiro_cfg: bool = False  # Alternative CFG for better prompt adherence
    epsilon_scaling: float = 0.0  # Epsilon scaling factor (0 = disabled)
    skip_early_cond: float = 0.0  # Skip uncond for first N% of steps (0-1)
    use_flash_attention: bool = True
    use_sage_attention: bool = False

    # Hires.fix
    enable_hires: bool = False
    hires_scale: float = 2.0
    hires_steps: int = 20
    hires_denoising: float = 0.5
    hires_upscaler: str = "latent"  # latent, esrgan, real-esrgan, lanczos

    # Upscaler (standalone)
    upscale_enabled: bool = False
    upscaler_model: str = "RealESRGAN_x4plus"
    upscale_factor: float = 2.0

    # ControlNet
    controlnet_enabled: bool = False
    controlnet_model: Optional[str] = None
    controlnet_image: Optional[str] = None
    controlnet_strength: float = 1.0
    controlnet_start: float = 0.0
    controlnet_end: float = 1.0

    # Model selection (for auto-loading)
    model_path: Optional[str] = None
    model_type: Optional[ModelType] = None
    precision: str = "bf16"
    vae_path: Optional[str] = None


class GeneratedImage(BaseModel):
    id: str
    path: str
    thumbnail: str  # base64
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    sampler: str
    seed: int
    model: str
    created_at: str
    generation_time: float


class ModelInfo(BaseModel):
    loaded: bool
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    vae_path: Optional[str] = None
    precision: Optional[str] = None
    loras: List[LoRAConfig] = []


class SystemStatus(BaseModel):
    gpu_name: str
    gpu_memory_total: int
    gpu_memory_used: int
    gpu_memory_free: int
    gpu_utilization: Optional[int] = None
    model_info: ModelInfo
    is_generating: bool
    progress: int
    current_step: int
    total_steps: int


# ============================================================================
# Inference Engine
# ============================================================================

class InferenceEngine:
    """Core inference engine supporting multiple model types."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.model_path: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.vae_path: Optional[str] = None
        self.precision: str = "bf16"
        self.loras: List[LoRAConfig] = []

        # Generation state
        self.is_generating = False
        self.should_cancel = False
        self.progress = 0
        self.current_step = 0
        self.total_steps = 0

        # Output directory
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Gallery
        self.gallery: List[GeneratedImage] = []

        # WebSocket connections
        self.websockets: set = set()
        self._lock = threading.Lock()

    def get_dtype(self, precision: str):
        """Get torch dtype from precision string."""
        dtypes = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtypes.get(precision, torch.bfloat16)

    def load_model(self, request: LoadModelRequest) -> Dict[str, Any]:
        """Load a model for inference."""
        try:
            # Validate model path
            model_path = request.model_path
            if not model_path or not model_path.strip():
                return {"success": False, "error": "No model path specified. Please select a model from the Quick Select dropdown or browse for a model file."}

            model_path = model_path.strip()

            # Check if path exists (allow HF model IDs for certain model types)
            hf_model_types = [ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO,
                             ModelType.QWEN_IMAGE, ModelType.QWEN_IMAGE_EDIT,
                             ModelType.LUMINA, ModelType.LUMINA_2,
                             ModelType.OMNIGEN, ModelType.OMNIGEN_2]
            if not Path(model_path).exists() and request.model_type not in hf_model_types:
                return {"success": False, "error": f"Model path not found: {model_path}"}

            # Unload existing model
            if self.pipeline is not None:
                self.unload_model()

            dtype = self.get_dtype(request.precision)

            print(f"Loading {request.model_type.value} from {model_path}...")

            # Check if it's a single file or directory
            is_single_file = Path(model_path).is_file() and model_path.endswith(('.safetensors', '.ckpt', '.pt', '.bin'))

            # Some models use HF IDs instead of local files - route them through single_file loader
            hf_id_model_types = [ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO,
                                ModelType.QWEN_IMAGE, ModelType.QWEN_IMAGE_EDIT,
                                ModelType.LUMINA, ModelType.LUMINA_2,
                                ModelType.OMNIGEN, ModelType.OMNIGEN_2]
            uses_hf_id = request.model_type in hf_id_model_types and not Path(model_path).exists()

            if is_single_file or uses_hf_id:
                # Single file loading (or HF ID loading)
                self.pipeline = self._load_single_file(model_path, request.model_type, dtype)
            else:
                # Directory loading (from_pretrained)
                pipeline_class = self._get_pipeline_class(request.model_type)
                if pipeline_class is None:
                    return {"success": False, "error": f"Unsupported model type: {request.model_type}"}

                self.pipeline = pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=Path(model_path).exists(),
                )

            if self.pipeline is None:
                return {"success": False, "error": "Failed to load pipeline"}

            # Load custom VAE if specified
            if request.vae_path:
                self._load_vae(request.vae_path, dtype)

            # Check if CPU offload is already enabled (for large models)
            # Large models use CPU offload due to their size - don't try to move them
            large_model_types = [
                ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL, ModelType.FLUX_2_DEV, ModelType.FLUX_FILL,
                ModelType.Z_IMAGE, ModelType.Z_IMAGE_TURBO,
                ModelType.SD_3, ModelType.SD_35, ModelType.SD_35_TURBO,
                ModelType.LUMINA, ModelType.LUMINA_2,
                ModelType.OMNIGEN, ModelType.OMNIGEN_2,
                ModelType.WAN_T2V, ModelType.WAN_I2V,
                ModelType.HUNYUAN_VIDEO,
                ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO
            ]
            is_large_model = request.model_type in large_model_types

            # Check if offload was already enabled during loading
            has_offload = hasattr(self.pipeline, '_hf_hook') or hasattr(self.pipeline, 'hf_device_map')

            if not is_large_model and not has_offload:
                # Move to device for smaller models only
                self.pipeline = self.pipeline.to(self.device)

            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()

            self.model_path = model_path
            self.model_type = request.model_type
            self.vae_path = request.vae_path
            self.precision = request.precision

            print(f"Model loaded successfully: {request.model_type.value}")
            return {"success": True, "message": f"Model loaded: {Path(model_path).name}"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _load_flux_single_file(self, model_path: str, dtype, fill: bool = False):
        """Load FLUX model from single file with components from SwarmUI/Models - Eri approach."""
        from diffusers import FluxPipeline, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, T5Config, CLIPConfig
        from safetensors.torch import load_file

        models_base = "/home/alex/SwarmUI/Models"
        vae_path = f"{models_base}/VAE/ae.safetensors"
        t5_path = f"{models_base}/clip/t5xxl_fp16.safetensors"
        clip_l_path = f"{models_base}/clip/clip_l.safetensors"
        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        # Load CLIP text encoder
        print(f"Loading CLIP text encoder...")
        if Path(clip_l_path).exists():
            # Load from local safetensors file using config from cache
            clip_config = CLIPConfig.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=hf_cache,
                local_files_only=True
            ).text_config
            text_encoder = CLIPTextModel(clip_config)
            clip_weights = load_file(clip_l_path, device="cpu")
            text_encoder.load_state_dict(clip_weights, strict=False)
            text_encoder = text_encoder.to(dtype)
            print(f"✅ Loaded CLIP from local file: {clip_l_path}")
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=dtype,
                cache_dir=hf_cache,
                local_files_only=True
            )
            print("✅ Loaded CLIP from cache")

        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=hf_cache,
            local_files_only=True
        )

        # Load T5 tokenizer
        print("Loading T5 tokenizer...")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            "google/t5-v1_1-xxl",
            cache_dir=hf_cache,
            local_files_only=True
        )

        # Load T5 encoder from local safetensors file - use google config like Eri
        print(f"Loading T5 encoder from {t5_path}...")
        t5_config = T5Config.from_pretrained(
            "google/t5-v1_1-xxl",
            cache_dir=hf_cache,
            local_files_only=True
        )
        text_encoder_2 = T5EncoderModel(t5_config)
        t5_weights = load_file(t5_path, device="cpu")
        text_encoder_2.load_state_dict(t5_weights)  # No strict=False, keys should match
        text_encoder_2 = text_encoder_2.to(dtype)
        print(f"✅ Loaded T5 from local file")

        # Load VAE from local ae.safetensors file
        print(f"Loading VAE from {vae_path}...")
        vae = None
        if Path(vae_path).exists():
            try:
                # Load VAE from single file with config from HF
                vae = AutoencoderKL.from_single_file(
                    vae_path,
                    config="black-forest-labs/FLUX.1-dev",
                    subfolder="vae",
                    torch_dtype=dtype,
                )
                print(f"✅ Loaded FLUX VAE from local file: {vae_path}")
            except Exception as e:
                print(f"⚠️ Could not load VAE from single file: {e}")
                # Try from cache as fallback
                try:
                    vae = AutoencoderKL.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        subfolder="vae",
                        torch_dtype=dtype,
                        cache_dir=hf_cache,
                        local_files_only=True
                    )
                    print("✅ Loaded FLUX VAE from cache")
                except Exception as e2:
                    print(f"⚠️ Could not load VAE from cache either: {e2}")

        # Load pipeline from single file, passing text encoders
        print(f"Loading FLUX pipeline from {model_path}...")
        pipeline_kwargs = {
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "torch_dtype": dtype
        }
        if vae is not None:
            pipeline_kwargs["vae"] = vae

        pipe = FluxPipeline.from_single_file(
            model_path,
            **pipeline_kwargs
        )

        # Enable sequential CPU offload for memory efficiency
        print("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()

        print("✅ FLUX pipeline loaded successfully")
        return pipe

    def _load_sd35_single_file(self, model_path: str, dtype):
        """Load SD3.5 model from single file with text encoders from cache."""
        from diffusers import StableDiffusion3Pipeline
        from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast

        models_base = "/home/alex/SwarmUI/Models"
        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"Loading SD3.5 from {model_path}...")

        # Load CLIP-G text encoder (text_encoder)
        print("Loading CLIP-G text encoder...")
        try:
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=hf_cache,
                local_files_only=True
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="tokenizer",
                cache_dir=hf_cache,
                local_files_only=True
            )
            print("✅ Loaded CLIP-G from cache")
        except Exception as e:
            print(f"⚠️ Loading CLIP-G from HuggingFace: {e}")
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=hf_cache
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="tokenizer",
                cache_dir=hf_cache
            )

        # Load CLIP-L text encoder (text_encoder_2)
        print("Loading CLIP-L text encoder...")
        try:
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                cache_dir=hf_cache,
                local_files_only=True
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="tokenizer_2",
                cache_dir=hf_cache,
                local_files_only=True
            )
            print("✅ Loaded CLIP-L from cache")
        except Exception as e:
            print(f"⚠️ Loading CLIP-L from HuggingFace: {e}")
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                cache_dir=hf_cache
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                subfolder="tokenizer_2",
                cache_dir=hf_cache
            )

        # Load T5 text encoder (text_encoder_3)
        print("Loading T5-XXL text encoder...")
        t5_path = f"{models_base}/clip/t5xxl_fp16.safetensors"
        if Path(t5_path).exists():
            from safetensors.torch import load_file
            from transformers import T5Config
            t5_config = T5Config.from_pretrained("google/t5-v1_1-xxl", cache_dir=hf_cache, local_files_only=True)
            text_encoder_3 = T5EncoderModel(t5_config)
            t5_weights = load_file(t5_path, device="cpu")
            text_encoder_3.load_state_dict(t5_weights, strict=False)
            text_encoder_3 = text_encoder_3.to(dtype)
            print(f"✅ Loaded T5 from local file: {t5_path}")
        else:
            try:
                text_encoder_3 = T5EncoderModel.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-large",
                    subfolder="text_encoder_3",
                    torch_dtype=dtype,
                    cache_dir=hf_cache,
                    local_files_only=True
                )
                print("✅ Loaded T5 from SD3.5 cache")
            except:
                text_encoder_3 = T5EncoderModel.from_pretrained(
                    "google/t5-v1_1-xxl",
                    torch_dtype=dtype,
                    cache_dir=hf_cache,
                    local_files_only=True
                )
                print("✅ Loaded T5 from google cache")

        tokenizer_3 = T5TokenizerFast.from_pretrained(
            "google/t5-v1_1-xxl",
            cache_dir=hf_cache,
            local_files_only=True
        )

        # Load pipeline from single file with text encoders
        print("Loading SD3.5 pipeline from single file...")
        pipe = StableDiffusion3Pipeline.from_single_file(
            model_path,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            torch_dtype=dtype,
        )

        # Enable CPU offload for memory efficiency
        print("Enabling sequential CPU offload for SD3.5...")
        pipe.enable_sequential_cpu_offload()

        print("✅ SD3.5 pipeline loaded successfully")
        return pipe

    def _load_zimage_single_file(self, model_path: str, dtype):
        """Load Z-Image model - try from_pretrained first (cached), then single file."""
        from diffusers import ZImagePipeline

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        # Check if this is a turbo model
        is_turbo = "turbo" in model_path.lower()
        model_id = "Tongyi-MAI/Z-Image-Turbo" if is_turbo else "Tongyi-MAI/Z-Image"

        print(f"Loading Z-Image from cache: {model_id}...")

        try:
            # Try to load from pretrained (uses HF cache)
            pipe = ZImagePipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=hf_cache,
                local_files_only=True
            )
            pipe.enable_sequential_cpu_offload()
            print("✅ Z-Image pipeline loaded from cache")
            return pipe
        except Exception as e:
            print(f"Cache load failed: {e}")
            print("Trying single file load...")

        # Fallback to single file loading
        from diffusers import ZImageTransformer2DModel, AutoencoderKL
        from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

        models_base = "/home/alex/SwarmUI/Models"
        qwen_path = f"{models_base}/clip/qwen_3_4b.safetensors"
        qwen_vae_path = f"{models_base}/VAE/qwen_image_vae.safetensors"
        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"Loading Z-Image from {model_path}...")

        # Try loading pipeline directly first (includes all components)
        try:
            print("Trying ZImagePipeline.from_single_file directly...")
            pipe = ZImagePipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
            print("✅ Z-Image pipeline loaded from single file")
        except Exception as e:
            print(f"Direct load failed: {e}")
            print("Loading components separately...")

            # Load Qwen text encoder - need HF model
            print("Loading Qwen3 text encoder...")
            try:
                text_encoder = Qwen3ForCausalLM.from_pretrained(
                    "Qwen/Qwen3-4B",
                    torch_dtype=dtype,
                    cache_dir=hf_cache,
                    local_files_only=True
                )
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    "Qwen/Qwen3-4B",
                    cache_dir=hf_cache,
                    local_files_only=True
                )
                print("✅ Loaded Qwen3 from cache")
            except Exception as e:
                print(f"⚠️ Could not load Qwen3: {e}")
                raise Exception("Z-Image requires Qwen3 model. Please download it first.")

            # Load transformer from single file
            print(f"Loading Z-Image transformer from {model_path}...")
            transformer = ZImageTransformer2DModel.from_single_file(
                model_path,
                torch_dtype=dtype,
            )

            # Load VAE
            print("Loading VAE...")
            if Path(qwen_vae_path).exists():
                vae = AutoencoderKL.from_single_file(
                    qwen_vae_path,
                    torch_dtype=dtype,
                )
                print(f"✅ Loaded VAE from {qwen_vae_path}")
            else:
                # Try from HF
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse",
                    torch_dtype=dtype,
                    cache_dir=hf_cache,
                    local_files_only=True
                )

            # Assemble pipeline
            pipe = ZImagePipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )

        # Enable CPU offload for memory efficiency
        print("Enabling sequential CPU offload for Z-Image...")
        pipe.enable_sequential_cpu_offload()

        print("✅ Z-Image pipeline loaded successfully")
        return pipe

    def _load_single_file(self, model_path: str, model_type: ModelType, dtype):
        """Load model from a single safetensors/ckpt file."""
        print(f"Loading single file: {model_path}")

        if model_type in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL, ModelType.FLUX_2_DEV]:
            return self._load_flux_single_file(model_path, dtype)
        elif model_type == ModelType.FLUX_FILL:
            return self._load_flux_single_file(model_path, dtype, fill=True)
        elif model_type in [ModelType.Z_IMAGE, ModelType.Z_IMAGE_TURBO]:
            return self._load_zimage_single_file(model_path, dtype)
        elif model_type == ModelType.SDXL:
            from diffusers import StableDiffusionXLPipeline
            return StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
        elif model_type in [ModelType.SD_15, ModelType.SD_21]:
            from diffusers import StableDiffusionPipeline
            return StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
        elif model_type in [ModelType.SD_3, ModelType.SD_35, ModelType.SD_35_TURBO]:
            return self._load_sd35_single_file(model_path, dtype)
        elif model_type in [ModelType.QWEN_IMAGE, ModelType.QWEN_IMAGE_EDIT]:
            return self._load_qwen_image_single_file(model_path, dtype, edit_mode=(model_type == ModelType.QWEN_IMAGE_EDIT))
        elif model_type in [ModelType.LUMINA, ModelType.LUMINA_2]:
            return self._load_lumina_single_file(model_path, dtype, v2=(model_type == ModelType.LUMINA_2))
        elif model_type in [ModelType.OMNIGEN, ModelType.OMNIGEN_2]:
            return self._load_omnigen_single_file(model_path, dtype, v2=(model_type == ModelType.OMNIGEN_2))
        elif model_type in [ModelType.WAN_T2V, ModelType.WAN_I2V, ModelType.WAN_VACE,
                           ModelType.WAN_T2V_HIGH, ModelType.WAN_T2V_LOW,
                           ModelType.WAN_I2V_HIGH, ModelType.WAN_I2V_LOW]:
            return self._load_wan_single_file(model_path, dtype, model_type=model_type)
        elif model_type == ModelType.HUNYUAN_VIDEO:
            return self._load_hunyuan_video_single_file(model_path, dtype)
        elif model_type in [ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO]:
            return self._load_kandinsky5_single_file(model_path, dtype, video=(model_type == ModelType.KANDINSKY_5_VIDEO))
        else:
            # Fallback - try generic single file load
            print(f"No single-file loader for {model_type}, trying from_pretrained")
            pipeline_class = self._get_pipeline_class(model_type)
            if pipeline_class and hasattr(pipeline_class, 'from_single_file'):
                return pipeline_class.from_single_file(model_path, torch_dtype=dtype)
            return None

    def _load_qwen_image_single_file(self, model_path: str, dtype, edit_mode: bool = False):
        """Load Qwen-Image model from HuggingFace (does not support from_single_file)."""
        from diffusers import DiffusionPipeline

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"Loading Qwen-Image {'Edit' if edit_mode else ''} from {model_path}...")

        # Magic quality tokens
        self._qwen_magic_en = ", Ultra HD, 4K, cinematic composition."
        self._qwen_magic_zh = ", 超清，4K，电影级构图."

        # Determine model ID
        model_id = "Qwen/Qwen-Image-Edit" if edit_mode else "Qwen/Qwen-Image"

        # Try loading from HF cache first
        try:
            print(f"Trying to load {model_id} from cache...")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=hf_cache,
                local_files_only=True
            )
            pipe.enable_sequential_cpu_offload()
            print(f"✅ {model_id} loaded from HuggingFace cache")
            return pipe
        except Exception as e:
            print(f"Cache load failed: {e}")

        # Try downloading from HuggingFace
        try:
            print(f"Downloading {model_id} from HuggingFace...")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            pipe.enable_sequential_cpu_offload()
            print(f"✅ {model_id} downloaded and loaded")
            return pipe
        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            raise Exception(f"Could not load {model_id}. Make sure the model is cached or you have internet access.")

    def _load_lumina_single_file(self, model_path: str, dtype, v2: bool = False):
        """Load Lumina/Lumina2 model from single file or HuggingFace."""
        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"Loading Lumina{'2' if v2 else ''} from {model_path}...")

        if v2:
            # Lumina 2 - use from_pretrained (no from_single_file available)
            try:
                from diffusers import Lumina2Pipeline

                # Check if model_path is a local directory or HF ID
                if Path(model_path).is_dir():
                    pipe = Lumina2Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                    )
                else:
                    # Use HuggingFace model ID
                    model_id = "Alpha-VLLM/Lumina-Image-2.0" if model_path in ["lumina_2", "lumina2"] else model_path
                    pipe = Lumina2Pipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        cache_dir=hf_cache,
                    )

                pipe.enable_sequential_cpu_offload()
                print(f"✅ Lumina2 pipeline loaded successfully")
                return pipe
            except Exception as e:
                print(f"❌ Failed to load Lumina2: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Lumina 1.x
            try:
                from diffusers import LuminaPipeline

                if Path(model_path).exists():
                    pipe = LuminaPipeline.from_single_file(model_path, torch_dtype=dtype)
                else:
                    pipe = LuminaPipeline.from_pretrained(
                        "Alpha-VLLM/Lumina-Next-T2I",
                        torch_dtype=dtype,
                        cache_dir=hf_cache,
                    )

                pipe.enable_sequential_cpu_offload()
                print(f"✅ Lumina pipeline loaded successfully")
                return pipe
            except Exception as e:
                print(f"❌ Failed to load Lumina: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _load_omnigen_single_file(self, model_path: str, dtype, v2: bool = False):
        """Load OmniGen/OmniGen2 model from HuggingFace or local submodule."""
        import sys

        omnigen2_path = Path("/home/alex/diffusion-pipe-lyco/submodules/OmniGen2")
        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"Loading OmniGen{'2' if v2 else ''} from {model_path}...")

        if v2:
            # Use OmniGen2 from local submodule
            if omnigen2_path.exists() and str(omnigen2_path) not in sys.path:
                sys.path.insert(0, str(omnigen2_path))

            try:
                from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

                # Check if path is local file or HF repo ID
                if Path(model_path).exists():
                    # Load from local file - need to use from_pretrained with local path
                    pipe = OmniGen2Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )
                else:
                    # Use HuggingFace model ID
                    model_id = "BAAI/OmniGen2" if model_path == "omnigen_2" else model_path
                    pipe = OmniGen2Pipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        cache_dir=hf_cache,
                        trust_remote_code=True,
                    )

                pipe.enable_sequential_cpu_offload()
                print("✅ OmniGen2 pipeline loaded successfully")
                return pipe
            except Exception as e:
                print(f"❌ Failed to load OmniGen2: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # OmniGen v1
            try:
                from diffusers import OmniGenPipeline
                pipe = OmniGenPipeline.from_single_file(model_path, torch_dtype=dtype)
                pipe.enable_sequential_cpu_offload()
                print("✅ OmniGen pipeline loaded successfully")
                return pipe
            except (ImportError, AttributeError):
                print("⚠️ OmniGen from_single_file not available, trying from_pretrained")
                try:
                    from diffusers import OmniGenPipeline
                    pipe = OmniGenPipeline.from_pretrained(
                        "Shitao/OmniGen-v1",
                        torch_dtype=dtype,
                        cache_dir=hf_cache,
                    )
                    pipe.enable_sequential_cpu_offload()
                    return pipe
                except Exception as e:
                    print(f"Failed to load OmniGen: {e}")
                    return None

    def _load_wan_single_file(self, model_path: str, dtype, model_type: ModelType = None):
        """Load Wan video model from single file (supports Wan 2.1 and 2.2)."""
        # Determine pipeline type
        is_i2v = model_type in [ModelType.WAN_I2V, ModelType.WAN_I2V_HIGH, ModelType.WAN_I2V_LOW]
        is_vace = model_type == ModelType.WAN_VACE

        mode_str = "VACE" if is_vace else ("I2V" if is_i2v else "T2V")
        print(f"Loading Wan {mode_str} from {model_path}...")

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")

        try:
            from diffusers import (
                WanPipeline,
                WanImageToVideoPipeline,
                WanTransformer3DModel,
                AutoencoderKLWan,
            )
            from diffusers.schedulers import UniPCMultistepScheduler
            from transformers import UMT5EncoderModel, AutoTokenizer

            # Load transformer from single file
            print(f"Loading Wan transformer from {model_path}...")
            transformer = WanTransformer3DModel.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
            print(f"✅ Loaded transformer")

            # Determine base model ID for other components
            if is_i2v:
                wan_model_id = "Wan-AI/Wan2.1-I2V-14B-Diffusers"
            else:
                wan_model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

            # Load VAE
            print("Loading Wan VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                wan_model_id,
                subfolder="vae",
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            print("✅ Loaded VAE")

            # Load text encoder (UMT5)
            print("Loading UMT5 text encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                wan_model_id,
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                wan_model_id,
                subfolder="tokenizer",
                cache_dir=hf_cache,
            )
            print("✅ Loaded text encoder")

            # Create scheduler
            scheduler = UniPCMultistepScheduler.from_pretrained(
                wan_model_id,
                subfolder="scheduler",
                cache_dir=hf_cache,
            )

            # Build pipeline
            PipeClass = WanImageToVideoPipeline if is_i2v else WanPipeline
            pipe = PipeClass(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
            )

            # Enable memory optimizations
            pipe.enable_model_cpu_offload()

            # Enable VAE optimizations
            if hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()

            print(f"✅ Wan {mode_str} pipeline loaded successfully")
            return pipe

        except Exception as e:
            print(f"Failed to load Wan: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_hunyuan_video_single_file(self, model_path: str, dtype):
        """Load HunyuanVideo model from single file."""
        print(f"Loading HunyuanVideo from {model_path}...")

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
        hunyuan_model_id = "hunyuanvideo-community/HunyuanVideo"

        try:
            from diffusers import (
                HunyuanVideoPipeline,
                HunyuanVideoTransformer3DModel,
                AutoencoderKLHunyuanVideo,
            )
            from transformers import LlamaModel, LlamaTokenizerFast, CLIPTextModel, CLIPTokenizer

            # Load transformer from single file
            print("Loading Hunyuan transformer...")
            transformer = HunyuanVideoTransformer3DModel.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
            print("✅ Loaded transformer")

            # Load text encoders from HuggingFace
            print("Loading LLAMA text encoder...")
            text_encoder = LlamaModel.from_pretrained(
                hunyuan_model_id,
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            tokenizer = LlamaTokenizerFast.from_pretrained(
                hunyuan_model_id,
                subfolder="tokenizer",
                cache_dir=hf_cache,
            )
            print("✅ Loaded LLAMA text encoder")

            print("Loading CLIP text encoder...")
            text_encoder_2 = CLIPTextModel.from_pretrained(
                hunyuan_model_id,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                hunyuan_model_id,
                subfolder="tokenizer_2",
                cache_dir=hf_cache,
            )
            print("✅ Loaded CLIP text encoder")

            # Load VAE
            print("Loading Hunyuan VAE...")
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                hunyuan_model_id,
                subfolder="vae",
                torch_dtype=dtype,
                cache_dir=hf_cache,
            )
            print("✅ Loaded VAE")

            # Load scheduler
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                hunyuan_model_id,
                subfolder="scheduler",
                cache_dir=hf_cache,
            )

            # Create pipeline
            pipe = HunyuanVideoPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler,
            )

            # Enable memory optimizations
            pipe.enable_model_cpu_offload()
            if hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
            if hasattr(vae, 'enable_slicing'):
                vae.enable_slicing()

            print("✅ HunyuanVideo pipeline loaded successfully")
            return pipe

        except Exception as e:
            print(f"Failed to load HunyuanVideo: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_pipeline_class(self, model_type: ModelType):
        """Get the appropriate diffusers pipeline class."""
        try:
            if model_type in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL]:
                from diffusers import FluxPipeline
                return FluxPipeline
            elif model_type == ModelType.FLUX_2_DEV:
                try:
                    from diffusers import Flux2Pipeline
                    return Flux2Pipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif model_type == ModelType.FLUX_FILL:
                from diffusers import FluxFillPipeline
                return FluxFillPipeline
            elif model_type == ModelType.SDXL:
                from diffusers import StableDiffusionXLPipeline
                return StableDiffusionXLPipeline
            elif model_type in [ModelType.SD_15, ModelType.SD_21]:
                from diffusers import StableDiffusionPipeline
                return StableDiffusionPipeline
            elif model_type in [ModelType.SD_3, ModelType.SD_35, ModelType.SD_35_TURBO]:
                from diffusers import StableDiffusion3Pipeline
                return StableDiffusion3Pipeline
            elif model_type in [ModelType.PIXART_ALPHA, ModelType.PIXART_SIGMA]:
                from diffusers import PixArtAlphaPipeline
                return PixArtAlphaPipeline
            elif model_type == ModelType.SANA:
                from diffusers import SanaPipeline
                return SanaPipeline
            elif model_type in [ModelType.QWEN_IMAGE, ModelType.QWEN_IMAGE_EDIT]:
                from diffusers import DiffusionPipeline
                return DiffusionPipeline
            elif model_type in [ModelType.Z_IMAGE, ModelType.Z_IMAGE_TURBO]:
                try:
                    from diffusers import ZImagePipeline
                    return ZImagePipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif model_type == ModelType.LUMINA:
                from diffusers import LuminaPipeline
                return LuminaPipeline
            elif model_type == ModelType.LUMINA_2:
                try:
                    from diffusers import Lumina2Pipeline
                    return Lumina2Pipeline
                except ImportError:
                    from diffusers import LuminaPipeline
                    return LuminaPipeline
            elif model_type in [ModelType.OMNIGEN, ModelType.OMNIGEN_2]:
                from diffusers import OmniGenPipeline
                return OmniGenPipeline
            elif model_type in [ModelType.WAN_T2V, ModelType.WAN_I2V]:
                from diffusers import WanPipeline
                return WanPipeline
            elif model_type == ModelType.HUNYUAN_VIDEO:
                from diffusers import HunyuanVideoPipeline
                return HunyuanVideoPipeline
            else:
                return None
        except ImportError as e:
            print(f"Failed to import pipeline for {model_type}: {e}")
            return None

    def _load_vae(self, vae_path: str, dtype):
        """Load a custom VAE."""
        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
            self.pipeline.vae = vae.to(self.device)
        except Exception as e:
            print(f"Failed to load VAE: {e}")

    def load_lora(self, lora_config: LoRAConfig) -> bool:
        """Load a LoRA or LyCORIS adapter."""
        if self.pipeline is None:
            return False
        try:
            lora_path = Path(lora_config.path)
            lora_name = lora_path.stem

            # Detect if it's a LyCORIS file (check filename patterns or try loading)
            is_lycoris = lora_config.is_lycoris or self._is_lycoris_file(lora_path)

            if is_lycoris:
                # Load as LyCORIS
                return self._load_lycoris(lora_config)
            else:
                # Standard LoRA loading
                self.pipeline.load_lora_weights(lora_config.path, adapter_name=lora_name)
                self.pipeline.set_adapters([lora_name], [lora_config.weight])
                self.loras.append(lora_config)
                print(f"✅ Loaded LoRA: {lora_name} (weight={lora_config.weight})")
                return True
        except Exception as e:
            print(f"Failed to load LoRA {lora_config.path}: {e}")
            # Try as LyCORIS fallback
            try:
                return self._load_lycoris(lora_config)
            except Exception as e2:
                print(f"Also failed as LyCORIS: {e2}")
                return False

    def _is_lycoris_file(self, path: Path) -> bool:
        """Check if file is a LyCORIS model by inspecting keys."""
        try:
            from safetensors.torch import load_file
            if path.suffix == '.safetensors':
                # Load just metadata/keys to check
                state_dict = load_file(str(path), device="cpu")
                # LyCORIS files typically have keys with these patterns
                lycoris_patterns = ['lokr', 'loha', 'hada', 'ia3', 'oft', 'boft', 'glora']
                for key in list(state_dict.keys())[:20]:  # Check first 20 keys
                    key_lower = key.lower()
                    for pattern in lycoris_patterns:
                        if pattern in key_lower:
                            return True
                del state_dict
            return False
        except:
            return False

    def _load_lycoris(self, lora_config: LoRAConfig) -> bool:
        """Load LyCORIS adapter using the LyCORIS library with name mapping."""
        try:
            # Import the mapper
            from lycoris_mapper import create_lycoris_with_mapping, analyze_lycoris_file

            # Get the base model (unet/transformer)
            if hasattr(self.pipeline, 'unet'):
                base_model = self.pipeline.unet
                architecture = "sdxl"
            elif hasattr(self.pipeline, 'transformer'):
                base_model = self.pipeline.transformer
                architecture = "flux"  # Could also be sd3
            else:
                print("⚠️ Could not find base model for LyCORIS")
                return False

            # Analyze the file first
            print(f"Analyzing LyCORIS file...")
            info = analyze_lycoris_file(lora_config.path)
            print(f"  Type: {info['lycoris_type']}, Architecture: {info['architecture']}")

            # Create LyCORIS network with mapping
            network = create_lycoris_with_mapping(
                lora_config.path,
                base_model,
                multiplier=lora_config.weight,
                architecture=info['architecture']
            )

            if network is None:
                print("⚠️ Failed to create LyCORIS network")
                return False

            # Apply and merge the network
            if len(network.loras) > 0:
                network.apply_to()
                network.merge_to(weight=lora_config.weight)
                print(f"✅ Applied and merged {len(network.loras)} LyCORIS modules")
            else:
                print("⚠️ LyCORIS network has 0 modules - weights may not match model architecture")

            # Store reference for later
            if not hasattr(self, '_lycoris_networks'):
                self._lycoris_networks = []
            self._lycoris_networks.append(network)

            lora_config.is_lycoris = True
            self.loras.append(lora_config)
            print(f"✅ Loaded LyCORIS: {Path(lora_config.path).stem} (weight={lora_config.weight})")
            return True

        except ImportError as e:
            print(f"⚠️ LyCORIS mapper import error: {e}")
            # Fallback to basic loading
            return self._load_lycoris_basic(lora_config)
        except Exception as e:
            print(f"Failed to load LyCORIS: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_lycoris_basic(self, lora_config: LoRAConfig) -> bool:
        """Basic LyCORIS loading without mapping (fallback)."""
        try:
            import sys
            lycoris_path = Path("/home/alex/diffusion-pipe-lyco/LyCORIS")
            if lycoris_path.exists() and str(lycoris_path) not in sys.path:
                sys.path.insert(0, str(lycoris_path))

            from lycoris import create_lycoris_from_weights

            if hasattr(self.pipeline, 'unet'):
                base_model = self.pipeline.unet
            elif hasattr(self.pipeline, 'transformer'):
                base_model = self.pipeline.transformer
            else:
                return False

            network, weights_sd = create_lycoris_from_weights(
                lora_config.weight,
                lora_config.path,
                base_model
            )

            if len(network.loras) > 0:
                network.apply_to()
                network.merge_to(weight=lora_config.weight)

            if not hasattr(self, '_lycoris_networks'):
                self._lycoris_networks = []
            self._lycoris_networks.append(network)

            lora_config.is_lycoris = True
            self.loras.append(lora_config)
            print(f"✅ Loaded LyCORIS (basic): {Path(lora_config.path).stem}")
            return True

        except Exception as e:
            print(f"Basic LyCORIS loading failed: {e}")
            return False

    def unload_model(self):
        """Unload the current model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        self.model_path = None
        self.model_type = None
        self.vae_path = None
        self.loras = []

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _needs_model_switch(self, request: GenerateRequest) -> bool:
        """Check if we need to load a different model."""
        if request.model_path is None or request.model_type is None:
            return False
        if self.pipeline is None:
            return True
        # Check if model path or type changed
        return (self.model_path != request.model_path or
                self.model_type != request.model_type)

    def _auto_load_model(self, request: GenerateRequest) -> Dict[str, Any]:
        """Auto-load model if needed. Returns error dict if failed, None if success."""
        if request.model_path is None or request.model_type is None:
            if self.pipeline is None:
                return {"success": False, "error": "No model selected. Please select a model and try again."}
            return None  # Use existing loaded model

        # Check if we need to switch models
        if not self._needs_model_switch(request):
            return None  # Model already loaded

        # Wait for any ongoing generation to finish
        if self.is_generating:
            return {"success": False, "error": "Cannot switch models while generation is in progress. Please wait or cancel."}

        # Notify loading
        self._broadcast_status("loading", f"Loading {request.model_type.value}...")
        print(f"Auto-loading model: {request.model_path} ({request.model_type.value})")

        # Load the model
        load_request = LoadModelRequest(
            model_path=request.model_path,
            model_type=request.model_type,
            precision=request.precision,
            vae_path=request.vae_path
        )
        result = self.load_model(load_request)

        if not result.get("success"):
            self._broadcast_status("error", result.get("error", "Failed to load model"))
            return result

        self._broadcast_status("ready", f"Model loaded: {request.model_type.value}")
        return None  # Success

    def _broadcast_status(self, status: str, message: str):
        """Broadcast status update to all websockets."""
        import asyncio
        data = {"type": "status", "status": status, "message": message}
        for ws in self.websockets:
            try:
                asyncio.create_task(ws.send_json(data))
            except:
                pass

    def generate(self, request: GenerateRequest) -> Dict[str, Any]:
        """Generate images based on request. Auto-loads model if needed."""
        # Auto-load model if specified in request
        load_error = self._auto_load_model(request)
        if load_error is not None:
            return load_error

        if self.pipeline is None:
            return {"success": False, "error": "No model loaded. Please select a model."}

        if self.is_generating:
            return {"success": False, "error": "Generation already in progress"}

        with self._lock:
            self.is_generating = True
            self.should_cancel = False
            self.progress = 0
            self.current_step = 0
            self.total_steps = request.steps

        results = []
        start_time = time.time()

        try:
            # Load LoRAs if specified
            if request.loras:
                for lora in request.loras:
                    if lora.enabled and lora.path:
                        print(f"Loading LoRA: {lora.path} (weight: {lora.weight})")
                        self.load_lora(lora)

            for batch_idx in range(request.batch_count):
                if self.should_cancel:
                    break

                # Generate seed
                seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
                if batch_idx > 0:
                    seed = random.randint(0, 2**32 - 1)

                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Build pipeline kwargs
                kwargs = self._build_pipeline_kwargs(request, generator)

                # Check if Kandinsky model
                is_kandinsky = self.model_type in [ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO]

                # Progress callback (not supported by Kandinsky)
                if not is_kandinsky:
                    def progress_callback(pipe, step, timestep, callback_kwargs):
                        self.current_step = step + 1
                        self.progress = int((step + 1) / request.steps * 100)
                        self._broadcast_progress()
                        if self.should_cancel:
                            raise InterruptedError("Cancelled")
                        return callback_kwargs
                    kwargs["callback_on_step_end"] = progress_callback

                    # Set optimal scheduler for model type (SDXL uses DPM++ 2M)
                    self._setup_scheduler_for_model()

                # For Kandinsky video, add save_path
                if is_kandinsky and self.model_type == ModelType.KANDINSKY_5_VIDEO:
                    video_id = str(uuid.uuid4())[:8]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{timestamp}_{video_id}.mp4"
                    video_path = self.output_dir / video_filename
                    kwargs["save_path"] = str(video_path)

                # Run inference
                result = self.pipeline(**kwargs)

                # Process output
                if is_kandinsky:
                    gen_time = time.time() - start_time
                    if self.model_type == ModelType.KANDINSKY_5_VIDEO:
                        # Video was saved via save_path
                        saved = self._save_kandinsky_video(str(video_path), request, seed, gen_time)
                        results.append(saved)
                    else:
                        # T2I returns PIL.Image directly
                        if isinstance(result, Image.Image):
                            saved = self._save_image(result, request, seed, gen_time)
                            results.append(saved)
                        else:
                            print(f"Warning: Unexpected Kandinsky result type: {type(result)}")
                elif hasattr(result, 'images') and result.images:
                    for img in result.images:
                        # Apply Hires.fix if enabled
                        if request.enable_hires:
                            img = self._run_hires_fix(img, request, generator)

                        # Apply standalone upscaling if enabled
                        if request.upscale_enabled:
                            img = self._upscale_with_realesrgan(img, request.upscale_factor)

                        gen_time = time.time() - start_time
                        saved = self._save_image(img, request, seed, gen_time)
                        results.append(saved)
                elif hasattr(result, 'frames'):
                    # Video output
                    gen_time = time.time() - start_time
                    saved = self._save_video(result.frames, request, seed, gen_time)
                    results.append(saved)

            return {"success": True, "images": results}

        except InterruptedError:
            return {"success": False, "error": "Generation cancelled"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
        finally:
            with self._lock:
                self.is_generating = False
                self.progress = 0

    def _setup_scheduler_for_model(self):
        """Setup optimal scheduler based on model type - from Eri approach."""
        if self.pipeline is None or not hasattr(self.pipeline, 'scheduler'):
            return

        scheduler_name = self.pipeline.scheduler.__class__.__name__

        # Don't change FlowMatch schedulers (FLUX, SD3.5)
        if 'FlowMatch' in scheduler_name:
            return

        # Use DPM++ 2M for SDXL and other SD models
        try:
            from diffusers import DPMSolverMultistepScheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            print(f"✅ Scheduler set to DPM++ 2M Karras")
        except Exception as e:
            print(f"⚠️ Could not set DPM++ scheduler: {e}")

    def _apply_forge_features(self, request: GenerateRequest):
        """Apply forge-classic features to pipeline."""
        if self.pipeline is None:
            return

        # Enable attention optimizations
        if request.use_flash_attention:
            try:
                # Try to enable flash attention via xformers or native
                if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ Enabled xformers memory efficient attention")
                elif hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing("auto")
            except Exception as e:
                print(f"⚠️ Flash attention not available: {e}")

        # Enable FreeU if requested
        if request.free_u:
            try:
                if hasattr(self.pipeline, 'enable_freeu'):
                    # Default FreeU parameters for SDXL
                    self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
                    print("✅ Enabled FreeU enhancement")
            except Exception as e:
                print(f"⚠️ FreeU not available: {e}")

        # Enable VAE tiling for large images
        if request.vae_tiling:
            try:
                if hasattr(self.pipeline, 'enable_vae_tiling'):
                    self.pipeline.enable_vae_tiling()
                    print("✅ Enabled VAE tiling")
            except Exception as e:
                print(f"⚠️ VAE tiling not available: {e}")

        # Store forge settings for callback use
        self._rescale_cfg = request.rescale_cfg
        self._mahiro_cfg = request.mahiro_cfg
        self._epsilon_scaling = request.epsilon_scaling
        self._skip_early_cond = request.skip_early_cond

    def _apply_rescale_cfg(self, noise_pred, noise_pred_uncond, guidance_scale):
        """
        Apply RescaleCFG to reduce burnt colors at high CFG.
        From forge-classic: prevents over-saturation especially for v-pred models.
        """
        if not hasattr(self, '_rescale_cfg') or self._rescale_cfg <= 0:
            return noise_pred

        # Standard CFG result
        noise_cfg = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        # Calculate rescale factor
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        std_pos = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)

        # Rescale to match original std
        rescale_factor = std_pos / (std_cfg + 1e-8)
        rescaled = noise_cfg * rescale_factor

        # Interpolate based on rescale_cfg value
        result = self._rescale_cfg * rescaled + (1 - self._rescale_cfg) * noise_cfg

        return result

    def _apply_mahiro_cfg(self, noise_pred_cond, noise_pred_uncond, guidance_scale):
        """
        MaHiRo CFG - Alternative CFG calculation for better prompt adherence.
        Uses the magnitude of the conditional prediction to scale the guidance.
        """
        # Calculate direction and magnitude
        diff = noise_pred_cond - noise_pred_uncond
        magnitude = torch.norm(diff, dim=1, keepdim=True)

        # Normalize direction
        direction = diff / (magnitude + 1e-8)

        # Apply guidance with magnitude preservation
        guided_magnitude = magnitude * guidance_scale
        result = noise_pred_uncond + direction * guided_magnitude

        return result

    def _apply_epsilon_scaling(self, noise_pred, sigma):
        """
        Epsilon scaling for refined generation control.
        Scales the noise prediction based on the current sigma value.
        """
        if not hasattr(self, '_epsilon_scaling') or self._epsilon_scaling <= 0:
            return noise_pred

        # Scale epsilon based on sigma
        scale_factor = 1.0 + self._epsilon_scaling * (1.0 - sigma / (sigma + 1.0))
        return noise_pred * scale_factor

    def _run_hires_fix(self, image: Image.Image, request: GenerateRequest, generator) -> Image.Image:
        """
        Hires.fix - Two-pass upscaling with denoising.
        1. Upscale the image
        2. Run img2img with low denoising to add detail
        """
        print(f"🔍 Running Hires.fix: {request.hires_scale}x, {request.hires_steps} steps, {request.hires_denoising} denoise")

        # Calculate new dimensions
        new_width = int(image.width * request.hires_scale)
        new_height = int(image.height * request.hires_scale)

        # Round to 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        # Step 1: Upscale
        if request.hires_upscaler == "latent":
            # Upscale in latent space (just resize for now)
            upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif request.hires_upscaler in ["esrgan", "real-esrgan", "realesrgan"]:
            upscaled = self._upscale_with_realesrgan(image, request.hires_scale)
        else:
            # Lanczos fallback
            upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Step 2: img2img pass
        if self.pipeline is None:
            return upscaled

        try:
            hires_kwargs = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt if self.model_type not in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL] else None,
                "image": upscaled,
                "strength": request.hires_denoising,
                "num_inference_steps": request.hires_steps,
                "guidance_scale": request.cfg_scale,
                "generator": generator,
            }
            # Remove None values
            hires_kwargs = {k: v for k, v in hires_kwargs.items() if v is not None}

            result = self.pipeline(**hires_kwargs)

            if hasattr(result, 'images') and result.images:
                print(f"✅ Hires.fix complete: {new_width}x{new_height}")
                return result.images[0]
        except Exception as e:
            print(f"⚠️ Hires.fix img2img failed: {e}, returning upscaled image")

        return upscaled

    def _upscale_with_realesrgan(self, image: Image.Image, scale: float) -> Image.Image:
        """Upscale image using RealESRGAN."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            import numpy as np

            # Initialize RealESRGAN
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

            upsampler = RealESRGANer(
                scale=4,
                model_path=None,  # Will use default
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True
            )

            # Convert to numpy
            img_np = np.array(image)

            # Upscale
            output, _ = upsampler.enhance(img_np, outscale=scale)

            return Image.fromarray(output)

        except ImportError:
            print("⚠️ RealESRGAN not available, using Lanczos")
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"⚠️ RealESRGAN failed: {e}, using Lanczos")
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)

    def _load_kandinsky5_single_file(self, model_path: str, dtype, video: bool = False):
        """Load Kandinsky 5 model using local repo pipelines."""
        import sys

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
        kandinsky_repo = Path("/home/alex/OneTrainer/models/kandinsky-5-code")

        # Add Kandinsky repo to path
        if kandinsky_repo.exists() and str(kandinsky_repo) not in sys.path:
            sys.path.insert(0, str(kandinsky_repo))

        print(f"Loading Kandinsky 5 {'Video' if video else 'Image'} from {model_path}...")

        # Check if model_path is local file or needs to use default weights
        if not Path(model_path).exists():
            # Check for local weights
            if video:
                local_weights = Path("/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors")
            else:
                local_weights = Path("/home/alex/OneTrainer/models/kandinsky-5-code")  # T2I weights path
            if local_weights.exists():
                model_path = str(local_weights)
                print(f"Using local weights: {model_path}")

        # Detect model variant (Pro vs Lite) based on filename
        is_pro = "pro" in model_path.lower()
        variant = "Pro" if is_pro else "Lite"
        print(f"Detected Kandinsky 5 {variant} model")

        # Get appropriate config file
        config_dir = kandinsky_repo / "configs"
        if video:
            if is_pro:
                config_file = config_dir / "k5_pro_t2v_5s_sft_sd.yaml"
            else:
                config_file = config_dir / "k5_lite_t2v_5s_sft_sd.yaml"
        else:
            if is_pro:
                # Pro T2I config doesn't exist in configs, use T2V Pro config architecture
                config_file = None  # Will use default with modified params
            else:
                config_file = config_dir / "k5_lite_t2i_sft_hd.yaml"

        try:
            from kandinsky.utils import get_T2V_pipeline, get_T2I_pipeline
            from omegaconf import OmegaConf

            # Determine device map
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = {"dit": device, "vae": device, "text_embedder": device}

            if video:
                # Load T2V pipeline with appropriate config
                print(f"Loading Kandinsky 5 {variant} T2V pipeline...")

                dit_path = model_path if Path(model_path).exists() else None
                conf_path_str = str(config_file) if config_file and config_file.exists() else None
                
                if conf_path_str:
                    print(f"Using config: {conf_path_str}")
                print(f"Using dit_path: {dit_path}")
                
                # Pass HF cache IDs for text encoders, let kandinsky handle VAE
                pipe = get_T2V_pipeline(
                    device_map=device_map,
                    cache_dir=hf_cache,
                    dit_path=dit_path,
                    text_encoder_path="Qwen/Qwen2.5-VL-7B-Instruct",
                    text_encoder2_path="openai/clip-vit-large-patch14",
                    conf_path=conf_path_str,
                    offload=True,
                )

                print(f"✅ Kandinsky 5 {variant} T2V pipeline loaded")
                return pipe
            else:
                # Load T2I pipeline
                print(f"Loading Kandinsky 5 {variant} T2I pipeline...")

                dit_path = model_path if Path(model_path).exists() else None
                conf_path_str = str(config_file) if config_file and config_file.exists() else None
                
                if conf_path_str:
                    print(f"Using config: {conf_path_str}")
                print(f"Using dit_path: {dit_path}")

                pipe = get_T2I_pipeline(
                    device_map=device_map,
                    cache_dir=hf_cache,
                    dit_path=dit_path,
                    text_encoder_path="Qwen/Qwen2.5-VL-7B-Instruct",
                    text_encoder2_path="openai/clip-vit-large-patch14",
                    conf_path=conf_path_str,
                    offload=True,
                )

                print(f"✅ Kandinsky 5 {variant} T2I pipeline loaded")
                return pipe

        except Exception as e:
            print(f"❌ Failed to load Kandinsky 5 via local repo: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_controlnet(self, controlnet_model: str):
        """Load a ControlNet model."""
        try:
            from diffusers import ControlNetModel

            # Check if it's a path or model ID
            if Path(controlnet_model).exists():
                controlnet = ControlNetModel.from_single_file(
                    controlnet_model,
                    torch_dtype=self.get_dtype(self.precision)
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model,
                    torch_dtype=self.get_dtype(self.precision)
                )

            print(f"✅ Loaded ControlNet: {controlnet_model}")
            return controlnet

        except Exception as e:
            print(f"❌ Failed to load ControlNet: {e}")
            return None

    def _apply_controlnet(self, request: GenerateRequest, kwargs: Dict[str, Any]):
        """Apply ControlNet to generation kwargs."""
        if not request.controlnet_enabled or not request.controlnet_model:
            return

        try:
            from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline

            # Load ControlNet
            controlnet = self._load_controlnet(request.controlnet_model)
            if controlnet is None:
                return

            # Load control image
            if request.controlnet_image:
                control_image = self._load_image(request.controlnet_image, request.width, request.height)
                kwargs["image"] = control_image
                kwargs["controlnet_conditioning_scale"] = request.controlnet_strength

                # Swap to ControlNet pipeline
                if self.model_type == ModelType.SDXL:
                    self.pipeline = StableDiffusionXLControlNetPipeline(
                        vae=self.pipeline.vae,
                        text_encoder=self.pipeline.text_encoder,
                        text_encoder_2=self.pipeline.text_encoder_2,
                        tokenizer=self.pipeline.tokenizer,
                        tokenizer_2=self.pipeline.tokenizer_2,
                        unet=self.pipeline.unet,
                        scheduler=self.pipeline.scheduler,
                        controlnet=controlnet,
                    )
                print(f"✅ ControlNet applied with strength {request.controlnet_strength}")

        except Exception as e:
            print(f"⚠️ ControlNet application failed: {e}")

    def _build_pipeline_kwargs(self, request: GenerateRequest, generator) -> Dict[str, Any]:
        """Build kwargs for pipeline call based on mode."""
        # Check if this is a Kandinsky 5 model
        is_kandinsky = self.model_type in [ModelType.KANDINSKY_5, ModelType.KANDINSKY_5_VIDEO]

        if is_kandinsky:
            # Kandinsky 5 uses different parameter names
            # Extract seed from generator
            seed = generator.initial_seed() if hasattr(generator, 'initial_seed') else 42

            kwargs = {
                "text": request.prompt,
                "num_steps": request.steps,
                "guidance_weight": request.cfg_scale,
                "seed": seed,
                "width": request.width,
                "height": request.height,
                "negative_caption": request.negative_prompt or "",
                "expand_prompts": True,
                "progress": True,
            }

            # For video models, add time_length
            if self.model_type == ModelType.KANDINSKY_5_VIDEO or request.mode == GenerationMode.VIDEO:
                # Default to 5 seconds, or calculate from num_frames
                time_length = 5
                if hasattr(request, 'num_frames') and request.num_frames > 0:
                    # Kandinsky: num_frames = time_length * 24 // 4 + 1
                    time_length = max(1, (request.num_frames - 1) * 4 // 24)
                kwargs["time_length"] = time_length
                kwargs["scheduler_scale"] = 10.0
            else:
                kwargs["scheduler_scale"] = 3.0

            return kwargs

        # Standard diffusers parameters for non-Kandinsky models
        kwargs = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.cfg_scale,
            "generator": generator,
        }

        # Add negative prompt for non-FLUX models
        if self.model_type not in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL, ModelType.FLUX_2_DEV, ModelType.FLUX_FILL]:
            kwargs["negative_prompt"] = request.negative_prompt

        # Apply forge-classic features
        self._apply_forge_features(request)

        # Apply ControlNet if enabled
        if request.controlnet_enabled:
            self._apply_controlnet(request, kwargs)

        # Mode-specific kwargs
        if request.mode == GenerationMode.TXT2IMG:
            kwargs["width"] = request.width
            kwargs["height"] = request.height
        elif request.mode == GenerationMode.IMG2IMG:
            kwargs["image"] = self._load_image(request.init_image, request.width, request.height)
            kwargs["strength"] = request.strength
        elif request.mode == GenerationMode.INPAINT:
            kwargs["image"] = self._load_image(request.init_image, request.width, request.height)
            kwargs["mask_image"] = self._load_image(request.mask_image, request.width, request.height, mode="L")
            kwargs["strength"] = request.strength
        elif request.mode == GenerationMode.EDIT:
            kwargs["image"] = self._load_image(request.init_image, request.width, request.height)
            if request.edit_instruction:
                kwargs["prompt"] = request.edit_instruction
        elif request.mode == GenerationMode.VIDEO:
            kwargs["width"] = request.width
            kwargs["height"] = request.height
            kwargs["num_frames"] = request.num_frames

        return kwargs

    def _load_image(self, image_data: str, width: int, height: int, mode: str = "RGB") -> Image.Image:
        """Load image from base64 or file path."""
        if image_data.startswith("data:"):
            # Base64 data URL
            base64_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(img_bytes))
        elif Path(image_data).exists():
            img = Image.open(image_data)
        else:
            # Try as raw base64
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))

        img = img.convert(mode)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img

    def _save_image(self, image: Image.Image, request: GenerateRequest, seed: int, gen_time: float) -> GeneratedImage:
        """Save generated image and create record."""
        image_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image_id}.png"
        filepath = self.output_dir / filename

        image.save(filepath, "PNG")

        # Create thumbnail
        thumb = image.copy()
        thumb.thumbnail((256, 256))
        thumb_buffer = io.BytesIO()
        thumb.save(thumb_buffer, format="PNG")
        thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode()

        record = GeneratedImage(
            id=image_id,
            path=str(filepath),
            thumbnail=f"data:image/png;base64,{thumb_b64}",
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            sampler=request.sampler.value,
            seed=seed,
            model=self.model_path or "unknown",
            created_at=datetime.now().isoformat(),
            generation_time=gen_time,
        )

        self.gallery.insert(0, record)
        return record

    def _save_video(self, frames, request: GenerateRequest, seed: int, gen_time: float) -> GeneratedImage:
        """Save generated video."""
        from torchvision.io import write_video

        video_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{video_id}.mp4"
        filepath = self.output_dir / filename

        # Convert frames to video
        if len(frames.shape) == 5:
            frames = frames[0]
        write_video(str(filepath), frames.permute(0, 2, 3, 1).cpu(), fps=request.fps)

        record = GeneratedImage(
            id=video_id,
            path=str(filepath),
            thumbnail="",  # TODO: Generate video thumbnail
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            sampler=request.sampler.value,
            seed=seed,
            model=self.model_path or "unknown",
            created_at=datetime.now().isoformat(),
            generation_time=gen_time,
        )

        self.gallery.insert(0, record)
        return record

    def _save_kandinsky_video(self, video_path: str, request: GenerateRequest, seed: int, gen_time: float) -> GeneratedImage:
        """Save Kandinsky video that was already saved by the pipeline."""
        video_id = str(uuid.uuid4())[:8]

        record = GeneratedImage(
            id=video_id,
            path=video_path,
            thumbnail="",  # TODO: Generate video thumbnail
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            sampler=request.sampler.value if hasattr(request.sampler, 'value') else str(request.sampler),
            seed=seed,
            model=self.model_path or "unknown",
            created_at=datetime.now().isoformat(),
            generation_time=gen_time,
        )

        self.gallery.insert(0, record)
        return record

    def cancel(self):
        """Cancel current generation."""
        self.should_cancel = True

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        gpu_name = "CPU"
        gpu_total = 0
        gpu_used = 0
        gpu_free = 0
        gpu_util = None

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated(0)
            gpu_free = gpu_total - gpu_used

        return SystemStatus(
            gpu_name=gpu_name,
            gpu_memory_total=gpu_total,
            gpu_memory_used=gpu_used,
            gpu_memory_free=gpu_free,
            gpu_utilization=gpu_util,
            model_info=ModelInfo(
                loaded=self.pipeline is not None,
                model_path=self.model_path,
                model_type=self.model_type.value if self.model_type else None,
                vae_path=self.vae_path,
                precision=self.precision,
                loras=self.loras,
            ),
            is_generating=self.is_generating,
            progress=self.progress,
            current_step=self.current_step,
            total_steps=self.total_steps,
        )

    def _broadcast_progress(self):
        """Broadcast progress to WebSocket clients."""
        # This will be called from the progress callback
        pass  # Implement WebSocket broadcast


# ============================================================================
# FastAPI App
# ============================================================================

engine = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Inference App starting...")
    yield
    print("Inference App shutting down...")
    engine.unload_model()


app = FastAPI(
    title="OneTrainer Inference",
    description="Standalone inference server for diffusion models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/status")
async def get_status() -> SystemStatus:
    """Get system and generation status."""
    return engine.get_status()


@app.post("/api/model/load")
async def load_model(request: LoadModelRequest):
    """Load a model for inference."""
    result = engine.load_model(request)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/model/unload")
async def unload_model():
    """Unload the current model."""
    engine.unload_model()
    return {"success": True, "message": "Model unloaded"}


@app.post("/api/model/lora")
async def load_lora(config: LoRAConfig):
    """Load a LoRA adapter."""
    if engine.load_lora(config):
        return {"success": True}
    raise HTTPException(status_code=400, detail="Failed to load LoRA")


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate images."""
    result = engine.generate(request)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/generate/cancel")
async def cancel_generation():
    """Cancel current generation."""
    engine.cancel()
    return {"success": True}


@app.get("/api/gallery")
async def get_gallery(limit: int = 50):
    """Get generated images gallery."""
    return {"images": engine.gallery[:limit]}


@app.get("/api/gallery/{image_id}")
async def get_image(image_id: str):
    """Get a specific image."""
    for img in engine.gallery:
        if img.id == image_id:
            return FileResponse(img.path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/api/gallery/{image_id}")
async def delete_image(image_id: str):
    """Delete an image."""
    for i, img in enumerate(engine.gallery):
        if img.id == image_id:
            try:
                Path(img.path).unlink(missing_ok=True)
            except:
                pass
            engine.gallery.pop(i)
            return {"success": True}
    raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/api/gallery")
async def clear_gallery():
    """Clear all gallery images."""
    for img in engine.gallery:
        try:
            Path(img.path).unlink(missing_ok=True)
        except:
            pass
    engine.gallery.clear()
    return {"success": True}


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for img2img/inpainting."""
    content = await file.read()
    b64 = base64.b64encode(content).decode()
    return {"image": f"data:image/png;base64,{b64}"}


# ============================================================================
# SAM2 Segmentation API
# ============================================================================

class SegmentPointRequest(BaseModel):
    """Request for point-based segmentation."""
    image: str  # base64 encoded image
    points: List[List[int]]  # [[x, y], [x, y], ...]
    labels: Optional[List[int]] = None  # 1 = foreground, 0 = background

class SegmentBoxRequest(BaseModel):
    """Request for box-based segmentation."""
    image: str  # base64 encoded image
    box: List[int]  # [x1, y1, x2, y2]

class AutoSegmentRequest(BaseModel):
    """Request for automatic segmentation."""
    image: str  # base64 encoded image

# SAM2 instance (lazy loaded)
_sam2_instance = None

def get_sam2():
    """Get or create SAM2 instance."""
    global _sam2_instance
    if _sam2_instance is None:
        try:
            from tools.segmentation_tools import SAM2Segmenter
            _sam2_instance = SAM2Segmenter()
            _sam2_instance.load()
        except Exception as e:
            print(f"Failed to load SAM2: {e}")
            return None
    return _sam2_instance

def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

def encode_mask(mask) -> str:
    """Encode numpy mask to base64 PNG."""
    import numpy as np
    # Convert boolean/float mask to uint8
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = (mask * 255).astype(np.uint8)

    mask_img = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/api/segment/point")
async def segment_point(request: SegmentPointRequest):
    """Segment image based on point clicks (SAM2)."""
    sam2 = get_sam2()
    if sam2 is None:
        raise HTTPException(status_code=503, detail="SAM2 not available")

    try:
        image = decode_image(request.image)
        points = [(p[0], p[1]) for p in request.points]
        labels = request.labels if request.labels else [1] * len(points)

        mask = sam2.segment_point(image, points, labels)
        mask_b64 = encode_mask(mask)

        return {
            "success": True,
            "mask": f"data:image/png;base64,{mask_b64}",
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/box")
async def segment_box(request: SegmentBoxRequest):
    """Segment image based on bounding box (SAM2)."""
    sam2 = get_sam2()
    if sam2 is None:
        raise HTTPException(status_code=503, detail="SAM2 not available")

    try:
        image = decode_image(request.image)
        mask = sam2.segment_box(image, tuple(request.box))
        mask_b64 = encode_mask(mask)

        return {
            "success": True,
            "mask": f"data:image/png;base64,{mask_b64}",
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment/auto")
async def segment_auto(request: AutoSegmentRequest):
    """Automatically segment all objects in image (SAM2)."""
    sam2 = get_sam2()
    if sam2 is None:
        raise HTTPException(status_code=503, detail="SAM2 not available")

    try:
        image = decode_image(request.image)
        masks = sam2.auto_segment(image)

        masks_b64 = [f"data:image/png;base64,{encode_mask(m)}" for m in masks]

        return {
            "success": True,
            "masks": masks_b64,
            "count": len(masks),
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browse")
async def browse_files(path: str = ".", extensions: str = ""):
    """Browse filesystem for models/images."""
    p = Path(path).expanduser()
    if not p.exists():
        p = Path.home()

    ext_list = [e.strip() for e in extensions.split(",") if e.strip()]

    items = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue

            is_dir = item.is_dir()
            if not is_dir and ext_list:
                if not any(item.suffix.lower() == f".{ext}" for ext in ext_list):
                    continue

            items.append({
                "name": item.name,
                "path": str(item),
                "is_dir": is_dir,
                "size": item.stat().st_size if not is_dir else 0,
            })
    except PermissionError:
        pass

    return {
        "current_path": str(p),
        "parent_path": str(p.parent),
        "items": items,
    }


@app.get("/api/config/samplers")
async def get_samplers():
    """Get available samplers."""
    return {"samplers": [s.value for s in Sampler]}


@app.get("/api/config/model_types")
async def get_model_types():
    """Get available model types."""
    return {"model_types": [m.value for m in ModelType]}


@app.get("/api/config/aspect_ratios")
async def get_aspect_ratios():
    """Get aspect ratio presets."""
    return {"aspect_ratios": ASPECT_RATIOS}


@app.get("/api/config/resolutions")
async def get_resolutions():
    """Get resolution presets."""
    return {"resolutions": RESOLUTION_PRESETS}


# ============================================================================
# Video Editor API
# ============================================================================

from video_editor_ffmpeg import (
    get_video_editor, Project, Track, Clip, Effect, MediaFile,
    ClipType, EffectType, TransitionType
)

class NewProjectRequest(BaseModel):
    name: str = "Untitled"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0

class AddClipRequest(BaseModel):
    type: str = "video"
    name: str = ""
    source_path: str = ""
    track_id: str = ""
    start_time: float = 0.0
    duration: float = 5.0
    text_content: str = ""
    font_size: int = 48
    font_color: str = "#FFFFFF"

class UpdateClipRequest(BaseModel):
    updates: Dict[str, Any]

class AddEffectRequest(BaseModel):
    effect_type: str
    params: Dict[str, Any] = {}

# Helper functions for serialization
def clip_to_dict(clip: Clip) -> Dict[str, Any]:
    return {
        "id": clip.id,
        "type": clip.type.value if hasattr(clip.type, 'value') else clip.type,
        "name": clip.name,
        "media_id": clip.media_id,
        "source_path": clip.source_path,
        "source_in": clip.source_in,
        "source_out": clip.source_out,
        "track_id": clip.track_id,
        "start_time": clip.start_time,
        "duration": clip.duration,
        "end_time": clip.end_time,
        "position_x": clip.position_x,
        "position_y": clip.position_y,
        "scale": clip.scale,
        "rotation": clip.rotation,
        "opacity": clip.opacity,
        "volume": clip.volume,
        "muted": clip.muted,
        "text_content": clip.text_content,
        "font_family": clip.font_family,
        "font_size": clip.font_size,
        "font_color": clip.font_color,
        "color": clip.color,
        "effects": [{"id": e.id, "type": e.type.value, "enabled": e.enabled, "params": e.params} for e in clip.effects],
    }

def track_to_dict(track: Track) -> Dict[str, Any]:
    return {
        "id": track.id,
        "name": track.name,
        "type": track.type,
        "order": track.order,
        "muted": track.muted,
        "locked": track.locked,
        "visible": track.visible,
        "height": track.height,
    }

def media_to_dict(media: MediaFile) -> Dict[str, Any]:
    return {
        "id": media.id,
        "path": media.path,
        "name": media.name,
        "type": media.type,
        "duration": media.duration,
        "width": media.width,
        "height": media.height,
        "fps": media.fps,
        "codec": media.codec,
        "file_size": media.file_size,
    }

def project_to_dict(project: Project) -> Dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "width": project.width,
        "height": project.height,
        "fps": project.fps,
        "sample_rate": project.sample_rate,
        "background_color": project.background_color,
        "duration": project.duration,
        "media": [media_to_dict(m) for m in project.media],
        "tracks": [track_to_dict(t) for t in project.tracks],
        "clips": [clip_to_dict(c) for c in project.clips],
    }

@app.post("/api/editor/project/new")
async def editor_new_project(request: NewProjectRequest):
    """Create a new video editor project."""
    editor = get_video_editor()
    project = editor.new_project(
        name=request.name,
        width=request.width,
        height=request.height,
        fps=request.fps
    )
    return {"success": True, "project": project_to_dict(project)}

@app.get("/api/editor/project")
async def editor_get_project():
    """Get current project."""
    editor = get_video_editor()
    if not editor.project:
        return {"success": False, "error": "No project loaded"}
    return {"success": True, "project": project_to_dict(editor.project)}

@app.get("/api/editor/timeline")
async def editor_get_timeline():
    """Get timeline data for UI."""
    editor = get_video_editor()
    if not editor.project:
        return {"tracks": [], "clips": [], "duration": 0}
    return {
        "tracks": [track_to_dict(t) for t in editor.project.tracks],
        "clips": [clip_to_dict(c) for c in editor.project.clips],
        "duration": editor.project.duration,
    }

@app.post("/api/editor/track/add")
async def editor_add_track(name: str, track_type: str = "video"):
    """Add a new track."""
    editor = get_video_editor()
    try:
        track = editor.add_track(name, track_type)
        return {"success": True, "track": track_to_dict(track)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/editor/track/{track_id}")
async def editor_remove_track(track_id: str):
    """Remove a track."""
    editor = get_video_editor()
    success = editor.remove_track(track_id)
    return {"success": success}

@app.post("/api/editor/clip/add")
async def editor_add_clip(request: AddClipRequest):
    """Add a clip to the timeline."""
    editor = get_video_editor()
    try:
        clip_data = {
            "type": request.type,
            "name": request.name,
            "source_path": request.source_path,
            "track_id": request.track_id,
            "start_time": request.start_time,
            "source_in": 0,
            "source_out": request.duration,
            "text_content": request.text_content,
            "font_size": request.font_size,
            "font_color": request.font_color,
        }
        added_clip = editor.add_clip(clip_data)
        return {"success": True, "clip": clip_to_dict(added_clip)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/editor/clip/{clip_id}")
async def editor_remove_clip(clip_id: str):
    """Remove a clip."""
    editor = get_video_editor()
    success = editor.remove_clip(clip_id)
    return {"success": success}

@app.put("/api/editor/clip/{clip_id}")
async def editor_update_clip(clip_id: str, request: UpdateClipRequest):
    """Update a clip's properties."""
    editor = get_video_editor()
    clip = editor.update_clip(clip_id, request.updates)
    if clip:
        return {"success": True, "clip": clip_to_dict(clip)}
    return {"success": False, "error": "Clip not found"}

@app.post("/api/editor/clip/{clip_id}/split")
async def editor_split_clip(clip_id: str, split_time: float):
    """Split a clip at specified time."""
    editor = get_video_editor()
    clip1, clip2 = editor.split_clip(clip_id, split_time)
    if clip1 and clip2:
        return {
            "success": True,
            "clips": [clip_to_dict(clip1), clip_to_dict(clip2)]
        }
    return {"success": False, "error": "Failed to split clip"}

@app.post("/api/editor/clip/{clip_id}/effect")
async def editor_add_effect(clip_id: str, request: AddEffectRequest):
    """Add effect to a clip."""
    editor = get_video_editor()
    try:
        effect = editor.add_effect(clip_id, request.effect_type, request.params)
        if effect:
            return {"success": True, "effect": {"id": effect.id, "type": effect.type.value, "enabled": effect.enabled, "params": effect.params}}
        return {"success": False, "error": "Failed to add effect"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/editor/clip/{clip_id}/effect/{effect_id}")
async def editor_remove_effect(clip_id: str, effect_id: str):
    """Remove effect from a clip."""
    editor = get_video_editor()
    success = editor.remove_effect(clip_id, effect_id)
    return {"success": success}

@app.get("/api/editor/preview/{time}")
async def editor_preview_frame(time: float, width: int = 640, height: int = 360):
    """Get a preview frame at specified time."""
    editor = get_video_editor()
    frame_bytes = editor.get_preview_frame(time, width, height)
    if frame_bytes is None or len(frame_bytes) == 0:
        raise HTTPException(status_code=404, detail="Failed to render frame")

    # Convert bytes to base64
    b64 = base64.b64encode(frame_bytes).decode()
    return {"success": True, "frame": f"data:image/png;base64,{b64}"}

@app.post("/api/editor/export")
async def editor_export(output_name: str, format: str = "mp4", quality: str = "high"):
    """Export the project to a video file."""
    editor = get_video_editor()
    exports_dir = Path("editor_exports")
    exports_dir.mkdir(exist_ok=True)
    output_path = str(exports_dir / f"{output_name}.{format}")

    # Run export in background
    import threading
    def do_export():
        editor.export_video(output_path, format, quality)
    thread = threading.Thread(target=do_export)
    thread.start()

    return {"success": True, "path": output_path, "status": "started"}

@app.get("/api/editor/export/progress")
async def editor_export_progress():
    """Get export progress."""
    editor = get_video_editor()
    return {
        "is_exporting": editor.is_exporting,
        "progress": editor.export_progress
    }

@app.post("/api/editor/export/cancel")
async def editor_cancel_export():
    """Cancel ongoing export."""
    editor = get_video_editor()
    editor.cancel_export()
    return {"success": True}

@app.post("/api/editor/import")
async def editor_import_media(file_path: str):
    """Import a media file and get its metadata."""
    editor = get_video_editor()
    try:
        media = editor.import_media(file_path)
        return {"success": True, "media": media_to_dict(media)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/editor/upload")
async def editor_upload_media(file: UploadFile = File(...)):
    """Upload a media file for editing."""
    import subprocess

    # Create uploads directory
    uploads_dir = Path("editor_uploads")
    uploads_dir.mkdir(exist_ok=True)

    # Save file
    file_path = uploads_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Get metadata using ffprobe
    metadata = {"duration": 5.0}  # Default duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_format", "-show_streams", "-print_format", "json", str(file_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            import json as json_module
            probe_data = json_module.loads(result.stdout)
            if "format" in probe_data and "duration" in probe_data["format"]:
                metadata["duration"] = float(probe_data["format"]["duration"])
            if "streams" in probe_data:
                for stream in probe_data["streams"]:
                    if stream.get("codec_type") == "video":
                        metadata["width"] = stream.get("width")
                        metadata["height"] = stream.get("height")
                        metadata["fps"] = eval(stream.get("r_frame_rate", "30/1"))
                    elif stream.get("codec_type") == "audio":
                        metadata["sample_rate"] = stream.get("sample_rate")
                        metadata["channels"] = stream.get("channels")
    except Exception as e:
        print(f"Failed to get media metadata: {e}")

    return {
        "success": True,
        "file_path": str(file_path.absolute()),
        "metadata": metadata
    }

@app.get("/api/editor/effects")
async def editor_get_effects():
    """Get available effects."""
    return {
        "effects": [
            # Color
            {"type": "brightness", "name": "Brightness", "category": "color", "params": {"value": {"type": "float", "default": 0, "min": -1, "max": 1}}},
            {"type": "contrast", "name": "Contrast", "category": "color", "params": {"value": {"type": "float", "default": 1, "min": 0, "max": 3}}},
            {"type": "saturation", "name": "Saturation", "category": "color", "params": {"value": {"type": "float", "default": 1, "min": 0, "max": 3}}},
            {"type": "hue", "name": "Hue Shift", "category": "color", "params": {"value": {"type": "float", "default": 0, "min": -180, "max": 180}}},
            {"type": "gamma", "name": "Gamma", "category": "color", "params": {"value": {"type": "float", "default": 1, "min": 0.1, "max": 3}}},
            # Stylize
            {"type": "blur", "name": "Gaussian Blur", "category": "stylize", "params": {"sigma": {"type": "float", "default": 5, "min": 0, "max": 50}}},
            {"type": "sharpen", "name": "Sharpen", "category": "stylize", "params": {"amount": {"type": "float", "default": 1, "min": 0, "max": 5}}},
            {"type": "denoise", "name": "Denoise", "category": "stylize", "params": {"strength": {"type": "float", "default": 4, "min": 0, "max": 10}}},
            {"type": "glow", "name": "Glow", "category": "stylize", "params": {"amount": {"type": "float", "default": 0.5, "min": 0, "max": 1}}},
            {"type": "vignette", "name": "Vignette", "category": "stylize", "params": {"amount": {"type": "float", "default": 0.5, "min": 0, "max": 1}}},
            # Utility
            {"type": "speed", "name": "Speed", "category": "utility", "params": {"rate": {"type": "float", "default": 1, "min": 0.1, "max": 4}}},
            {"type": "reverse", "name": "Reverse", "category": "utility", "params": {}},
            {"type": "chromakey", "name": "Chroma Key", "category": "utility", "params": {"color": {"type": "color", "default": "0x00FF00"}, "similarity": {"type": "float", "default": 0.3, "min": 0, "max": 1}, "blend": {"type": "float", "default": 0.1, "min": 0, "max": 1}}},
            {"type": "opacity", "name": "Opacity", "category": "utility", "params": {"value": {"type": "float", "default": 1, "min": 0, "max": 1}}},
            {"type": "flip_h", "name": "Flip Horizontal", "category": "utility", "params": {}},
            {"type": "flip_v", "name": "Flip Vertical", "category": "utility", "params": {}},
        ]
    }

@app.get("/api/editor/transitions")
async def editor_get_transitions():
    """Get available transitions."""
    return {
        "transitions": [
            {"type": "none", "name": "None"},
            {"type": "fade", "name": "Fade"},
            {"type": "fadeblack", "name": "Fade to Black"},
            {"type": "fadewhite", "name": "Fade to White"},
            {"type": "dissolve", "name": "Dissolve"},
            {"type": "wipeleft", "name": "Wipe Left"},
            {"type": "wiperight", "name": "Wipe Right"},
            {"type": "wipeup", "name": "Wipe Up"},
            {"type": "wipedown", "name": "Wipe Down"},
            {"type": "slideleft", "name": "Slide Left"},
            {"type": "slideright", "name": "Slide Right"},
            {"type": "circleopen", "name": "Circle Open"},
            {"type": "circleclose", "name": "Circle Close"},
        ]
    }


# ============================================================================
# VidPrep - Dataset Video Preparation
# ============================================================================

class VidPrepScanRequest(BaseModel):
    folder: str

class CropRegion(BaseModel):
    x: float
    y: float
    width: float
    height: float

class VideoRangeRequest(BaseModel):
    id: str
    start: float
    end: float
    caption: str
    crop: Optional[CropRegion] = None

class VidPrepSettings(BaseModel):
    target_fps: int = 16
    target_width: int = 640
    target_height: int = 360
    target_frames: int = 37
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    export_cropped: bool = True
    export_uncropped: bool = False
    export_first_frame: bool = False
    max_longest_edge: Optional[int] = None

class VidPrepProcessRequest(BaseModel):
    input_folder: str
    output_folder: str
    video_path: str
    ranges: List[VideoRangeRequest]
    settings: VidPrepSettings

@app.post("/api/vidprep/scan")
async def vidprep_scan_folder(request: VidPrepScanRequest):
    """Scan folder for video files and return metadata."""
    import subprocess

    folder = Path(request.folder)
    if not folder.exists():
        return {"videos": [], "error": "Folder not found"}

    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}
    videos = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in video_extensions:
            try:
                # Get video metadata using ffprobe
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_format", "-show_streams",
                     "-print_format", "json", str(file_path)],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)

                    duration = 0.0
                    width = 0
                    height = 0
                    fps = 30.0

                    if "format" in probe_data:
                        duration = float(probe_data["format"].get("duration", 0))

                    if "streams" in probe_data:
                        for stream in probe_data["streams"]:
                            if stream.get("codec_type") == "video":
                                width = stream.get("width", 0)
                                height = stream.get("height", 0)
                                # Parse frame rate
                                fps_str = stream.get("r_frame_rate", "30/1")
                                if "/" in fps_str:
                                    num, den = fps_str.split("/")
                                    fps = float(num) / float(den) if float(den) > 0 else 30.0
                                else:
                                    fps = float(fps_str)
                                break

                    videos.append({
                        "name": file_path.name,
                        "path": str(file_path.absolute()),
                        "duration": duration,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "frames": int(duration * fps),
                        "size": file_path.stat().st_size,
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return {"videos": videos}

@app.get("/api/vidprep/video")
async def vidprep_get_video(path: str):
    """Stream video file for preview."""
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )

@app.post("/api/vidprep/process")
async def vidprep_process_videos(request: VidPrepProcessRequest):
    """Process video ranges and export clips."""
    import subprocess

    output_dir = Path(request.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(request.video_path)
    if not video_path.exists():
        return {"success": False, "error": "Source video not found", "results": []}

    results = []
    settings = request.settings

    for i, range_req in enumerate(request.ranges):
        range_id = range_req.id
        start = range_req.start
        end = range_req.end
        duration = end - start

        # Generate output filename
        base_name = video_path.stem
        output_name = f"{base_name}_clip{i+1:03d}"

        try:
            # Build ffmpeg command
            filters = []

            # Crop if specified
            if range_req.crop:
                c = range_req.crop
                filters.append(f"crop={int(c.width)}:{int(c.height)}:{int(c.x)}:{int(c.y)}")

            # Scale to target resolution
            if settings.max_longest_edge:
                # Scale maintaining aspect ratio with max edge
                filters.append(
                    f"scale='if(gt(iw,ih),min({settings.max_longest_edge},iw),-2):"
                    f"if(gt(ih,iw),min({settings.max_longest_edge},ih),-2)'"
                )
            else:
                filters.append(f"scale={settings.target_width}:{settings.target_height}")

            # FPS conversion
            filters.append(f"fps={settings.target_fps}")

            filter_str = ",".join(filters)

            # Export cropped clip
            if settings.export_cropped:
                output_path = output_dir / f"{output_name}.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-t", str(duration),
                    "-vf", filter_str,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-an",  # No audio for training
                    str(output_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    results.append({
                        "range_id": range_id,
                        "success": False,
                        "error": result.stderr[:500]
                    })
                    continue

            # Export uncropped clip (time-aligned, no spatial crop)
            if settings.export_uncropped:
                uncropped_filters = [
                    f"scale={settings.target_width}:{settings.target_height}",
                    f"fps={settings.target_fps}"
                ]
                output_path_uncropped = output_dir / f"{output_name}_uncropped.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-t", str(duration),
                    "-vf", ",".join(uncropped_filters),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-an",
                    str(output_path_uncropped)
                ]
                subprocess.run(cmd, capture_output=True, timeout=300)

            # Export first frame as image
            if settings.export_first_frame:
                frame_path = output_dir / f"{output_name}_frame.png"
                frame_filters = []
                if range_req.crop:
                    c = range_req.crop
                    frame_filters.append(f"crop={int(c.width)}:{int(c.height)}:{int(c.x)}:{int(c.y)}")
                frame_filters.append(f"scale={settings.target_width}:{settings.target_height}")

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-vf", ",".join(frame_filters),
                    "-frames:v", "1",
                    str(frame_path)
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)

            # Write caption file
            if range_req.caption:
                caption_path = output_dir / f"{output_name}.txt"
                with open(caption_path, "w") as f:
                    f.write(range_req.caption)

            results.append({
                "range_id": range_id,
                "success": True,
                "output_path": str(output_dir / f"{output_name}.mp4")
            })

        except subprocess.TimeoutExpired:
            results.append({
                "range_id": range_id,
                "success": False,
                "error": "Processing timeout"
            })
        except Exception as e:
            results.append({
                "range_id": range_id,
                "success": False,
                "error": str(e)
            })

    return {"success": True, "results": results}


# ============================================================================
# WebSocket for real-time updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time progress updates."""
    await websocket.accept()
    engine.websockets.add(websocket)

    try:
        while True:
            # Send status updates periodically
            status = engine.get_status()
            await websocket.send_json(status.model_dump())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        engine.websockets.discard(websocket)


# ============================================================================
# Static files (frontend)
# ============================================================================

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # API routes are handled above
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"error": "Frontend not built"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
