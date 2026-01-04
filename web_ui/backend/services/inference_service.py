"""
Inference Service for generating images with trained models.

Uses OneTrainer's sampler infrastructure (BaseModelSampler, SampleConfig)
to generate images with loaded models.

Provides functionality for:
- Loading base models with optional LoRA/adapters
- Generating images using the sampler pattern
- Managing inference state and gallery
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from safetensors.torch import load_file as load_safetensors
import os

# OneTrainer imports
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.torch_util import torch_gc
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput


@dataclass
class GenerationRequest:
    """Parameters for image generation - maps to SampleConfig."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 7.0
    seed: int = -1  # -1 for random
    random_seed: bool = True
    noise_scheduler: str = "DDIM"
    batch_size: int = 1
    # Generation mode
    mode: str = "txt2img"  # txt2img, img2img, inpainting, edit, video
    # img2img / inpainting inputs
    init_image_path: str = ""
    mask_image_path: str = ""
    strength: float = 0.75  # For img2img denoising strength
    # Video generation
    num_frames: int = 16
    fps: int = 8
    # Edit mode (for Z-Image-Edit, Qwen-Edit)
    edit_instruction: str = ""
    # Multi-image input (for FLUX 2)
    reference_images: list = None

    
    def to_sample_config(self) -> SampleConfig:
        """Convert to OneTrainer SampleConfig."""
        config = SampleConfig.default_values()
        config.enabled = True
        config.prompt = self.prompt
        config.negative_prompt = self.negative_prompt
        config.width = self.width
        config.height = self.height
        config.seed = self.seed if self.seed >= 0 else 42
        config.random_seed = self.seed < 0 or self.random_seed
        config.diffusion_steps = self.steps
        config.cfg_scale = self.guidance_scale
        
        # Map noise scheduler string to enum
        try:
            config.noise_scheduler = NoiseScheduler[self.noise_scheduler]
        except KeyError:
            config.noise_scheduler = NoiseScheduler.DDIM
        
        return config


@dataclass
class GeneratedImage:
    """Represents a generated image."""
    id: str
    path: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "created_at": self.created_at.isoformat(),
        }


class InferenceService:
    """
    Singleton service for managing inference operations.
    
    Uses OneTrainer's sampler infrastructure for image generation.
    Handles model loading, image generation, and gallery management.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self._state_lock = threading.Lock()
        
        # Model state
        self._model_loaded = False
        self._model_path: Optional[str] = None
        self._model_type: Optional[str] = None
        self._lora_paths: List[str] = []
        
        # Loaded components (populated when model is loaded)
        self._model = None  # Loaded OneTrainer model
        self._sampler: Optional[BaseModelSampler] = None
        self._train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._temp_device = torch.device("cpu")
        
        # Generation state
        self._is_generating = False
        self._generation_progress = 0
        self._should_cancel = False
        
        # Gallery
        self._gallery: List[GeneratedImage] = []
        self._output_dir = Path("workspace/inference_outputs")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # WebSocket connections for updates
        self._websockets = set()
    
    def register_websocket(self, websocket):
        """Register a WebSocket connection for updates."""
        self._websockets.add(websocket)
    
    def unregister_websocket(self, websocket):
        """Unregister a WebSocket connection."""
        self._websockets.discard(websocket)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current inference state."""
        with self._state_lock:
            return {
                "model_loaded": self._model_loaded,
                "model_path": self._model_path,
                "model_type": self._model_type,
                "lora_paths": self._lora_paths.copy(),
                "is_generating": self._is_generating,
                "generation_progress": self._generation_progress,
            }
    
    def _get_sampler_class(self, model_type: str):
        """
        Get the appropriate sampler class for the given model type.
        
        Each model type has its own sampler implementation that knows
        how to generate images with that specific architecture.
        """
        from modules.util.enum.ModelType import ModelType
        
        try:
            mt = ModelType[model_type]
        except KeyError:
            return None
        
        # Map model types to their samplers
        if mt.is_flux():
            from modules.modelSampler.FluxSampler import FluxSampler
            return FluxSampler
        elif mt.is_stable_diffusion_xl():
            from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
            return StableDiffusionXLSampler
        elif mt.is_stable_diffusion_3():
            from modules.modelSampler.StableDiffusion3Sampler import StableDiffusion3Sampler
            return StableDiffusion3Sampler
        elif mt.is_stable_diffusion():
            from modules.modelSampler.StableDiffusionSampler import StableDiffusionSampler
            return StableDiffusionSampler
        elif mt.is_pixart():
            from modules.modelSampler.PixArtAlphaSampler import PixArtAlphaSampler
            return PixArtAlphaSampler
        elif mt.is_sana():
            from modules.modelSampler.SanaSampler import SanaSampler
            return SanaSampler
        elif mt.is_chroma():
            from modules.modelSampler.ChromaSampler import ChromaSampler
            return ChromaSampler
        elif mt.is_wuerstchen():
            from modules.modelSampler.WuerstchenSampler import WuerstchenSampler
            return WuerstchenSampler
        elif mt.is_hunyuan_video():
            from modules.modelSampler.HunyuanVideoSampler import HunyuanVideoSampler
            return HunyuanVideoSampler
        elif mt.is_hi_dream():
            from modules.modelSampler.HiDreamSampler import HiDreamSampler
            return HiDreamSampler
        elif mt.is_qwen():
            from modules.modelSampler.QwenSampler import QwenSampler
            return QwenSampler
        elif mt.is_z_image():
            from modules.modelSampler.ZImageSampler import ZImageSampler
            return ZImageSampler
        else:
            return None
    
    def _get_pipeline_class(self, model_type: str, mode: str = "txt2img"):
        """
        Get the appropriate diffusers pipeline class for the model type and generation mode.

        Args:
            model_type: The model type string
            mode: Generation mode (txt2img, img2img, inpainting, edit, video)
        """
        # Handle extended model types that aren't in OneTrainer's ModelType enum
        extended_types = {
            'FLUX_SCHNELL': 'flux',
            'FLUX_2_DEV': 'flux2',
            'Z_IMAGE_TURBO': 'z_image',
            'Z_IMAGE_EDIT': 'z_image_edit',
            'QWEN_IMAGE': 'qwen_image',
            'QWEN_IMAGE_EDIT': 'qwen_image_edit',
            'QWEN_IMAGE_LAYERED': 'qwen_image_layered',
        }

        if model_type in extended_types:
            extended = extended_types[model_type]

            if extended == 'flux2':
                try:
                    from diffusers import Flux2Pipeline
                    return Flux2Pipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif extended == 'z_image_edit':
                # Z-Image-Edit for instruction-based editing
                try:
                    from diffusers import ZImageEditPipeline
                    return ZImageEditPipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif extended == 'qwen_image':
                # Use DiffusionPipeline to support from_single_file
                from diffusers import DiffusionPipeline
                return DiffusionPipeline
            elif extended == 'qwen_image_edit':
                try:
                    from diffusers import QwenImageEditPipeline
                    return QwenImageEditPipeline
                except ImportError:
                    return None
            elif extended == 'qwen_image_layered':
                try:
                    from diffusers import QwenImageLayeredPipeline
                    return QwenImageLayeredPipeline
                except ImportError:
                    return None
            elif extended == 'z_image':
                try:
                    from diffusers import ZImagePipeline
                    return ZImagePipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif extended == 'flux':
                from diffusers import FluxPipeline
                return FluxPipeline

        # Try to use OneTrainer's ModelType enum for known types
        try:
            from modules.util.enum.ModelType import ModelType
            mt = ModelType[model_type]
        except (KeyError, ImportError):
            mt = None

        if mt is None:
            return None

        # Select pipeline based on model type and generation mode
        if mt.is_chroma():
            from diffusers import ChromaPipeline
            return ChromaPipeline
        elif mt.is_flux() or mt.is_z_image():
            if mode == "inpainting" or model_type == "FLUX_FILL_DEV_1":
                try:
                    from diffusers import FluxFillPipeline
                    return FluxFillPipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            elif mode == "img2img":
                try:
                    from diffusers import FluxImg2ImgPipeline
                    return FluxImg2ImgPipeline
                except ImportError:
                    from diffusers import FluxPipeline
                    return FluxPipeline
            else:
                from diffusers import FluxPipeline
                return FluxPipeline
        elif mt.is_stable_diffusion_xl():
            if mode == "inpainting":
                from diffusers import StableDiffusionXLInpaintPipeline
                return StableDiffusionXLInpaintPipeline
            elif mode == "img2img":
                from diffusers import StableDiffusionXLImg2ImgPipeline
                return StableDiffusionXLImg2ImgPipeline
            else:
                from diffusers import StableDiffusionXLPipeline
                return StableDiffusionXLPipeline
        elif mt.is_stable_diffusion_3():
            if mode == "img2img":
                try:
                    from diffusers import StableDiffusion3Img2ImgPipeline
                    return StableDiffusion3Img2ImgPipeline
                except ImportError:
                    from diffusers import StableDiffusion3Pipeline
                    return StableDiffusion3Pipeline
            else:
                from diffusers import StableDiffusion3Pipeline
                return StableDiffusion3Pipeline
        elif mt.is_stable_diffusion():
            if mode == "inpainting":
                from diffusers import StableDiffusionInpaintPipeline
                return StableDiffusionInpaintPipeline
            elif mode == "img2img":
                from diffusers import StableDiffusionImg2ImgPipeline
                return StableDiffusionImg2ImgPipeline
            else:
                from diffusers import StableDiffusionPipeline
                return StableDiffusionPipeline
        elif mt.is_pixart():
            from diffusers import PixArtAlphaPipeline
            return PixArtAlphaPipeline
        elif mt.is_sana():
            try:
                from diffusers import SanaPipeline
                return SanaPipeline
            except ImportError:
                return None
        elif mt.is_hunyuan_video():
            try:
                from diffusers import HunyuanVideoPipeline
                return HunyuanVideoPipeline
            except ImportError:
                return None
        elif mt.is_wan():
            try:
                from diffusers import WanPipeline
                return WanPipeline
            except ImportError:
                return None
        elif mt.is_kandinsky_5():
            # Kandinsky 5 uses custom pipeline from local repo, not diffusers
            # Return None to trigger special loading path
            return None
        else:
            return None

    def _get_hf_model_id(self, model_type: str) -> Optional[str]:
        """Get HuggingFace model ID for fallback loading."""
        hf_models = {
            # Image models
            'QWEN_IMAGE': 'Qwen/Qwen-Image',
            'QWEN_IMAGE_EDIT': 'Qwen/Qwen-Image-Edit',
            'Z_IMAGE': 'Tongyi-MAI/Z-Image',
            'Z_IMAGE_TURBO': 'Tongyi-MAI/Z-Image-Turbo',
            'FLUX_DEV': 'black-forest-labs/FLUX.1-dev',
            'FLUX_SCHNELL': 'black-forest-labs/FLUX.1-schnell',
            'SD_35': 'stabilityai/stable-diffusion-3.5-large',
            'SD_35_TURBO': 'stabilityai/stable-diffusion-3.5-large-turbo',
            'SD_3': 'stabilityai/stable-diffusion-3-medium-diffusers',
            'SDXL': 'stabilityai/stable-diffusion-xl-base-1.0',
            'SD_15': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
            'SD_21': 'stabilityai/stable-diffusion-2-1',
            'PIXART_ALPHA': 'PixArt-alpha/PixArt-XL-2-1024-MS',
            'PIXART_SIGMA': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
            'SANA': 'Efficient-Large-Model/Sana_1600M_1024px_diffusers',
            'CHROMA': 'lodestone-horizon/chroma-v2',
            'LUMINA': 'Alpha-VLLM/Lumina-Next-T2I',
            'LUMINA_2': 'Alpha-VLLM/Lumina-Image-2.0',
            # Video models
            'WAN_T2V': 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
            'WAN_I2V': 'Wan-AI/Wan2.1-I2V-14B-Diffusers',
            'WAN_T2V_HIGH': 'Wan-AI/Wan2.2-T2V-14B-Diffusers',
            'WAN_I2V_HIGH': 'Wan-AI/Wan2.2-I2V-14B-Diffusers',
            'HUNYUAN_VIDEO': 'hunyuanvideo-community/HunyuanVideo',
        }
        return hf_models.get(model_type)

    
    def load_model(
        self,
        model_path: str,
        model_type: str,
        lora_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Load a base model with optional LoRA adapters using diffusers.
        
        Args:
            model_path: Path to base model or checkpoint
            model_type: Model type (e.g., "FLUX_DEV_1", "STABLE_DIFFUSION_XL_10_BASE")
            lora_paths: Optional list of LoRA paths to apply
            
        Returns:
            Dict with success status and message
        """
        try:
            # Check if model_path is a local path or HuggingFace model ID
            is_local_path = Path(model_path).exists()
            is_hf_model_id = '/' in model_path and not is_local_path  # HF format: org/model
            
            if not is_local_path and not is_hf_model_id:
                return {"success": False, "error": f"Model path not found: {model_path}"}
            
            # Validate LoRA paths (these must be local)
            for lora_path in (lora_paths or []):
                if not Path(lora_path).exists():
                    return {"success": False, "error": f"LoRA path not found: {lora_path}"}

            
            # Get pipeline class
            pipeline_class = self._get_pipeline_class(model_type)

            # Special handling for Kandinsky 5 models
            if 'kandinsky_5' in model_type.lower() or 'KANDINSKY_5' in model_type:
                result = self._load_kandinsky5(model_path, model_type)
                if result.get("success"):
                    return result
                # If Kandinsky loading failed, log and continue to placeholder
                print(f"Kandinsky 5 loading failed: {result.get('error')}")

            if pipeline_class is None:
                # Fall back to placeholder mode for unsupported types
                with self._state_lock:
                    if self._model is not None:
                        self._unload_model_internal()
                    self._model_path = model_path
                    self._model_type = model_type
                    self._lora_paths = lora_paths or []
                    self._model_loaded = True
                return {
                    "success": True,
                    "message": f"Model loaded (placeholder mode): {Path(model_path).name}",
                    "model_type": model_type,
                }
            
            with self._state_lock:
                # Unload previous model if loaded
                if self._model is not None:
                    self._unload_model_internal()
            
            # Load the pipeline
            print(f"Loading {model_type} from {model_path}...")
            
            # Determine dtype based on available GPU
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            # Load from local path or HuggingFace
            # Use device_map for automatic memory management on large models
            if Path(model_path).is_file():
                # Single file checkpoint (safetensors, etc)
                print(f"Loading single file checkpoint: {model_path}")
                
                extra_kwargs = {}
                
                # For Flux models ONLY, pre-load CLIP+T5 text encoders from local files
                if "FLUX" in model_type and "Z_IMAGE" not in model_type and "QWEN" not in model_type:
                    print("Pre-loading text encoders for Flux from local files...")
                    clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors"
                    t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors"
                    
                    if not os.path.exists(t5_path):
                        t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors"
                    
                    text_encoder_1 = None
                    text_encoder_2 = None
                    tokenizer_1 = None
                    tokenizer_2 = None
                    
                    if os.path.exists(clip_path):
                        try:
                            print(f"Loading CLIP L from {clip_path}")
                            from transformers import CLIPTextModel, CLIPTokenizer, CLIPConfig
                            clip_config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
                            text_encoder_1 = CLIPTextModel(clip_config.text_config)
                            sd = load_safetensors(clip_path)
                            text_encoder_1.load_state_dict(sd, strict=False)
                            text_encoder_1 = text_encoder_1.to(dtype)
                            tokenizer_1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
                            print("CLIP L loaded successfully")
                        except Exception as e:
                            print(f"Warning: Could not load CLIP L: {e}")
                    
                    if os.path.exists(t5_path):
                        try:
                            print(f"Loading T5 XXL from {t5_path}")
                            from transformers import T5EncoderModel, T5Tokenizer, T5Config
                            t5_config = T5Config.from_pretrained("google/t5-v1_1-xxl", local_files_only=True)
                            text_encoder_2 = T5EncoderModel(t5_config)
                            sd = load_safetensors(t5_path)
                            text_encoder_2.load_state_dict(sd, strict=False)
                            text_encoder_2 = text_encoder_2.to(dtype)
                            tokenizer_2 = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", local_files_only=True)
                            print("T5 XXL loaded successfully")
                        except Exception as e:
                            print(f"Warning: Could not load T5: {e}")
                    
                    # Pre-load VAE for Flux from HF cache
                    vae = None
                    try:
                        print("Loading VAE from cached HuggingFace model...")
                        from diffusers import AutoencoderKL
                        vae = AutoencoderKL.from_pretrained(
                            "black-forest-labs/FLUX.1-dev",
                            subfolder="vae",
                            torch_dtype=dtype,
                            local_files_only=True,
                        )
                        print("VAE loaded successfully")
                    except Exception as e:
                        print(f"Warning: Could not load VAE: {e}")
                    
                    # Build extra kwargs for Flux
                    if text_encoder_1 is not None:
                        extra_kwargs["text_encoder"] = text_encoder_1
                    if text_encoder_2 is not None:
                        extra_kwargs["text_encoder_2"] = text_encoder_2
                    if tokenizer_1 is not None:
                        extra_kwargs["tokenizer"] = tokenizer_1
                    if tokenizer_2 is not None:
                        extra_kwargs["tokenizer_2"] = tokenizer_2
                    if vae is not None:
                        extra_kwargs["vae"] = vae
                
                # For Z-Image and Qwen models, try from_pretrained with HuggingFace cache
                # These models use Qwen text encoder which is auto-loaded from HF
                elif "Z_IMAGE" in model_type or "QWEN" in model_type:
                    print(f"Loading {model_type} - will use HuggingFace cache for text encoder...")
                    # These models don't support from_single_file properly
                    # Go directly to from_pretrained
                    hf_model_id = self._get_hf_model_id(model_type)
                    if hf_model_id:
                        try:
                            pipeline = pipeline_class.from_pretrained(
                                hf_model_id,
                                torch_dtype=dtype,
                                local_files_only=True,  # Try cache first
                            )
                            print(f"Loaded {model_type} from cache: {hf_model_id}")
                        except Exception as e:
                            print(f"Cache load failed: {e}, downloading...")
                            pipeline = pipeline_class.from_pretrained(
                                hf_model_id,
                                torch_dtype=dtype,
                            )
                            print(f"Downloaded {model_type} from HuggingFace: {hf_model_id}")
                    else:
                        raise Exception(f"No HuggingFace model ID for {model_type}")
                
                # For FLUX models with pre-loaded encoders
                elif "FLUX" in model_type:
                    # Check if from_single_file is available
                    if hasattr(pipeline_class, 'from_single_file'):
                        try:
                            pipeline = pipeline_class.from_single_file(
                                model_path,
                                torch_dtype=dtype,
                                **extra_kwargs,
                            )
                            print(f"Loaded {model_type} pipeline successfully")
                        except Exception as e:
                            print(f"from_single_file failed: {e}, trying from_pretrained fallback...")
                            # Try HuggingFace model ID as fallback
                            hf_fallback = self._get_hf_model_id(model_type)
                            if hf_fallback:
                                pipeline = pipeline_class.from_pretrained(
                                    hf_fallback,
                                    torch_dtype=dtype,
                                )
                                print(f"Loaded {model_type} from HuggingFace: {hf_fallback}")
                            else:
                                raise
                    else:
                        # No from_single_file - use from_pretrained
                        hf_model_id = self._get_hf_model_id(model_type)
                        if hf_model_id:
                            print(f"Loading {model_type} from HuggingFace: {hf_model_id}")
                            pipeline = pipeline_class.from_pretrained(
                                hf_model_id,
                                torch_dtype=dtype,
                            )
                        else:
                            raise Exception(f"No from_single_file support and no HuggingFace fallback for {model_type}")
                
                # Non-Flux/Z-Image/Qwen models - check for from_single_file support
                else:
                    if hasattr(pipeline_class, 'from_single_file'):
                        try:
                            pipeline = pipeline_class.from_single_file(
                                model_path,
                                torch_dtype=dtype,
                            )
                        except Exception as e:
                            print(f"from_single_file failed: {e}, trying from_pretrained...")
                            hf_fallback = self._get_hf_model_id(model_type)
                            if hf_fallback:
                                pipeline = pipeline_class.from_pretrained(
                                    hf_fallback,
                                    torch_dtype=dtype,
                                )
                            else:
                                raise
                    else:
                        # No from_single_file - use from_pretrained
                        hf_model_id = self._get_hf_model_id(model_type)
                        if hf_model_id:
                            print(f"Loading {model_type} from HuggingFace: {hf_model_id}")
                            pipeline = pipeline_class.from_pretrained(
                                hf_model_id,
                                torch_dtype=dtype,
                            )
                        else:
                            raise Exception(f"No from_single_file support and no HuggingFace fallback for {model_type}")
            else:
                # Directory or HF repo ID
                pipeline = pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=Path(model_path).exists(),
                )

            # Enable CPU offload for memory efficiency on large models
            # Use sequential offload for large models (more aggressive memory management)
            try:
                if "FLUX" in model_type or "Z_IMAGE" in model_type or "QWEN_IMAGE_2512" in model_type:
                    pipeline.enable_sequential_cpu_offload()
                    print(f"Enabled SEQUENTIAL CPU offload for {model_type}")
                else:
                    pipeline.enable_model_cpu_offload()
                    print(f"Enabled CPU offload for {model_type}")
            except Exception as e:
                print(f"CPU offload not available, moving to GPU: {e}")
                pipeline = pipeline.to(self._train_device)
            
            # Apply LoRAs if provided
            if lora_paths:
                for lora_path in lora_paths:
                    try:
                        lora_name = Path(lora_path).stem
                        pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
                        print(f"Loaded LoRA: {lora_name}")
                    except Exception as e:
                        print(f"Warning: Failed to load LoRA {lora_path}: {e}")
            
            with self._state_lock:
                self._model = pipeline
                self._model_path = model_path
                self._model_type = model_type
                self._lora_paths = lora_paths or []
                self._model_loaded = True
            
            return {
                "success": True,
                "message": f"Model loaded: {Path(model_path).name}",
                "model_type": model_type,
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _unload_model_internal(self):
        """Internal method to unload model (must hold lock)."""
        # Cleanup sampler
        if self._sampler is not None:
            del self._sampler
            self._sampler = None
        
        # Cleanup model
        if self._model is not None:
            del self._model
            self._model = None
        
        self._model_loaded = False
        self._model_path = None
        self._model_type = None
        self._lora_paths = []
        
        # Clear CUDA cache
        torch_gc()
    
    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model and free memory."""
        with self._state_lock:
            if not self._model_loaded:
                return {"success": False, "error": "No model loaded"}
            
            self._unload_model_internal()
        
        return {"success": True, "message": "Model unloaded"}

    def _load_kandinsky5(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load Kandinsky 5 model using local repo pipelines."""
        import sys

        hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
        kandinsky_repo = Path("/home/alex/OneTrainer/models/kandinsky-5-code")

        # Add Kandinsky repo to path
        if kandinsky_repo.exists() and str(kandinsky_repo) not in sys.path:
            sys.path.insert(0, str(kandinsky_repo))

        # Determine if this is a video model
        is_video = 'video' in model_type.lower()

        print(f"Loading Kandinsky 5 {'Video' if is_video else 'Image'} from {model_path}...")

        # Check if model_path is local file or needs to use default weights
        if not Path(model_path).exists():
            # Check for local weights
            if is_video:
                local_weights = Path("/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors")
            else:
                local_weights = Path("/home/alex/OneTrainer/models/kandinsky-5-code")
            if local_weights.exists():
                model_path = str(local_weights)
                print(f"Using local weights: {model_path}")

        # Detect model variant (Pro vs Lite) based on filename
        is_pro = "pro" in model_path.lower()
        variant = "Pro" if is_pro else "Lite"
        print(f"Detected Kandinsky 5 {variant} model")

        # Get appropriate config file
        config_dir = kandinsky_repo / "configs"
        if is_video:
            if is_pro:
                config_file = config_dir / "k5_pro_t2v_5s_sft_sd.yaml"
            else:
                config_file = config_dir / "k5_lite_t2v_5s_sft_sd.yaml"
        else:
            if is_pro:
                config_file = None
            else:
                config_file = config_dir / "k5_lite_t2i_sft_hd.yaml"

        try:
            from kandinsky.utils import get_T2V_pipeline, get_T2I_pipeline

            # Device map as dict (like standalone app)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = {"dit": device, "vae": device, "text_embedder": device}

            dit_path = model_path if Path(model_path).exists() else None

            # Get config file for architecture match
            conf_path_str = str(config_file) if config_file and config_file.exists() else None

            print(f"Loading Kandinsky 5 {variant} pipeline...")
            print(f"  dit_path: {dit_path}")
            print(f"  conf_path: {conf_path_str}")

            if is_video:
                pipe = get_T2V_pipeline(
                    device_map=device_map,
                    cache_dir=hf_cache,
                    dit_path=dit_path,
                    vae_path=None,  # Let it download
                    text_encoder_path="Qwen/Qwen2.5-VL-7B-Instruct",
                    text_encoder2_path="openai/clip-vit-large-patch14",
                    conf_path=conf_path_str,
                    offload=True,
                )
            else:
                pipe = get_T2I_pipeline(
                    device_map=device_map,
                    cache_dir=hf_cache,
                    dit_path=dit_path,
                    text_encoder_path="Qwen/Qwen2.5-VL-7B-Instruct",
                    text_encoder2_path="openai/clip-vit-large-patch14",
                    conf_path=conf_path_str,
                    offload=True,
                )

            print(f"✅ Kandinsky 5 {variant} {'T2V' if is_video else 'T2I'} pipeline loaded")

            with self._state_lock:
                if self._model is not None:
                    self._unload_model_internal()
                self._model = pipe
                self._model_path = model_path
                self._model_type = model_type
                self._lora_paths = []
                self._model_loaded = True

            return {
                "success": True,
                "message": f"Model loaded: Kandinsky 5 {variant} {'T2V' if is_video else 'T2I'}",
                "model_type": model_type,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Failed to load Kandinsky 5: {str(e)}"}

    def _on_update_progress(self, current: int, total: int):
        """Callback for sampler progress updates."""
        with self._state_lock:
            self._generation_progress = int((current / total) * 100) if total > 0 else 0
    
    def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate images based on the request using the loaded sampler.
        
        Args:
            request: Generation parameters
            
        Returns:
            Dict with generated image info or error
        """
        with self._state_lock:
            if not self._model_loaded:
                return {"success": False, "error": "No model loaded"}
            
            if self._is_generating:
                return {"success": False, "error": "Generation already in progress"}
            
            self._is_generating = True
            self._generation_progress = 0
            self._should_cancel = False
        
        try:
            # Generate unique ID and output path
            image_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{image_id}"
            output_path = self._output_dir / filename
            
            # Store the actual seed used
            import random
            actual_seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
            
            # Check if we have a real pipeline loaded
            if self._model is not None and hasattr(self._model, '__call__'):
                # Use diffusers pipeline for actual generation
                generator = torch.Generator(device=self._train_device).manual_seed(actual_seed)

                # Define progress callback
                def progress_callback(pipe, step, timestep, callback_kwargs):
                    self._on_update_progress(step + 1, request.steps)
                    if self._should_cancel:
                        raise InterruptedError("Generation cancelled")
                    return callback_kwargs

                # Run inference with pipeline based on mode
                try:
                    from PIL import Image as PILImage

                    # Special handling for Kandinsky 5 models
                    is_kandinsky = 'kandinsky_5' in (self._model_type or '').lower()
                    is_video_model = 'video' in (self._model_type or '').lower()

                    if is_kandinsky:
                        # Kandinsky 5 uses different parameter names
                        pipeline_kwargs = {
                            "text": request.prompt,
                            "num_steps": request.steps,
                            "guidance_weight": request.guidance_scale,
                            "seed": actual_seed,
                            "width": request.width,
                            "height": request.height,
                            "negative_caption": request.negative_prompt or "",
                            "expand_prompts": False,  # Disable - causes tensor mismatch
                            "progress": True,
                        }

                        # For video models, add time_length and scheduler_scale
                        if is_video_model or request.mode == "video":
                            # Calculate duration from frames: time_length = (num_frames - 1) * 4 / 24
                            # Kandinsky: num_frames = time_length * 24 // 4 + 1
                            # Default to 5 seconds
                            time_length = 5
                            if hasattr(request, 'num_frames') and request.num_frames > 0:
                                time_length = max(1, (request.num_frames - 1) * 4 // 24)
                            pipeline_kwargs["time_length"] = time_length
                            pipeline_kwargs["scheduler_scale"] = 10.0
                        else:
                            pipeline_kwargs["scheduler_scale"] = 3.0
                    else:
                        # Standard diffusers parameters
                        pipeline_kwargs = {
                            "prompt": request.prompt,
                            "num_inference_steps": request.steps,
                            "guidance_scale": request.guidance_scale,
                            "generator": generator,
                            "callback_on_step_end": progress_callback,
                        }

                        # Add negative prompt if supported (FLUX and Chroma don't use it)
                        if request.negative_prompt and self._model_type not in ['FLUX_DEV_1', 'FLUX_SCHNELL', 'FLUX_2_DEV', 'FLUX_FILL_DEV_1', 'CHROMA_1']:
                            pipeline_kwargs["negative_prompt"] = request.negative_prompt

                        # Qwen-Image-2512 uses true_cfg_scale instead of guidance_scale
                        if self._model_type == 'QWEN_IMAGE_2512':
                            pipeline_kwargs["true_cfg_scale"] = request.guidance_scale
                            # Remove standard guidance_scale as it may conflict
                            pipeline_kwargs.pop("guidance_scale", None)

                    # Mode-specific handling (non-Kandinsky only)
                    if not is_kandinsky:
                        if request.mode == "img2img" and request.init_image_path:
                            # Load init image
                            init_image = PILImage.open(request.init_image_path).convert("RGB")
                            init_image = init_image.resize((request.width, request.height))
                            pipeline_kwargs["image"] = init_image
                            pipeline_kwargs["strength"] = request.strength
                        elif request.mode == "inpainting" and request.init_image_path and request.mask_image_path:
                            # Load init and mask images
                            init_image = PILImage.open(request.init_image_path).convert("RGB")
                            mask_image = PILImage.open(request.mask_image_path).convert("L")
                            init_image = init_image.resize((request.width, request.height))
                            mask_image = mask_image.resize((request.width, request.height))
                            pipeline_kwargs["image"] = init_image
                            pipeline_kwargs["mask_image"] = mask_image
                            pipeline_kwargs["strength"] = request.strength
                        elif request.mode == "edit" and request.init_image_path:
                            # Edit mode for Z-Image-Edit, Qwen-Edit
                            init_image = PILImage.open(request.init_image_path).convert("RGB")
                            init_image = init_image.resize((request.width, request.height))
                            pipeline_kwargs["image"] = init_image
                            # Use edit instruction as prompt for edit models
                            if request.edit_instruction:
                                pipeline_kwargs["prompt"] = request.edit_instruction
                        elif request.mode == "video":
                            # Video generation mode
                            pipeline_kwargs["num_frames"] = request.num_frames
                            pipeline_kwargs["height"] = request.height
                            pipeline_kwargs["width"] = request.width
                        else:
                            # txt2img mode - add dimensions
                            pipeline_kwargs["width"] = request.width
                            pipeline_kwargs["height"] = request.height

                        # Handle multi-image input for FLUX 2
                        if request.reference_images and self._model_type == 'FLUX_2_DEV':
                            ref_images = []
                            for ref_path in request.reference_images:
                                if Path(ref_path).exists():
                                    ref_img = PILImage.open(ref_path).convert("RGB")
                                    ref_images.append(ref_img)
                            if ref_images:
                                pipeline_kwargs["image"] = ref_images

                    # Kandinsky video: pass save_path to save directly
                    if is_kandinsky and is_video_model:
                        video_path = str(output_path) + ".mp4"
                        pipeline_kwargs["save_path"] = video_path

                    result = self._model(**pipeline_kwargs)

                    # Handle output based on type
                    if is_kandinsky:
                        # Kandinsky returns PIL.Image for T2I or saves video directly
                        if is_video_model:
                            # Video was saved directly via save_path
                            final_path = video_path
                        elif isinstance(result, PILImage.Image):
                            final_path = str(output_path) + ".png"
                            result.save(final_path, "PNG")
                        else:
                            raise ValueError(f"Unexpected Kandinsky result type: {type(result)}")
                    elif request.mode == "video" and hasattr(result, 'frames'):
                        # Video output
                        from torchvision.io import write_video
                        video_path = str(output_path) + ".mp4"
                        # Frames are usually in format [batch, frames, channels, height, width]
                        frames = result.frames[0] if len(result.frames.shape) == 5 else result.frames
                        write_video(video_path, frames.permute(0, 2, 3, 1).cpu(), fps=request.fps)
                        final_path = video_path
                    elif hasattr(result, 'images') and result.images:
                        img = result.images[0]
                        final_path = str(output_path) + ".png"
                        img.save(final_path, "PNG")
                    else:
                        raise ValueError("No output generated")

                except InterruptedError:
                    with self._state_lock:
                        self._is_generating = False
                    return {"success": False, "error": "Generation cancelled"}
            
            elif self._sampler is not None:
                # Use OneTrainer sampler if available
                sample_config = request.to_sample_config()
                self._sampler.sample(
                    sample_config=sample_config,
                    destination=str(output_path),
                    image_format=ImageFormat.PNG,
                    video_format=None,
                    audio_format=None,
                    on_sample=lambda _: None,
                    on_update_progress=self._on_update_progress,
                )
                final_path = str(output_path) + ".png"
            else:
                # Placeholder: create a test image when no real model is available
                from PIL import Image
                
                # Simulate progress
                for i in range(request.steps):
                    if self._should_cancel:
                        with self._state_lock:
                            self._is_generating = False
                        return {"success": False, "error": "Generation cancelled"}
                    
                    self._on_update_progress(i + 1, request.steps)
                    import time
                    time.sleep(0.02)  # Small delay for demo
                
                # Create placeholder gradient image
                img = Image.new('RGB', (request.width, request.height))
                pixels = img.load()
                for y in range(request.height):
                    for x in range(request.width):
                        # Create a gradient based on seed
                        r = int((x / request.width) * 255) ^ (actual_seed & 0xFF)
                        g = int((y / request.height) * 255) ^ ((actual_seed >> 8) & 0xFF)
                        b = int(((x + y) / (request.width + request.height)) * 255) ^ ((actual_seed >> 16) & 0xFF)
                        pixels[x, y] = (r % 256, g % 256, b % 256)
                
                final_path = str(output_path) + ".png"
                img.save(final_path, "PNG")
            
            # Create generated image record
            generated = GeneratedImage(
                id=image_id,
                path=final_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=actual_seed,
            )
            
            with self._state_lock:
                self._gallery.insert(0, generated)
                self._is_generating = False
                self._generation_progress = 100
            
            return {
                "success": True,
                "image": generated.to_dict(),
            }
            
        except Exception as e:
            with self._state_lock:
                self._is_generating = False
            return {"success": False, "error": str(e)}
    
    def cancel_generation(self) -> Dict[str, Any]:
        """Cancel the current generation."""
        with self._state_lock:
            if not self._is_generating:
                return {"success": False, "error": "No generation in progress"}
            self._should_cancel = True
        return {"success": True, "message": "Cancellation requested"}
    
    def get_gallery(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get generated images gallery."""
        with self._state_lock:
            return [img.to_dict() for img in self._gallery[:limit]]
    
    def clear_gallery(self):
        """Clear the gallery."""
        with self._state_lock:
            # Delete files
            for img in self._gallery:
                try:
                    Path(img.path).unlink(missing_ok=True)
                except:
                    pass
            self._gallery = []
    
    def delete_image(self, image_id: str) -> bool:
        """Delete an image from gallery."""
        with self._state_lock:
            for i, img in enumerate(self._gallery):
                if img.id == image_id:
                    # Delete file if exists
                    try:
                        Path(img.path).unlink(missing_ok=True)
                    except:
                        pass
                    self._gallery.pop(i)
                    return True
        return False


# Singleton accessor
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get the inference service singleton instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
