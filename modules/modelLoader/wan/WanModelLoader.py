"""
Wan Model Loader - Loads Wan 2.1/2.2 models from official checkpoints

Based on diffusion-pipe's loading logic, adapted for OneTrainer.
Supports:
- T2V (text-to-video) 1.3B and 14B
- I2V (image-to-video)
- Wan 2.2 variants
"""

import copy
import os
import re
import json
import traceback
from pathlib import Path

from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch
from torch import nn
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


# Layer names to keep in high precision
KEEP_IN_HIGH_PRECISION = [
    'norm', 'bias', 'patch_embedding', 'text_embedding',
    'time_embedding', 'time_projection', 'head', 'modulation'
]


class WanModelLoader(HFModelLoaderMixin):
    """
    Loader for Wan video generation models.
    
    Loads from official checkpoint directories which contain:
    - transformer safetensors
    - T5 text encoder
    - VAE
    - Optional CLIP for i2v
    """
    
    def __init__(self):
        super().__init__()
    
    def _detect_model_variant(self, ckpt_dir: Path, weight_keys: set) -> tuple[str, dict]:
        """
        Auto-detect model variant from config and weights.
        
        Returns:
            (model_type, config_dict)
        """
        # Look for config file - diffusion-pipe format
        config_path = ckpt_dir / 'config.json'
        if not config_path.exists():
            # Wan2.2 has subdirectories
            config_path = ckpt_dir / 'low_noise_model' / 'config.json'
        
        # Check for diffusers format - config in transformer subdirectory
        if not config_path.exists():
            config_path = ckpt_dir / 'transformer' / 'config.json'
        
        # Also check for model_index.json (diffusers standard)
        model_index_path = ckpt_dir / 'model_index.json'
        is_diffusers_format = model_index_path.exists() or (ckpt_dir / 'transformer').exists()
        
        if not config_path.exists():
            if is_diffusers_format:
                # Diffusers format without explicit config - infer from structure
                print("Detected diffusers format without config.json, inferring defaults...")
                config = {
                    'model_type': 't2v',
                    'dim': 1536,  # Default for 14B
                    'is_diffusers_format': True,
                }
                return 't2v', config
            raise ValueError(f"Could not find config.json in {ckpt_dir}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Handle diffusers format config (WanTransformer3DModel config)
        if '_class_name' in config and 'WanTransformer' in config.get('_class_name', ''):
            config['is_diffusers_format'] = True
            config['model_type'] = config.get('model_type', 't2v')
            config['dim'] = config.get('hidden_size', 1536)
            return config.get('model_type', 't2v'), config
        
        model_type = config.get('model_type', 't2v')
        model_dim = config.get('dim', 1536)
        
        # Detect i2v_v2 (Wan 2.2)
        if model_type == 'i2v' and 'blocks.0.cross_attn.k_img.weight' not in weight_keys:
            model_type = 'i2v_v2'
        
        return model_type, config

    
    def _load_t5_encoder(
            self,
            ckpt_dir: Path,
            dtype: torch.dtype,
            config: dict,
            t5_path: Path | None = None,
    ) -> tuple[nn.Module, object]:
        """
        Load T5 text encoder and tokenizer.
        
        Returns:
            (text_encoder, tokenizer)
        """
        # Check if this is diffusers format
        if config.get('is_diffusers_format'):
            # Load from diffusers/transformers format
            text_encoder_dir = t5_path if t5_path else ckpt_dir / 'text_encoder'
            
            # For diffusers format, tokenizer is usually separate. 
            # If t5_path is provided, we assume it might contain tokenizer or we look relative to it?
            # Or we stick to original logic if t5_path is not provided.
            
            # If override is provided, assume it's a full diffusers model path or the component path
            if t5_path:
                # Check if it's pointing to the component directly or a model root
                if (t5_path / 'config.json').exists(): # Direct component
                    text_encoder_dir = t5_path
                    tokenizer_dir = t5_path # Often tokenizer is in same or separate.
                    # This might be tricky. Let's assume standard structure if override is root.
                
                # Check if it's a root containing 'text_encoder'
                if (t5_path / 'text_encoder').exists():
                    text_encoder_dir = t5_path / 'text_encoder'
                    tokenizer_dir = t5_path / 'tokenizer'
                else:
                    # Assume mapped directly
                    tokenizer_dir = text_encoder_dir 
            else:
                 text_encoder_dir = ckpt_dir / 'text_encoder'
                 tokenizer_dir = ckpt_dir / 'tokenizer'

            
            if text_encoder_dir.exists():
                print(f"Loading T5 from diffusers format: {text_encoder_dir}")
                try:
                    from transformers import UMT5EncoderModel, AutoTokenizer
                    
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        str(text_encoder_dir),
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map='cpu',
                    )
                    
                    # Try loading tokenizer
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                    except Exception as e:
                        print(f"Warning: Failed to load tokenizer from {tokenizer_dir}: {e}")
                        # Fallback to default tokenizer if path is weird
                        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

                    return text_encoder, tokenizer
                except Exception as e:
                    print(f"Warning: Failed to load T5 from diffusers format: {e}")
                    return None, None
            else:
                print(f"Warning: T5 encoder not found at {text_encoder_dir}")
                return None, None
        
        # Original diffusion-pipe format loading
        try:
            from modules.model.wan.t5 import T5EncoderModel
        except ImportError:
            print("Warning: Wan T5 encoder not found, returning None")
            return None, None
        
        t5_checkpoint = config.get('t5_checkpoint', 'models_t5_umt5-xxl-enc-bf16.pth')
        t5_tokenizer = config.get('t5_tokenizer', 'google/umt5-xxl')
        text_len = config.get('text_len', 512)
        
        llm_path = ckpt_dir / t5_checkpoint
        
        text_encoder = T5EncoderModel(
            text_len=text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=str(llm_path),
            tokenizer_path=str(ckpt_dir / t5_tokenizer) if (ckpt_dir / t5_tokenizer).exists() else t5_tokenizer,
        )
        
        return text_encoder.model, text_encoder.tokenizer
    
    def _load_vae(
            self,
            ckpt_dir: Path,
            dtype: torch.dtype,
            config: dict,
            vae_path: Path | None = None,
    ) -> object:
        """
        Load Wan 3D VAE.
        """
        # Check if this is diffusers format
        if config.get('is_diffusers_format'):
            vae_dir = vae_path if vae_path else ckpt_dir / 'vae'
            
            if vae_dir.exists():
                print(f"Loading VAE from diffusers format: {vae_dir}")
                try:
                    from diffusers import AutoencoderKLWan
                    
                    vae = AutoencoderKLWan.from_pretrained(
                        str(vae_dir),
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map='cpu',
                    )
                    return vae
                except Exception as e:
                    print(f"Warning: Failed to load VAE from diffusers format: {e}")
                    return None
            else:
                print(f"Warning: VAE not found at {vae_dir}")
                return None
        
        # Original diffusion-pipe format
        try:
            from modules.model.wan.vae import WanVAE
        except ImportError:
            print("Warning: Wan VAE not found, returning None")
            return None
        
        model_type = config.get('model_type', 't2v')
        vae_checkpoint = config.get('vae_checkpoint', 'Wan2.1_VAE.pth')
        
        vae = WanVAE(
            vae_pth=str(ckpt_dir / vae_checkpoint),
            device='cpu',
            dtype=dtype,
        )
        
        return vae
    
    def _load_clip(
            self,
            ckpt_dir: Path,
            dtype: torch.dtype,
            config: dict,
    ) -> nn.Module:
        """
        Load CLIP model for i2v conditioning.
        """
        try:
            from modules.model.wan.clip import CLIPModel
        except ImportError:
            print("Warning: Wan CLIP not found, returning None")
            return None
        
        clip_checkpoint = config.get('clip_checkpoint', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
        clip_tokenizer = config.get('clip_tokenizer', 'xlm-roberta-large')
        
        clip = CLIPModel(
            dtype=dtype,
            device='cpu',
            checkpoint_path=str(ckpt_dir / clip_checkpoint),
            tokenizer_path=str(ckpt_dir / clip_tokenizer) if (ckpt_dir / clip_tokenizer).exists() else clip_tokenizer,
        )
        
        return clip
    
    def _load_transformer(
            self,
            ckpt_dir: Path,
            transformer_path: Path | None,
            dtype: torch.dtype,
            config: dict,
            weight_dtypes: ModelWeightDtypes = None,
    ) -> nn.Module:
        """
        Load Wan transformer model.
        
        For Wan 2.2 14B, detects and loads both transformer and transformer_2,
        returning a DualWanTransformer3DModel wrapper.
        """
        # Check if this is diffusers format
        if config.get('is_diffusers_format'):
            tf_path = ckpt_dir / 'transformer'
            tf2_path = ckpt_dir / 'transformer_2'
            
            # Detect if this is Wan 2.2 with dual transformers
            is_dual = tf_path.exists() and tf2_path.exists()
            
            if tf_path.exists():
                print(f"Loading transformer from diffusers format: {tf_path}")
                if is_dual:
                    print("Detected Wan 2.2 dual transformer model (transformer + transformer_2)")
                
                try:
                    from diffusers import WanTransformer3DModel
                    
                    # Check if quantization is enabled via weight dtype
                    use_quantization = weight_dtypes and weight_dtypes.transformer.quantize_nf4()
                    
                    def quantize_transformer(transformer, name="transformer"):
                        """Quantize transformer using torchao uint4 (like ai-toolkit).
                        
                        Quantizes block-by-block to fit in VRAM:
                        1. Move one block to GPU
                        2. Quantize it
                        3. Move back to CPU
                        4. Repeat for all blocks
                        """
                        try:
                            from torchao.quantization.quant_api import quantize_, UIntXWeightOnlyConfig
                            from optimum.quanto import freeze
                            from tqdm import tqdm
                            
                            print(f"Quantizing {name} with torchao uint4...")
                            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                            
                            # Keep transformer on CPU, quantize blocks one at a time
                            transformer.to('cpu', dtype=dtype)
                            
                            # Quantize transformer blocks one-by-one
                            config = UIntXWeightOnlyConfig(torch.uint4)
                            blocks = list(transformer.blocks)
                            
                            for block in tqdm(blocks, desc=f"Quantizing {name} blocks"):
                                # Move block to GPU
                                block.to(device, dtype=dtype)
                                # Quantize
                                quantize_(block, config)
                                freeze(block)
                                # Move back to CPU
                                block.to('cpu')
                                torch.cuda.empty_cache()
                            
                            print(f"✓ {name} quantized ({len(blocks)} blocks)")
                            
                        except ImportError as e:
                            print(f"Warning: torchao/quanto not available, using bf16: {e}")
                            transformer.to('cpu', dtype=dtype)
                    
                    # Load transformer_1 (always exists)
                    print("Loading transformer_1 (high noise stage)...")
                    transformer_1 = WanTransformer3DModel.from_pretrained(
                        str(tf_path),
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map='cpu',
                    )
                    
                    if use_quantization and is_dual:
                        # Quantize to save memory before loading second transformer
                        quantize_transformer(transformer_1, "transformer_1")
                    else:
                        transformer_1.to('cpu', dtype=dtype)
                    
                    if not is_dual:
                        # Single transformer model (Wan 2.1 or others)
                        if use_quantization:
                            quantize_transformer(transformer_1, "transformer")
                        return transformer_1
                    
                    torch.cuda.empty_cache()
                    
                    # Load transformer_2 for Wan 2.2
                    print("Loading transformer_2 (low noise stage)...")
                    transformer_2 = WanTransformer3DModel.from_pretrained(
                        str(tf2_path),
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map='cpu',
                    )
                    
                    if use_quantization:
                        quantize_transformer(transformer_2, "transformer_2")
                    else:
                        transformer_2.to('cpu', dtype=dtype)
                    
                    # Create dual transformer wrapper
                    from modules.model.wan.DualWanTransformer import DualWanTransformer3DModel
                    
                    # Low VRAM mode is required for 24GB training - always enable for dual transformer
                    low_vram = True
                    
                    print(f"Creating DualWanTransformer3DModel (low_vram={low_vram})...")
                    dual_transformer = DualWanTransformer3DModel(
                        transformer_1=transformer_1,
                        transformer_2=transformer_2,
                        boundary_ratio=0.875,  # Wan 2.2 standard boundary
                        low_vram=low_vram,
                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                        dtype=dtype,
                    )
                    
                    return dual_transformer
                    
                except Exception as e:
                    print(f"Warning: Failed to load transformer from diffusers format: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                print(f"Warning: Transformer not found at {tf_path}")
                return None


        
        # Original diffusion-pipe format
        try:
            from modules.model.wan.model import WanModel as WanTransformer
        except ImportError:
            print("Warning: Wan transformer not found, returning None")
            return None

        
        # Determine weights file
        if transformer_path and transformer_path.exists():
            weights_path = transformer_path
        else:
            # Look for safetensors in ckpt_dir
            safetensors_files = list(ckpt_dir.glob('*.safetensors'))
            if safetensors_files:
                weights_path = safetensors_files[0]
            else:
                # Check subfolders
                for subfolder in ['transformer', 'low_noise_model']:
                    sf_path = ckpt_dir / subfolder
                    if sf_path.exists():
                        safetensors_files = list(sf_path.glob('*.safetensors'))
                        if safetensors_files:
                            weights_path = sf_path
                            break
        
        # Create model with empty weights
        with init_empty_weights():
            transformer = WanTransformer.from_config(config)
        
        # Load weights
        state_dict = {}
        if weights_path.is_file():
            with safetensors.safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[re.sub(r'^model\.diffusion_model\.', '', key)] = f.get_tensor(key)
        else:
            # Load from directory with multiple shards
            for shard in weights_path.glob('*.safetensors'):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[re.sub(r'^model\.diffusion_model\.', '', key)] = f.get_tensor(key)
        
        # Set tensors to device
        for name, param in transformer.named_parameters():
            dtype_to_use = dtype if any(kw in name for kw in KEEP_IN_HIGH_PRECISION) else dtype
            if name in state_dict:
                set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
        
        return transformer
    
    def load(
            self,
            model: WanModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        """
        Load a Wan model from checkpoint.
        """
        stacktraces = []

        # Resolve HuggingFace model name to local path if needed
        base_model_path = model_names.base_model
        if '/' in base_model_path and not Path(base_model_path).exists():
            # Looks like a HuggingFace model name (org/model)
            try:
                from huggingface_hub import snapshot_download
                print(f"Resolving HuggingFace model: {base_model_path}")
                base_model_path = snapshot_download(
                    repo_id=base_model_path,
                    local_files_only=True,  # Use cached version
                )
                print(f"Resolved to: {base_model_path}")
            except Exception as e:
                print(f"Could not resolve HuggingFace model, trying as local path: {e}")

        ckpt_dir = Path(base_model_path)
        transformer_path = Path(model_names.transformer_model) if model_names.transformer_model else None
        
        try:
            # Get weight keys for variant detection
            weight_keys = set()
            safetensors_files = list(ckpt_dir.glob('*.safetensors'))
            if not safetensors_files:
                safetensors_files = list((ckpt_dir / 'transformer').glob('*.safetensors'))
            
            for sf in safetensors_files[:1]:  # Just check first file for keys
                with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        weight_keys.add(re.sub(r'^model\.diffusion_model\.', '', k))
            
            # Detect variant
            model_variant, config = self._detect_model_variant(ckpt_dir, weight_keys)
            
            dtype = weight_dtypes.transformer.torch_dtype() or torch.bfloat16
            
            # Load components
            if model_names.include_text_encoder:
                t5_override = Path(model_names.text_encoder_model) if model_names.text_encoder_model else None
                text_encoder, tokenizer = self._load_t5_encoder(ckpt_dir, dtype, config, t5_path=t5_override)
            else:
                text_encoder, tokenizer = None, None
            
            vae_override = Path(model_names.vae_model) if model_names.vae_model else None
            vae = self._load_vae(ckpt_dir, weight_dtypes.vae.torch_dtype() or dtype, config, vae_path=vae_override)
            
            # Load CLIP for i2v
            clip = None
            if model_variant in ('i2v', 'flf2v'):
                clip = self._load_clip(ckpt_dir, dtype, config)
            
            transformer = self._load_transformer(ckpt_dir, transformer_path, dtype, config, weight_dtypes)
            
            # Set model properties
            model.model_type = model_type
            model.model_type_variant = model_variant
            
            # Handle different formats - diffusers models don't have .model attribute
            if config.get('is_diffusers_format'):
                model.text_encoder = text_encoder
                model.tokenizer = tokenizer
                model.vae = vae
                model.clip = clip
                model.transformer = transformer
                
                # Check if this is a dual transformer (Wan 2.2)
                from modules.model.wan.DualWanTransformer import DualWanTransformer3DModel
                if isinstance(transformer, DualWanTransformer3DModel):
                    model.is_dual_transformer = True
                    model.low_vram = transformer.low_vram
                    print("✓ Loaded Wan 2.2 dual transformer model")
                else:
                    model.is_dual_transformer = False
            else:
                model.text_encoder = text_encoder.model if text_encoder and hasattr(text_encoder, 'model') else text_encoder
                model.tokenizer = tokenizer
                model.vae = vae.model if vae and hasattr(vae, 'model') else vae
                model.clip = clip.model if clip and hasattr(clip, 'model') else clip
                model.transformer = transformer
                model.is_dual_transformer = False
            
            model.text_len = config.get('text_len', 512)
            
            # Set framerate
            if model_variant == 'ti2v':
                model.framerate = 24
            else:
                model.framerate = 16
            
            # Store original tokenizer
            model.orig_tokenizer = copy.deepcopy(tokenizer)
            
            return
            
        except Exception:
            stacktraces.append(traceback.format_exc())
        
        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("Could not load Wan model: " + model_names.base_model)
