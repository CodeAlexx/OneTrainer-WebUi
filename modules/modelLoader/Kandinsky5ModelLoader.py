"""
Kandinsky 5 Model Loader

Loads Kandinsky 5 model components using the official kandinsky-5-code implementation.
"""

import os
import sys
import torch
import safetensors.torch

from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import (
    replace_linear_with_quantized_layers,
    quantize_layers,
)

# Add kandinsky-5-code to path
KANDINSKY_CODE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'kandinsky-5-code')
if KANDINSKY_CODE_PATH not in sys.path:
    sys.path.insert(0, KANDINSKY_CODE_PATH)


class Kandinsky5ModelLoader(BaseModelLoader):
    """Loader for Kandinsky 5 models."""

    def _get_hf_cache_path(self, repo_id: str) -> str:
        """Get the path to a HuggingFace cached model."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, model_dir)

        if os.path.exists(model_path):
            # Find the snapshot
            refs_path = os.path.join(model_path, "refs", "main")
            if os.path.exists(refs_path):
                with open(refs_path, 'r') as f:
                    commit_hash = f.read().strip()
                snapshot_path = os.path.join(model_path, "snapshots", commit_hash)
                if os.path.exists(snapshot_path):
                    return snapshot_path
        return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig = None
    ) -> Kandinsky5Model | None:
        """
        Load a Kandinsky 5 model.

        Expected model_names.base_model format:
        - Path to safetensors file: /path/to/kandinsky5pro_t2v_sft_5s.safetensors
        - Or directory containing model/ subfolder
        """
        model = Kandinsky5Model(model_type)

        base_path = model_names.base_model

        # Determine safetensors path
        if base_path.endswith('.safetensors'):
            safetensors_path = base_path
        elif os.path.isdir(base_path):
            # Look for safetensors in model/ subfolder
            model_dir = os.path.join(base_path, 'model')
            if os.path.isdir(model_dir):
                safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
                if safetensors_files:
                    safetensors_path = os.path.join(model_dir, safetensors_files[0])
                else:
                    print(f"Error: No safetensors files found in {model_dir}")
                    return None
            else:
                safetensors_files = [f for f in os.listdir(base_path) if f.endswith('.safetensors')]
                if safetensors_files:
                    safetensors_path = os.path.join(base_path, safetensors_files[0])
                else:
                    print(f"Error: No safetensors files found in {base_path}")
                    return None
        else:
            print(f"Error: Invalid base path: {base_path}")
            return None

        print(f"Loading Kandinsky 5 model from {safetensors_path}...")

        # Detect model variant from filename
        is_pro = 'pro' in safetensors_path.lower()
        is_lite = 'lite' in safetensors_path.lower()

        # Pro model config (60 visual blocks, 4096 dim)
        if is_pro:
            dit_config = {
                'in_visual_dim': 16,
                'out_visual_dim': 16,
                'in_text_dim': 3584,
                'in_text_dim2': 768,
                'time_dim': 1024,
                'patch_size': (1, 2, 2),
                'model_dim': 4096,
                'ff_dim': 16384,
                'num_text_blocks': 4,
                'num_visual_blocks': 60,
                'axes_dims': (32, 48, 48),
                'visual_cond': True,
            }
        else:
            # Lite model config (32 visual blocks, 1792 dim) - from HF kandinsky5lite_t2v
            # Note: visual_embeddings expects 2*in_visual_dim+1=33 when visual_cond=True
            dit_config = {
                'in_visual_dim': 16,  # VAE channels (visual_embed_dim becomes 2*16+1=33 with visual_cond)
                'out_visual_dim': 16,
                'in_text_dim': 3584,
                'in_text_dim2': 768,
                'time_dim': 512,
                'patch_size': (1, 2, 2),
                'model_dim': 1792,
                'ff_dim': 7168,
                'num_text_blocks': 2,
                'num_visual_blocks': 32,
                'axes_dims': (16, 24, 24),  # sum=64 (head_dim for QK norm)
                'visual_cond': True,
            }

        # 1. Load DiT transformer
        try:
            from kandinsky.models.dit import DiffusionTransformer3D

            print(f"Creating DiffusionTransformer3D with config: {dit_config}")
            transformer = DiffusionTransformer3D(**dit_config)

            # Load weights
            print(f"Loading transformer weights from {safetensors_path}...")
            state_dict = safetensors.torch.load_file(safetensors_path)

            # Load with strict=False to handle any mismatches
            missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")

            transformer_dtype = weight_dtypes.transformer.torch_dtype() if weight_dtypes.transformer else torch.bfloat16
            train_dtype = weight_dtypes.train_dtype if weight_dtypes.train_dtype else weight_dtypes.fallback_train_dtype

            # Apply quantization if enabled (NF4, INT8, etc.)
            if weight_dtypes.transformer.is_quantized():
                print(f"Applying {weight_dtypes.transformer} quantization to transformer...")
                replace_linear_with_quantized_layers(
                    transformer,
                    weight_dtypes.transformer,
                    keep_in_fp32_modules=[],  # Keep all layers quantizable
                    quantization=quantization,
                    copy_parameters=True,
                )
                # Move to dtype and quantize
                transformer = transformer.to(dtype=train_dtype.torch_dtype())
            else:
                transformer = transformer.to(dtype=transformer_dtype)

            model.transformer = transformer
            print(f"Transformer loaded successfully with {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B parameters")

        except Exception as e:
            print(f"Error loading transformer: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 2. Load VAE (HunyuanVideo VAE)
        try:
            hunyuan_path = self._get_hf_cache_path("hunyuanvideo-community/HunyuanVideo")
            if hunyuan_path:
                print(f"Loading HunyuanVideo VAE from {hunyuan_path}...")
                from kandinsky.models.vae import AutoencoderKLHunyuanVideo

                vae_path = os.path.join(hunyuan_path, "vae")
                if os.path.exists(vae_path):
                    vae_dtype = weight_dtypes.vae.torch_dtype() if weight_dtypes.vae else torch.float16
                    model.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                        vae_path,
                        torch_dtype=vae_dtype
                    )
                    print("VAE loaded successfully")
                else:
                    print(f"VAE path not found: {vae_path}")
            else:
                print("HunyuanVideo not found in cache, trying diffusers...")
                from diffusers import AutoencoderKLHunyuanVideo
                model.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    subfolder="vae",
                    torch_dtype=torch.float16
                )
        except Exception as e:
            print(f"Error loading VAE: {e}")
            import traceback
            traceback.print_exc()

        # 3. Load Text Encoders
        try:
            # Qwen2.5-VL-7B
            qwen_path = self._get_hf_cache_path("Qwen/Qwen2.5-VL-7B-Instruct")
            if qwen_path:
                print(f"Loading Qwen2.5-VL from {qwen_path}...")
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

                text_encoder_dtype = weight_dtypes.text_encoder.torch_dtype() if weight_dtypes.text_encoder else torch.bfloat16
                model.text_encoder_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    qwen_path,
                    torch_dtype=text_encoder_dtype,
                    device_map='cpu',
                    low_cpu_mem_usage=True,
                )
                model.processor_qwen = AutoProcessor.from_pretrained(qwen_path, use_fast=True)
                print("Qwen2.5-VL loaded successfully")
            else:
                print("Qwen2.5-VL-7B-Instruct not found in cache")

            # CLIP
            clip_path = self._get_hf_cache_path("openai/clip-vit-large-patch14")
            if clip_path:
                print(f"Loading CLIP from {clip_path}...")
                from transformers import CLIPTextModel, CLIPTokenizer

                model.text_encoder_clip = CLIPTextModel.from_pretrained(
                    clip_path,
                    torch_dtype=torch.float16
                )
                model.tokenizer_clip = CLIPTokenizer.from_pretrained(clip_path)
                print("CLIP loaded successfully")
            else:
                print("CLIP not found in cache")

        except Exception as e:
            print(f"Error loading text encoders: {e}")
            import traceback
            traceback.print_exc()

        # 4. Set up scheduler
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=1.0,
            )
        except Exception as e:
            print(f"Error setting up scheduler: {e}")

        # Store config for later use
        model.dit_config = dit_config
        model.transformer_train_dtype = weight_dtypes.transformer

        return model
