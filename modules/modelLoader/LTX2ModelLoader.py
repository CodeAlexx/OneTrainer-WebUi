"""
LTX-2 Model Loader for OneTrainer

Loads LTX-2 model components from safetensors files with optional INT8 quantization.
"""
import os
import traceback
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.model.LTX2Model import LTX2Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch
from safetensors.torch import load_file, load_model

# LTX-2 imports
try:
    from ltx_core.model.transformer import LTXModel, LTXModelConfigurator
    from ltx_core.model.video_vae import VideoEncoder, VideoDecoder
    from ltx_core.model.video_vae.model_configurator import (
        VideoEncoderConfigurator,
        VideoDecoderConfigurator,
    )
    from ltx_core.text_encoders.gemma import AVGemmaTextEncoderModel
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma.encoders.av_encoder import (
        AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        AVGemmaTextEncoderModelConfigurator,
    )
    from ltx_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
    LTX_CORE_AVAILABLE = True
except ImportError:
    LTX_CORE_AVAILABLE = False

# Quantization support
try:
    from optimum.quanto import freeze, qint8, quantize
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False


class LTX2VAEEncoderWrapper(torch.nn.Module):
    """Wrapper to make ltx-core VideoEncoder compatible with diffusers-style API."""

    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder

    def encode(self, x):
        """Encode input and return a distribution-like object."""
        latent = self._encoder(x)
        return LTX2LatentDistribution(latent)

    def forward(self, x):
        return self._encoder(x)

    def to(self, *args, **kwargs):
        self._encoder = self._encoder.to(*args, **kwargs)
        return self

    def eval(self):
        self._encoder.eval()
        return self


class LTX2VAEDecoderWrapper(torch.nn.Module):
    """Wrapper to make ltx-core VideoDecoder compatible with diffusers-style API."""

    def __init__(self, decoder):
        super().__init__()
        self._decoder = decoder

    def decode(self, x):
        """Decode latents."""
        return self._decoder(x)

    def forward(self, x):
        return self._decoder(x)

    def to(self, *args, **kwargs):
        self._decoder = self._decoder.to(*args, **kwargs)
        return self

    def eval(self):
        self._decoder.eval()
        return self


class LTX2LatentDistribution:
    """Simple distribution-like object for VAE latents.

    Mimics diffusers DiagonalGaussianDistribution for compatibility.
    Note: EncodeVAE checks hasattr for latent_dist and latent - if both exist,
    latent takes precedence and returns raw tensor. So we only provide latent_dist.
    """

    def __init__(self, latent_tensor):
        self._latent_tensor = latent_tensor
        # For diffusers compatibility (EncodeVAE checks this)
        # Set latent_dist to self so SampleVAEDistribution can call .mode()
        self.latent_dist = self

    def sample(self):
        return self._latent_tensor

    def mean(self):
        return self._latent_tensor

    def mode(self):
        return self._latent_tensor


class LTX2ModelLoader:
    def __init__(self):
        if not LTX_CORE_AVAILABLE:
            raise ImportError(
                "ltx-core package required for LTX-2. Install from: "
                "https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core"
            )

    def _load_transformer(
            self,
            checkpoint_path: str,
            dtype: torch.dtype = torch.bfloat16,
            device: str = "cpu",
            quantize_int8: bool = False,
    ) -> LTXModel:
        """Load LTX-2 transformer from safetensors checkpoint."""
        import json
        from safetensors import safe_open

        print(f"Loading LTX-2 transformer from {checkpoint_path}...")

        # Load config from safetensors metadata (the proper way)
        with safe_open(checkpoint_path, framework='pt') as f:
            metadata = f.metadata()

        if metadata and 'config' in metadata:
            config = json.loads(metadata['config'])
            print(f"Loaded config from safetensors metadata")
        else:
            # Fallback config
            config = {
                "transformer": {
                    "num_attention_heads": 32,
                    "attention_head_dim": 128,
                    "in_channels": 128,
                    "out_channels": 128,
                    "num_layers": 48,
                    "cross_attention_dim": 4096,
                    "caption_channels": 3840,
                    "dropout": 0.0,
                    "attention_bias": True,
                    "activation_fn": "gelu-approximate",
                }
            }
            print("Using fallback config")

        # Create model on meta device first (instant, no memory allocation)
        print("Creating model on meta device...")
        with torch.device("meta"):
            transformer = LTXModelConfigurator.from_config(config)
        print("Meta model created")

        # Load state dict
        print("Loading state dict...")
        full_state_dict = load_file(checkpoint_path)

        # Keys in file have 'model.diffusion_model.' prefix, model expects no prefix
        transformer_state_dict = {}
        prefix = "model.diffusion_model."
        for key, value in full_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                transformer_state_dict[new_key] = value

        print(f"Found {len(transformer_state_dict)} transformer keys")

        # Load weights with assign=True to move from meta to real tensors
        missing, unexpected = transformer.load_state_dict(transformer_state_dict, strict=False, assign=True)

        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        # Apply quantization if requested
        if quantize_int8 and QUANTO_AVAILABLE:
            print("Applying INT8 quantization to transformer...")
            # Exclude certain layers from quantization
            exclude_patterns = [
                "proj_in",
                "time_embed",
                "caption_projection",
                "rope",
                "norm",
                "proj_out",
            ]

            def should_quantize(name):
                return not any(pattern in name for pattern in exclude_patterns)

            # Quantize linear layers
            quantize(transformer, weights=qint8, exclude=lambda n, m: not should_quantize(n))
            freeze(transformer)
            print("INT8 quantization applied")
        else:
            transformer = transformer.to(dtype=dtype)

        return transformer

    def _load_text_encoder(
            self,
            checkpoint_path: str,
            gemma_model_path: str,
            dtype: torch.dtype = torch.bfloat16,
            device: str = "cpu",
            quantize_int8: bool = False,
    ) -> AVGemmaTextEncoderModel:
        """Load Gemma text encoder for LTX-2."""

        if not Path(gemma_model_path).is_dir():
            raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

        print(f"Loading Gemma text encoder from {gemma_model_path}...")

        # Build text encoder using ltx-core loader
        text_encoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=AVGemmaTextEncoderModelConfigurator,
            model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            module_ops=module_ops_from_gemma_root(str(gemma_model_path)),
        ).build(device=torch.device(device), dtype=dtype)

        # Apply quantization if requested
        if quantize_int8 and QUANTO_AVAILABLE:
            print("Applying INT8 quantization to text encoder...")
            quantize(text_encoder, weights=qint8)
            freeze(text_encoder)

        return text_encoder

    def _load_vae(
            self,
            checkpoint_path: str,
            dtype: torch.dtype = torch.bfloat16,
            device: str = "cpu",
            load_encoder: bool = True,
            load_decoder: bool = True,
    ) -> tuple:
        """Load Video VAE encoder and/or decoder."""
        import json
        from safetensors import safe_open

        encoder = None
        decoder = None

        # Load config from safetensors metadata
        with safe_open(checkpoint_path, framework='pt') as f:
            metadata = f.metadata()

        if metadata and 'config' in metadata:
            config = json.loads(metadata['config'])
        else:
            config = {}

        # Load full state dict once
        state_dict = load_file(checkpoint_path)

        if load_encoder:
            print("Loading Video VAE encoder...")
            # Create encoder from config on meta device
            with torch.device("meta"):
                encoder = VideoEncoderConfigurator.from_config(config)
            print("Encoder meta model created")

            # VAE encoder keys have 'vae.encoder.' prefix in checkpoint
            encoder_prefix = "vae.encoder."
            encoder_state = {
                k[len(encoder_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(encoder_prefix)
            }
            # Also get per_channel_statistics
            stats_prefix = "vae.per_channel_statistics."
            for k, v in state_dict.items():
                if k.startswith(stats_prefix):
                    encoder_state["per_channel_statistics." + k[len(stats_prefix):]] = v

            if encoder_state:
                print(f"Found {len(encoder_state)} encoder keys")
                encoder.load_state_dict(encoder_state, strict=False, assign=True)
            else:
                print("Warning: No VAE encoder keys found in checkpoint")
            encoder = encoder.to(dtype=dtype)

        if load_decoder:
            print("Loading Video VAE decoder...")
            # Create decoder from config on meta device
            with torch.device("meta"):
                decoder = VideoDecoderConfigurator.from_config(config)
            print("Decoder meta model created")

            # VAE decoder keys have 'vae.decoder.' prefix in checkpoint
            decoder_prefix = "vae.decoder."
            decoder_state = {
                k[len(decoder_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(decoder_prefix)
            }
            # Also get per_channel_statistics
            stats_prefix = "vae.per_channel_statistics."
            for k, v in state_dict.items():
                if k.startswith(stats_prefix):
                    decoder_state["per_channel_statistics." + k[len(stats_prefix):]] = v

            if decoder_state:
                print(f"Found {len(decoder_state)} decoder keys")
                decoder.load_state_dict(decoder_state, strict=False, assign=True)
            else:
                print("Warning: No VAE decoder keys found in checkpoint")
            decoder = decoder.to(dtype=dtype)

        # Wrap encoder/decoder to provide diffusers-style API (encode/decode methods)
        if encoder is not None:
            encoder = LTX2VAEEncoderWrapper(encoder)
        if decoder is not None:
            decoder = LTX2VAEDecoderWrapper(decoder)

        return encoder, decoder

    def load(
            self,
            model: LTX2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        """Load all LTX-2 model components."""

        stacktraces = []

        try:
            self._load_from_safetensors(
                model, model_type, weight_dtypes, model_names, quantization
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception(f"Could not load LTX-2 model: {model_names.base_model}")

    def _load_from_safetensors(
            self,
            model: LTX2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            model_names: ModelNames,
            quantization: QuantizationConfig,
    ):
        """Load model from safetensors checkpoint."""

        base_model_path = model_names.base_model
        text_encoder_path = model_names.text_encoder_model if hasattr(model_names, 'text_encoder_model') else None

        # Determine dtype
        transformer_dtype = weight_dtypes.transformer.torch_dtype() or torch.bfloat16
        text_encoder_dtype = weight_dtypes.text_encoder.torch_dtype() or torch.bfloat16
        vae_dtype = weight_dtypes.vae.torch_dtype() or torch.bfloat16

        # Check for INT8 quantization
        use_int8 = quantization is not None and "int8" in str(quantization).lower()

        # Load transformer
        transformer_path = model_names.transformer_model if model_names.transformer_model else base_model_path
        model.transformer = self._load_transformer(
            transformer_path,
            dtype=transformer_dtype,
            quantize_int8=use_int8,
        )

        # Load text encoder (requires Gemma model directory)
        if text_encoder_path and os.path.isdir(text_encoder_path):
            model.text_encoder = self._load_text_encoder(
                transformer_path,
                text_encoder_path,
                dtype=text_encoder_dtype,
                quantize_int8=use_int8,
            )
        else:
            print(f"Warning: Text encoder path not provided or not found: {text_encoder_path}")
            model.text_encoder = None

        # Load VAE
        vae_path = model_names.vae_model if model_names.vae_model else base_model_path
        model.vae_encoder, model.vae_decoder = self._load_vae(
            vae_path,
            dtype=vae_dtype,
            load_encoder=True,
            load_decoder=True,
        )

        # Create scheduler
        model.noise_scheduler = LTX2Scheduler()

        # Create patchifier
        model.patchifier = VideoLatentPatchifier(patch_size=1)

        model.model_type = model_type


class LTX2LoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: LTX2Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


# Create composite loaders
LTX2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.LTX_2: "resources/sd_model_spec/ltx2-lora.json",
    },
    model_class=LTX2Model,
    model_loader_class=LTX2ModelLoader,
    lora_loader_class=LTX2LoRALoader,
    embedding_loader_class=None,
)

LTX2FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.LTX_2: "resources/sd_model_spec/ltx2.json",
    },
    model_class=LTX2Model,
    model_loader_class=LTX2ModelLoader,
    embedding_loader_class=None,
)
