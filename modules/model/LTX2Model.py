"""
LTX-2 Video Model for OneTrainer

LTX-2 is a video generation model using:
- LTXModel transformer from ltx-core
- Gemma 3 text encoder
- Video VAE (encoder/decoder)
- LTX2Scheduler (flow matching)
"""
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

# LTX-2 uses custom ltx-core package
try:
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoEncoder, VideoDecoder
    from ltx_core.text_encoders.gemma import AVGemmaTextEncoderModel
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    LTX_CORE_AVAILABLE = True
except ImportError:
    LTX_CORE_AVAILABLE = False
    LTXModel = None
    VideoEncoder = None
    VideoDecoder = None
    AVGemmaTextEncoderModel = None
    LTX2Scheduler = None
    Modality = None
    VideoLatentPatchifier = None


class LTX2Model(BaseModel):
    # base model data
    tokenizer: object  # LTXVGemmaTokenizer
    noise_scheduler: object  # LTX2Scheduler
    text_encoder: object  # AVGemmaTextEncoderModel
    vae_encoder: object  # VideoEncoder
    vae_decoder: object  # VideoDecoder
    transformer: object  # LTXModel
    patchifier: object  # VideoLatentPatchifier

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext
    transformer_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType
    transformer_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent lora training data
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    # model config
    vae_spatial_compression: int
    vae_temporal_compression: int

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        if not LTX_CORE_AVAILABLE:
            raise ImportError(
                "ltx-core package not found. Install it from: "
                "https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core"
            )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae_encoder = None
        self.vae_decoder = None
        self.transformer = None
        self.patchifier = None

        self.text_encoder_autocast_context = nullcontext()
        self.transformer_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.BFLOAT_16
        self.transformer_train_dtype = DataType.BFLOAT_16

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

        # LTX-2 VAE compression factors
        self.vae_spatial_compression = 32
        self.vae_temporal_compression = 8

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        if self.vae_encoder is not None:
            self.vae_encoder.to(device=device)
        if self.vae_decoder is not None:
            self.vae_decoder.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        if self.vae_encoder is not None:
            self.vae_encoder.eval()
        if self.vae_decoder is not None:
            self.vae_decoder.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def encode_video(self, video: Tensor) -> Tensor:
        """Encode video frames to latent space.

        Args:
            video: [B, C, T, H, W] tensor of video frames (normalized to [-1, 1])

        Returns:
            latents: [B, C, T', H', W'] tensor of latent representations
        """
        if self.vae_encoder is None:
            raise RuntimeError("VAE encoder not loaded")

        # LTX-2 VAE expects [B, C, T, H, W]
        with torch.no_grad():
            latents = self.vae_encoder(video)
        return latents

    def decode_video(self, latents: Tensor) -> Tensor:
        """Decode latents to video frames.

        Args:
            latents: [B, C, T', H', W'] tensor of latent representations

        Returns:
            video: [B, C, T, H, W] tensor of video frames
        """
        if self.vae_decoder is None:
            raise RuntimeError("VAE decoder not loaded")

        with torch.no_grad():
            video = self.vae_decoder(latents)
        return video

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            text_encoder_output: Tensor = None,
            text_encoder_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode text prompts to embeddings.

        Returns:
            tuple of (embeddings, attention_mask)
        """
        if text_encoder_output is not None and text_encoder_mask is not None:
            return text_encoder_output, text_encoder_mask

        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        if isinstance(text, str):
            text = [text]

        with self.text_encoder_autocast_context:
            # Use ltx-core's text encoding
            embeddings, mask = self.text_encoder.encode(text)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            # Apply dropout by zeroing some embeddings
            if rand is not None and rand.random() < text_encoder_dropout_probability:
                embeddings = torch.zeros_like(embeddings)

        return embeddings, mask

    def create_modality(
            self,
            latents: Tensor,
            timesteps: Tensor,
            text_embeddings: Tensor,
            text_mask: Tensor,
    ) -> "Modality":
        """Create LTX-2 Modality object for transformer input.

        Args:
            latents: [B, C, T, H, W] noisy latents
            timesteps: [B] timesteps (0-1 range)
            text_embeddings: [B, L, D] text embeddings
            text_mask: [B, L] attention mask

        Returns:
            Modality object for transformer forward pass
        """
        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # Use patchifier to get positions
        if self.patchifier is None:
            # Create default patchifier if not loaded
            self.patchifier = VideoLatentPatchifier(
                patch_size=1,
                patch_size_t=1,
            )

        # Patchify latents to get token positions
        patches, positions = self.patchifier.patchify(latents)

        # Expand timesteps to per-token
        num_tokens = patches.shape[1]
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, num_tokens)

        return Modality(
            latent=patches,
            timesteps=timesteps_expanded,
            positions=positions,
            context=text_embeddings,
            context_mask=text_mask,
        )

    def scale_latents(self, latents: Tensor) -> Tensor:
        """Scale latents for training (if needed)."""
        # LTX-2 may or may not need scaling - check scheduler config
        return latents

    def unscale_latents(self, latents: Tensor) -> Tensor:
        """Unscale latents after training."""
        return latents
