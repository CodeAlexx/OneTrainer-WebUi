"""
QwenImageEditModel - Model class for Qwen-Image-Edit (image editing model)

This is different from QwenModel (text-to-image):
- Uses QwenImageEditPipeline (not QwenImagePipeline)
- Takes control image + text prompt as input
- Outputs edited image
"""

import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.musubi_block_swap import MusubiBlockSwapManager

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLQwenImage,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

DEFAULT_PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_PROMPT_TEMPLATE_CROP_START = 34
PROMPT_MAX_LENGTH = 512


class QwenImageEditModel(BaseModel):
    """
    Model class for Qwen-Image-Edit.
    
    Key differences from QwenModel:
    - Uses QwenImageEditPipeline for image editing workflow
    - Supports control image input
    - Encodes both control image and target image
    """
    
    # base model data
    tokenizer: Qwen2Tokenizer | None
    processor: Qwen2VLProcessor | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen2_5_VLForConditionalGeneration | None
    vae: AutoencoderKLQwenImage | None
    transformer: QwenImageTransformer2DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # Musubi block swap manager for training
    musubi_manager: MusubiBlockSwapManager | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.processor = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.musubi_manager = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        elif hasattr(self, 'musubi_manager') and self.musubi_manager is not None:
            # When Musubi is active, only move non-block parts to GPU
            # The transformer_blocks stay on CPU and are managed by Musubi hooks
            for name, module in self.transformer.named_children():
                if name != 'transformer_blocks':
                    module.to(device=device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        """Create QwenImageEditPipeline for inference."""
        return QwenImageEditPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

    def encode_control_image(
            self,
            control_image: Tensor,
            train_device: torch.device,
    ) -> Tensor:
        """
        Encode control/input image to latents using VAE.
        
        Args:
            control_image: [B, C, H, W] tensor in range [-1, 1]
            train_device: Device to use
            
        Returns:
            control_latents: [B, C, 1, H//8, W//8] tensor
        """
        with torch.no_grad():
            # Add frame dimension for VAE: [B, C, H, W] -> [B, C, 1, H, W]
            if control_image.dim() == 4:
                control_image = control_image.unsqueeze(2)
            
            control_image = control_image.to(train_device)
            control_latents = self.vae.encode(control_image).latent_dist.sample()
            control_latents = self.scale_latents(control_latents)
            
        return control_latents

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
            control_image: Tensor = None,  # For image-conditioned encoding
    ) -> tuple[Tensor, Tensor]:
        """
        Encode text (and optionally control image) for image editing.
        
        For Qwen-Image-Edit, the text encoder (Qwen2.5-VL) can process
        both text and image to create a joint embedding.
        """
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            text = [DEFAULT_PROMPT_TEMPLATE.format(t) for t in text]
            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH + DEFAULT_PROMPT_TEMPLATE_CROP_START,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                text_encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    output_hidden_states=True,
                    return_dict=True,
                )
                text_encoder_output = text_encoder_output.hidden_states[-1]
                tokens_mask = tokens_mask[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:]
                text_encoder_output = text_encoder_output[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:,:] * tokens_mask.unsqueeze(-1)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # Prune masked tokens
        seq_lengths = tokens_mask.sum(dim=1)
        max_seq_length = seq_lengths.max().item()

        if max_seq_length % 16 > 0 and (seq_lengths != max_seq_length).any():
            max_seq_length += (16 - max_seq_length % 16)

        text_encoder_output = text_encoder_output[:, :max_seq_length, :]
        bool_attention_mask = tokens_mask[:, :max_seq_length].bool()

        return (text_encoder_output, bool_attention_mask)

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        """Pack latents for transformer input."""
        batch_size, channels, frames, height, width = latents.shape
        assert frames == 1

        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

        return latents

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        """Unpack latents from transformer output."""
        batch_size, _, channels = latents.shape

        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height * 2, width * 2)

        return latents

    def scale_latents(self, latents: Tensor) -> Tensor:
        """Scale latents using VAE config values."""
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        """Unscale latents for VAE decoding."""
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents / latents_std + latents_mean

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        """Calculate timestep shift for flow matching scheduler."""
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
