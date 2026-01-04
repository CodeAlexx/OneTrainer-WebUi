"""
Kandinsky 5 Model

Wrapper for Kandinsky 5 video generation model components.
Uses the official kandinsky-5-code DiffusionTransformer3D.
"""

from typing import Any, Optional
import torch
from torch import Tensor, nn

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelType import ModelType


class Kandinsky5Model(BaseModel):
    """
    Kandinsky 5 model container for training.

    Components:
    - transformer: DiffusionTransformer3D from kandinsky-5-code
    - vae: HunyuanVideo VAE (AutoencoderKLHunyuanVideo)
    - text_encoder_qwen: Qwen2.5-VL-7B-Instruct
    - text_encoder_clip: CLIP ViT-L/14
    """

    # Model components
    transformer: Any  # DiffusionTransformer3D
    transformer_lora: Any  # LoRAModuleWrapper for transformer
    vae: Any  # AutoencoderKLHunyuanVideo
    text_encoder_qwen: Any  # Qwen2_5_VLForConditionalGeneration
    text_encoder_clip: Any  # CLIPTextModel

    # Tokenizers/Processors
    processor_qwen: Any  # AutoProcessor for Qwen
    tokenizer_clip: Any  # CLIPTokenizer

    # Scheduler
    noise_scheduler: Any  # FlowMatchEulerDiscreteScheduler

    # Config and dtypes
    dit_config: dict
    transformer_train_dtype: Any
    text_encoder_autocast_context: Any
    transformer_autocast_context: Any

    # Offload conductor for memory management
    transformer_offload_conductor: Any

    # Text lengths
    text_len_qwen: int = 256  # From K5 Pro config
    text_len_clip: int = 77

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        # Initialize all components to None
        self.transformer = None
        self.vae = None
        self.text_encoder_qwen = None
        self.text_encoder_clip = None
        self.processor_qwen = None
        self.tokenizer_clip = None
        self.noise_scheduler = None

        self.dit_config = {}
        self.transformer_train_dtype = None
        self.transformer_offload_conductor = None
        self.transformer_lora = None

        # Default autocast contexts
        self.text_encoder_autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        self.transformer_autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    def vae_to(self, device: torch.device):
        """Move VAE to device."""
        if self.vae is not None:
            self.vae.to(device)

    def text_encoder_to(self, device: torch.device):
        """Move text encoders to device."""
        if self.text_encoder_qwen is not None:
            self.text_encoder_qwen.to(device)
        if self.text_encoder_clip is not None:
            self.text_encoder_clip.to(device)

    def transformer_to(self, device: torch.device):
        """Move transformer to device."""
        if self.transformer_offload_conductor is not None:
            self.transformer_offload_conductor.to(device)
        elif self.transformer is not None:
            self.transformer.to(device)

    def to(self, device: torch.device):
        """Move all components to device."""
        self.device = device
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
        return self

    def eval(self):
        """Set all components to eval mode."""
        if self.text_encoder_qwen is not None:
            self.text_encoder_qwen.eval()
        if self.text_encoder_clip is not None:
            self.text_encoder_clip.eval()
        if self.vae is not None:
            self.vae.eval()
        if self.transformer is not None:
            self.transformer.eval()
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        if self.transformer is not None:
            self.transformer.train(mode)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """Set requires_grad for all parameters."""
        if self.text_encoder_qwen is not None:
            self.text_encoder_qwen.requires_grad_(requires_grad)
        if self.text_encoder_clip is not None:
            self.text_encoder_clip.requires_grad_(requires_grad)
        if self.vae is not None:
            self.vae.requires_grad_(requires_grad)
        if self.transformer is not None:
            self.transformer.requires_grad_(requires_grad)
        return self

    def encode_text(self, texts: list[str], train_device: torch.device):
        """
        Encode text prompts using both Qwen2.5-VL and CLIP.

        Returns:
            text_embeds: Qwen embeddings [B, seq_len, 3584]
            pooled_embed: CLIP pooled embeddings [B, 768]
        """
        text_embeds = None
        pooled_embed = None

        # Qwen encoding
        if self.text_encoder_qwen is not None and self.processor_qwen is not None:
            text_embeds = self._encode_text_qwen(texts, train_device)

        # CLIP encoding
        if self.text_encoder_clip is not None and self.tokenizer_clip is not None:
            pooled_embed = self._encode_text_clip(texts, train_device)

        return text_embeds, pooled_embed

    def _encode_text_qwen(self, texts: list[str], train_device: torch.device) -> Tensor:
        """Encode text with Qwen2.5-VL."""
        # Prompt template from K5
        prompt_template = "\n".join([
            "<|im_start|>system\nYou are a prompt engineer. Describe the video in detail.",
            "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
            "Describe the location of the video, main characters or objects and their action.",
            "Describe the dynamism of the video and presented actions.",
            "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
            "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
            "Pay attention to the order of key actions shown in the scene.<|im_end|>",
            "<|im_start|>user\n{}<|im_end|>"
        ])
        crop_start = 129  # From K5 config for video

        full_texts = [prompt_template.format(t) for t in texts]
        max_length = self.text_len_qwen + crop_start

        inputs = self.processor_qwen(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(train_device)

        with torch.no_grad():
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder_qwen(
                    input_ids=inputs["input_ids"],
                    return_dict=True,
                    output_hidden_states=True,
                )
                embeds = outputs["hidden_states"][-1][:, crop_start:]

        return embeds

    def _encode_text_clip(self, texts: list[str], train_device: torch.device) -> Tensor:
        """Encode text with CLIP for pooled embeddings."""
        inputs = self.tokenizer_clip(
            texts,
            max_length=self.text_len_clip,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(train_device)

        with torch.no_grad():
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder_clip(**inputs)
                pooled_embed = outputs["pooler_output"]

        return pooled_embed

    def encode_text_qwen(self, tokens: Tensor, mask: Tensor, train_device: torch.device) -> Tensor:
        """
        Public wrapper for Qwen text encoding (for DataLoader compatibility).
        
        Args:
            tokens: Token IDs tensor
            mask: Attention mask tensor  
            train_device: Device to use
            
        Returns:
            Text embeddings tensor
        """
        # Decode tokens back to text if needed, then encode
        # For now, use the internal method with decoded text
        # This is a simplified implementation
        if self.text_encoder_qwen is None or self.processor_qwen is None:
            return None
            
        with torch.no_grad():
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder_qwen(
                    input_ids=tokens.to(self.text_encoder_qwen.device),
                    attention_mask=mask.to(self.text_encoder_qwen.device) if mask is not None else None,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Return last hidden state 
                embeds = outputs.hidden_states[-1]
                
        return embeds.to(train_device)

    def encode_text_clip(self, tokens: Tensor, mask: Tensor, train_device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Public wrapper for CLIP text encoding (for DataLoader compatibility).
        
        Args:
            tokens: Token IDs tensor
            mask: Attention mask tensor
            train_device: Device to use
            
        Returns:
            Tuple of (hidden_states, pooled_output)
        """
        if self.text_encoder_clip is None or self.tokenizer_clip is None:
            return None, None
            
        with torch.no_grad():
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder_clip(
                    input_ids=tokens.to(self.text_encoder_clip.device),
                    attention_mask=mask.to(self.text_encoder_clip.device) if mask is not None else None,
                )
                hidden_states = outputs.last_hidden_state
                pooled = outputs.pooler_output
                
        return hidden_states.to(train_device), pooled.to(train_device)


    def encode_video(self, video: Tensor) -> Tensor:
        """
        Encode video frames to latents using VAE.

        Args:
            video: [B, C, T, H, W] tensor in range [-1, 1]

        Returns:
            latents: [B, latent_channels, T', H', W'] tensor
        """
        if self.vae is None:
            raise ValueError("VAE not loaded")

        with torch.no_grad():
            posterior = self.vae.encode(video).latent_dist
            latents = posterior.sample()
            # K5 uses scaling_factor from VAE config
            latents = latents * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents: Tensor) -> Tensor:
        """
        Decode latents to video frames using VAE.

        Args:
            latents: [B, latent_channels, T', H', W'] tensor

        Returns:
            video: [B, C, T, H, W] tensor in range [-1, 1]
        """
        if self.vae is None:
            raise ValueError("VAE not loaded")

        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            video = self.vae.decode(latents).sample

        return video

    def get_transformer(self):
        """Get the transformer model."""
        return self.transformer

    def get_vae(self):
        """Get the VAE model."""
        return self.vae

    def get_text_encoders(self):
        """Get list of text encoders."""
        encoders = []
        if self.text_encoder_qwen is not None:
            encoders.append(self.text_encoder_qwen)
        if self.text_encoder_clip is not None:
            encoders.append(self.text_encoder_clip)
        return encoders

    def adapters(self) -> list:
        """Return list of active adapters (LoRA modules)."""
        # LoRA is applied directly to transformer, return empty list
        return []

    def get_components(self) -> dict:
        """Get all model components as a dictionary."""
        return {
            "transformer": self.transformer,
            "vae": self.vae,
            "text_encoder_qwen": self.text_encoder_qwen,
            "text_encoder_clip": self.text_encoder_clip,
            "processor_qwen": self.processor_qwen,
            "tokenizer_clip": self.tokenizer_clip,
            "noise_scheduler": self.noise_scheduler,
        }

    def get_trainable_blocks(self) -> nn.ModuleList:
        """Get the trainable transformer blocks for LoRA/checkpointing."""
        if self.transformer is not None:
            # Visual transformer blocks are the main trainable component
            if hasattr(self.transformer, 'visual_transformer_blocks'):
                return self.transformer.visual_transformer_blocks
        return nn.ModuleList()

    def parameters(self, recurse: bool = True):
        """Yield all parameters."""
        if self.transformer is not None:
            yield from self.transformer.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Yield all named parameters."""
        if self.transformer is not None:
            yield from self.transformer.named_parameters(prefix, recurse)

    @property
    def tokenizer_qwen(self):
        """Get the text tokenizer from the Qwen processor for data loading compatibility."""
        if self.processor_qwen is not None and hasattr(self.processor_qwen, 'tokenizer'):
            return self.processor_qwen.tokenizer
        return self.processor_qwen

    def create_rope_inputs(self, latents: Tensor, text_mask: Tensor = None):
        """
        Create RoPE position inputs for the DiffusionTransformer3D.

        The kandinsky-5-code expects:
        - visual_rope_pos: tuple of (pos_t, pos_h, pos_w) index tensors
        - text_rope_pos: LongTensor of position indices

        Args:
            latents: Latent tensor, can be [B, C, T, H, W] or [T, H, W, C]
            text_mask: Optional text attention mask

        Returns:
            visual_rope_pos: Tuple of 3 position index tensors
            text_rope_pos: LongTensor of text position indices
        """
        device = latents.device

        # Determine spatial dimensions based on latent shape
        if latents.dim() == 5:
            # [B, C, T, H, W] format
            _, _, lat_frames, lat_height, lat_width = latents.shape
        elif latents.dim() == 4:
            # [T, H, W, C] format (kandinsky-5-code expects this)
            lat_frames, lat_height, lat_width, _ = latents.shape
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")

        # Get patch size from config
        patch_size = self.dit_config.get('patch_size', (1, 2, 2))
        patched_t = max(1, lat_frames // patch_size[0])
        patched_h = lat_height // patch_size[1]
        patched_w = lat_width // patch_size[2]

        # Visual RoPE: tuple of position INDEX tensors (not full grids!)
        visual_rope_pos = (
            torch.arange(patched_t, device=device),
            torch.arange(patched_h, device=device),
            torch.arange(patched_w, device=device),
        )

        # Text RoPE: position indices as LongTensor
        if text_mask is not None:
            # Use actual text length from mask
            text_len = text_mask.shape[-1] if text_mask.dim() > 0 else self.text_len_qwen
        else:
            text_len = self.text_len_qwen

        text_rope_pos = torch.arange(text_len, device=device, dtype=torch.long)

        return visual_rope_pos, text_rope_pos

