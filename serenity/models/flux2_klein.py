"""
FLUX.2 Klein Model - Complete Implementation

Black Forest Labs FLUX.2 Klein models (4B and 9B variants).
Uses flow matching training with Qwen3 text encoder.

Features:
- FLUX_2_KLEIN_4B: 4B distilled model (5+20 blocks)
- FLUX_2_KLEIN_9B: 9B distilled model (8+24 blocks)
- FLUX_2_KLEIN_4B_BASE: 4B undistilled (better for training)
- FLUX_2_KLEIN_9B_BASE: 9B undistilled (better for training)

Key differences from FLUX.1:
- Single text encoder (Qwen3) instead of CLIP + T5
- Stacked layer embeddings [9, 18, 27] instead of last_hidden_state
- No guidance embeddings (Klein models ignore guidance)
- VAE uses batch normalization + pixel shuffle (32 → 128 channels)
- Different block counts and embedding dimensions
"""

from typing import Any, Optional, Dict, Tuple, TYPE_CHECKING, Union, List
from pathlib import Path

import torch
from torch import nn, Tensor

from serenity.core.interfaces import BaseModel, ModelType


# FLUX.2 Klein architecture constants
# Reference: SimpleTuner flux2/model.py and transformer config.json
FLUX2_KLEIN_CONFIG = {
    "klein-4b": {
        # Block structure
        "double_blocks": 5,
        "single_blocks": 20,
        "total_blocks": 25,
        "max_swappable": 24,
        # Transformer dimensions
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "inner_dim": 3072,  # 24 * 128
        "in_channels": 128,  # After VAE patchification
        # Text encoder (Qwen3)
        "joint_attention_dim": 7680,  # 3 * 2560 (stacked layers)
        "text_encoder_hidden": 2560,
        "text_encoder_layers": (9, 18, 27),
        # Config flags
        "guidance_embeds": False,  # Klein has NO guidance embeddings
        "mlp_ratio": 3.0,
        "rope_theta": 2000,
        "axes_dims_rope": (32, 32, 32, 32),
    },
    "klein-9b": {
        # Block structure
        "double_blocks": 8,
        "single_blocks": 24,
        "total_blocks": 32,
        "max_swappable": 31,
        # Transformer dimensions
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "inner_dim": 4096,  # 32 * 128
        "in_channels": 128,  # After VAE patchification
        # Text encoder (Qwen3)
        "joint_attention_dim": 12288,  # 3 * 4096 (stacked layers)
        "text_encoder_hidden": 4096,
        "text_encoder_layers": (9, 18, 27),
        # Config flags
        "guidance_embeds": False,  # Klein has NO guidance embeddings
        "mlp_ratio": 3.0,
        "rope_theta": 2000,
        "axes_dims_rope": (32, 32, 32, 32),
    },
}


class Flux2KleinModel(BaseModel):
    """
    FLUX.2 Klein model wrapper (4B and 9B variants).

    Architecture:
    - DiT-based transformer (MMDiT with joint attention)
    - Single text encoder: Qwen3 (bundled with model)
    - 32-channel VAE with batch normalization and pixel shuffle
    - Flow matching training objective (no guidance for Klein)

    Variants:
    - klein-4b: 25 blocks (5 double + 20 single), 7680 embedding dim
    - klein-9b: 32 blocks (8 double + 24 single), 12288 embedding dim
    - *-base: Undistilled variants for better training signal

    Attributes:
        transformer: MMDiT transformer module
        vae: 32-channel VAE with batch normalization
        text_encoder: Qwen3 text encoder
        tokenizer: Qwen3 tokenizer
        scheduler: Flow matching scheduler
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,  # Qwen3
        tokenizer: Any,  # Qwen3 tokenizer
        scheduler: Any,
        model_type: Optional["ModelType"] = None,
    ):
        self._transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = None  # Klein uses single encoder
        self.tokenizer = tokenizer
        self.tokenizer_2 = None
        self.scheduler = scheduler
        self._model_type = model_type

        # Model-specific settings
        self.max_sequence_length = 512

        # Determine variant config
        self._variant_config = self._get_variant_config()

        # Training state
        self.train_progress = None
        self._lora_applied = False
        self._lora_config = None

        # Block swap offloaders
        self.block_swap_offloaders = []

        # Initialize embedding lists
        self.embeddings = []
        self.output_embeddings = []

    def _get_variant_config(self) -> dict:
        """Get configuration for current variant."""
        # ModelType imported at top of file

        if self._model_type in (ModelType.FLUX_2_KLEIN_4B, ModelType.FLUX_2_KLEIN_4B_BASE):
            return FLUX2_KLEIN_CONFIG["klein-4b"]
        elif self._model_type in (ModelType.FLUX_2_KLEIN_9B, ModelType.FLUX_2_KLEIN_9B_BASE):
            return FLUX2_KLEIN_CONFIG["klein-9b"]
        else:
            # Default to 9B config
            return FLUX2_KLEIN_CONFIG["klein-9b"]

    @property
    def model_type(self) -> "ModelType":
        """Return the model type enum."""
        if self._model_type is not None:
            return self._model_type
        # ModelType imported at top of file
        return ModelType.FLUX_2_KLEIN_9B

    @property
    def transformer(self) -> nn.Module:
        """Return the main trainable module (transformer).

        This is the module where adapters (LoRA, LyCORIS) will be injected.
        """
        return self._transformer

    def is_4b(self) -> bool:
        """Check if this is the 4B variant."""
        # ModelType imported at top of file
        return self._model_type in (ModelType.FLUX_2_KLEIN_4B, ModelType.FLUX_2_KLEIN_4B_BASE)

    def is_9b(self) -> bool:
        """Check if this is the 9B variant."""
        # ModelType imported at top of file
        return self._model_type in (ModelType.FLUX_2_KLEIN_9B, ModelType.FLUX_2_KLEIN_9B_BASE)

    def is_base(self) -> bool:
        """Check if this is an undistilled (base) variant."""
        # ModelType imported at top of file
        return self._model_type in (ModelType.FLUX_2_KLEIN_4B_BASE, ModelType.FLUX_2_KLEIN_9B_BASE)

    def is_distilled(self) -> bool:
        """Check if this is a distilled variant."""
        return not self.is_base()

    @property
    def total_blocks(self) -> int:
        """Total number of transformer blocks."""
        return self._variant_config["total_blocks"]

    @property
    def max_swappable_blocks(self) -> int:
        """Maximum number of blocks that can be swapped to CPU."""
        return self._variant_config["max_swappable"]

    @property
    def embedding_dim(self) -> int:
        """Text embedding dimension (stacked layers). Alias for joint_attention_dim."""
        return self._variant_config["joint_attention_dim"]

    @property
    def joint_attention_dim(self) -> int:
        """Text embedding dimension (stacked Qwen3 layers)."""
        return self._variant_config["joint_attention_dim"]

    @property
    def inner_dim(self) -> int:
        """Transformer hidden dimension (num_heads * head_dim)."""
        return self._variant_config["inner_dim"]

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads in transformer."""
        return self._variant_config["num_attention_heads"]

    @property
    def attention_head_dim(self) -> int:
        """Dimension of each attention head."""
        return self._variant_config["attention_head_dim"]

    @property
    def in_channels(self) -> int:
        """Input channels (after VAE patchification)."""
        return self._variant_config["in_channels"]

    @property
    def guidance_embeds(self) -> bool:
        """Whether model has guidance embeddings (False for Klein)."""
        return self._variant_config["guidance_embeds"]

    @property
    def text_encoder_layers(self) -> Tuple[int, int, int]:
        """Layers to stack for text embeddings."""
        return self._variant_config["text_encoder_layers"]

    @property
    def text_encoder_hidden(self) -> int:
        """Hidden dimension of the Qwen3 text encoder."""
        return self._variant_config["text_encoder_hidden"]

    # =========================================================================
    # Timestep Shift Calculation
    # =========================================================================

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        """Calculate timestep shift based on resolution.

        Uses scheduler config if available, otherwise uses FlowMatchEulerDiscreteScheduler
        defaults: base_shift=0.5, max_shift=1.15, base_seq_len=256, max_seq_len=4096.

        Args:
            latent_height: Height in latent space (unpatchified)
            latent_width: Width in latent space (unpatchified)

        Returns:
            Exponential shift value (math.exp(mu)) for timestep sampling
        """
        import math

        # Get scheduler config or use defaults
        if self.scheduler is not None and hasattr(self.scheduler, 'config'):
            cfg = self.scheduler.config
            base_seq_len = getattr(cfg, 'base_image_seq_len', 256)
            max_seq_len = getattr(cfg, 'max_image_seq_len', 4096)
            base_shift = getattr(cfg, 'base_shift', 0.5)
            max_shift = getattr(cfg, 'max_shift', 1.15)
        else:
            # FlowMatchEulerDiscreteScheduler defaults
            base_seq_len = 256
            max_seq_len = 4096
            base_shift = 0.5
            max_shift = 1.15

        patch_size = 2
        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b

        return math.exp(mu)

    # =========================================================================
    # Text Encoding (Qwen3 with stacked layers)
    # =========================================================================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        max_sequence_length: int = 512,
    ) -> Tensor:
        """
        Encode prompt using Qwen3 with stacked layer outputs.

        Unlike FLUX.1 which uses CLIP+T5, FLUX.2 Klein uses a single Qwen3
        encoder and stacks hidden states from layers [9, 18, 27] to create
        the embedding.

        Args:
            prompt: Single string or list of strings
            device: Target device
            max_sequence_length: Maximum sequence length (default 512)

        Returns:
            prompt_embeds: Stacked embeddings [B, L, D] where D = 3 * hidden_dim
        """
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Apply Qwen3 chat template before tokenization (required for Klein)
        # Klein uses enable_thinking=False (unlike Z-Image which uses True)
        formatted_prompts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Klein uses False, Z-Image uses True
            )
            formatted_prompts.append(formatted)

        # Tokenize the chat-formatted prompts
        text_inputs = self.tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Get hidden states from all layers
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Stack layers [9, 18, 27]
        layers = self.text_encoder_layers
        hidden_states = outputs.hidden_states

        # Stack: [B, 3, L, D] then reshape to [B, L, 3*D]
        stacked = torch.stack([
            hidden_states[layers[0]],
            hidden_states[layers[1]],
            hidden_states[layers[2]],
        ], dim=1)

        # Reshape from [B, 3, L, D] to [B, L, 3*D]
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

        return prompt_embeds

    # =========================================================================
    # VAE Encoding/Decoding with Patchification
    # =========================================================================

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """
        Encode image to latent space.

        FLUX.2 VAE outputs 32 channels which are then patchified to 128.
        """
        with torch.no_grad():
            latent = self.vae.encode(pixel_values).latent_dist.sample()
            # Note: scaling handled separately via patchify/normalize
        return latent

    def decode_latent(self, latent: Tensor) -> Tensor:
        """
        Decode latent to image.

        Expects unpatchified latents (32 channels).
        """
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image

    def patchify_latents(self, latents: Tensor) -> Tensor:
        """
        Pixel-shuffle latents: (B, 32, H, W) → (B, 128, H/2, W/2).

        This matches FLUX.2's VAE output format where 32 channels
        are shuffled into 128 channels with 2x2 spatial reduction.

        Args:
            latents: VAE output [B, 32, H, W]

        Returns:
            Patchified latents [B, 128, H/2, W/2]
        """
        b, c, h, w = latents.shape

        if c == 128:
            # Already patchified
            return latents

        if c != 32:
            raise ValueError(f"Expected 32 channels for patchification, got {c}")

        # Reshape: (B, 32, H, W) → (B, 32, H/2, 2, W/2, 2)
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)

        # Permute: (B, 32, H/2, 2, W/2, 2) → (B, 32, 2, 2, H/2, W/2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)

        # Flatten channels: (B, 32*2*2, H/2, W/2) = (B, 128, H/2, W/2)
        latents = latents.reshape(b, c * 4, h // 2, w // 2)

        return latents

    def unpatchify_latents(self, latents: Tensor) -> Tensor:
        """
        Reverse pixel-shuffle: (B, 128, H/2, W/2) → (B, 32, H, W).

        Args:
            latents: Patchified latents [B, 128, H/2, W/2]

        Returns:
            Unpatchified latents [B, 32, H, W]
        """
        b, c, h, w = latents.shape

        if c == 32:
            # Already unpatchified
            return latents

        if c != 128:
            raise ValueError(f"Expected 128 channels for unpatchification, got {c}")

        # Reshape: (B, 128, H/2, W/2) → (B, 32, 2, 2, H/2, W/2)
        latents = latents.view(b, 32, 2, 2, h, w)

        # Permute: (B, 32, 2, 2, H/2, W/2) → (B, 32, H/2, 2, W/2, 2)
        latents = latents.permute(0, 1, 4, 2, 5, 3)

        # Flatten spatial: (B, 32, H, W)
        latents = latents.reshape(b, 32, h * 2, w * 2)

        return latents

    def normalize_latents(self, latents: Tensor) -> Tensor:
        """
        Apply VAE batch normalization statistics.

        FLUX.2 VAE uses batch normalization, so we need to normalize
        latents using the running mean and variance from the VAE.

        For Klein (32-channel VAE → 128 patchified):
        - If batch norm has 32 channels: unpatchify, normalize, repatchify
        - If batch norm has 64 channels: wrong VAE, skip normalization
        - If batch norm has 128 channels: apply directly

        Args:
            latents: Latents [B, C, H, W] - may be patchified (128) or not (32)

        Returns:
            Normalized latents [B, C, H, W]
        """
        if self.vae is None:
            return latents

        # Check if VAE has batch norm statistics
        bn = getattr(self.vae, "bn", None)
        if bn is None:
            # Try to find batch norm in encoder
            if hasattr(self.vae, "encoder") and hasattr(self.vae.encoder, "bn"):
                bn = self.vae.encoder.bn

        if bn is None:
            # No batch norm, return as-is
            return latents

        latent_channels = latents.shape[1]
        bn_channels = bn.running_mean.shape[0]

        # Check for dimension mismatch
        if bn_channels != latent_channels:
            if latent_channels == 128 and bn_channels == 32:
                # Klein VAE: apply batch norm in unpatchified space
                unpatchified = self.unpatchify_latents(latents)
                mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                var = bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                eps = bn.eps
                normalized = (unpatchified - mean) / torch.sqrt(var + eps)
                return self.patchify_latents(normalized)
            else:
                # Dimension mismatch we can't handle (wrong VAE loaded)
                # Skip normalization to avoid crash
                return latents

        # Get running statistics
        mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        var = bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        eps = bn.eps

        # Normalize
        return (latents - mean) / torch.sqrt(var + eps)

    def denormalize_latents(self, latents: Tensor) -> Tensor:
        """Reverse batch norm normalization.

        For Klein (32-channel VAE → 128 patchified):
        - If batch norm has 32 channels: unpatchify, denormalize, repatchify
        - If batch norm has 64 channels: wrong VAE, skip denormalization
        - If batch norm has 128 channels: apply directly
        """
        if self.vae is None:
            return latents

        bn = getattr(self.vae, "bn", None)
        if bn is None and hasattr(self.vae, "encoder"):
            bn = getattr(self.vae.encoder, "bn", None)

        if bn is None:
            return latents

        latent_channels = latents.shape[1]
        bn_channels = bn.running_mean.shape[0]

        # Check for dimension mismatch
        if bn_channels != latent_channels:
            if latent_channels == 128 and bn_channels == 32:
                # Klein VAE: apply batch norm in unpatchified space
                unpatchified = self.unpatchify_latents(latents)
                mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                var = bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                eps = bn.eps
                denormalized = unpatchified * torch.sqrt(var + eps) + mean
                return self.patchify_latents(denormalized)
            else:
                # Dimension mismatch we can't handle (wrong VAE loaded)
                # Skip denormalization to avoid crash
                return latents

        mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        var = bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        eps = bn.eps

        return latents * torch.sqrt(var + eps) + mean

    # =========================================================================
    # Latent Packing/Unpacking for Transformer
    # =========================================================================

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Prepare random latents for generation."""
        # FLUX.2 VAE uses 8x spatial compression (standard AutoencoderKL)
        # When using AutoencoderKLFlux2, this should be 16x, but
        # fallback AutoencoderKL uses 8x
        # VAE produces H/8, W/8 @ 32 channels, then patchify halves spatial
        vae_scale_factor = 8
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        channels = 32  # Before patchification

        shape = (batch_size, channels, latent_height, latent_width)
        latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        return latents

    def prepare_latent_image_ids(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Prepare latent image IDs for FLUX.2-style positional encoding.

        FLUX.2 uses 4D position IDs: (T, H, W, L) where:
        - T: Time coordinate (0 for generation, >0 for conditioning)
        - H: Height position
        - W: Width position
        - L: Patch/local position (within 2x2 patch)

        Args:
            height: Patchified latent height
            width: Patchified latent width
            device: Target device
            dtype: Target dtype

        Returns:
            Image IDs tensor [H*W, 4]
        """
        # Create grid of positions
        h_pos = torch.arange(height, device=device, dtype=dtype)
        w_pos = torch.arange(width, device=device, dtype=dtype)

        # Create meshgrid
        h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")

        # Stack into [H, W, 4] with T=0, L=0
        t_coord = torch.zeros_like(h_grid)
        l_coord = torch.zeros_like(h_grid)

        image_ids = torch.stack([t_coord, h_grid, w_grid, l_coord], dim=-1)

        # Flatten to [H*W, 4]
        image_ids = image_ids.reshape(-1, 4)

        return image_ids

    def pack_latents(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pack 2D latents into sequence format for transformer.

        Transforms [B, C, H, W] -> [B, H*W, C]

        Args:
            latents: Patchified latents [B, 128, H, W]

        Returns:
            packed: Packed latents [B, H*W, 128]
            img_ids: Position IDs [H*W, 4]
        """
        batch_size, channels, height, width = latents.shape

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        packed = latents.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        # Create position IDs
        img_ids = self.prepare_latent_image_ids(height, width, latents.device, latents.dtype)

        return packed, img_ids

    def unpack_latents(self, packed: Tensor, height: int, width: int) -> Tensor:
        """
        Unpack sequence latents back to 2D format.

        Transforms [B, H*W, C] -> [B, C, H, W]

        Args:
            packed: Packed latents [B, H*W, C]
            height: Target height
            width: Target width

        Returns:
            Unpacked latents [B, C, H, W]
        """
        batch_size, seq_len, channels = packed.shape

        # Reshape back to 2D: [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        unpacked = packed.view(batch_size, height, width, channels)
        unpacked = unpacked.permute(0, 3, 1, 2)

        return unpacked

    def pack_text(self, text_embeds: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pack text embeddings with position IDs.

        Text always uses T=1 to distinguish from image tokens.

        Args:
            text_embeds: Text embeddings [B, L, D]

        Returns:
            text: Text embeddings [B, L, D] (unchanged)
            txt_ids: Position IDs [L, 4] with T=1
        """
        batch_size, seq_len, embed_dim = text_embeds.shape

        # Create text position IDs with T=1 (different from image T=0)
        t_coord = torch.ones(seq_len, device=text_embeds.device, dtype=text_embeds.dtype)
        h_coord = torch.ones(seq_len, device=text_embeds.device, dtype=text_embeds.dtype)
        w_coord = torch.ones(seq_len, device=text_embeds.device, dtype=text_embeds.dtype)
        l_coord = torch.arange(seq_len, device=text_embeds.device, dtype=text_embeds.dtype)

        txt_ids = torch.stack([t_coord, h_coord, w_coord, l_coord], dim=-1)

        return text_embeds, txt_ids

    def pooled_text_projection(self, text_embeds: Tensor) -> Tensor:
        """
        Create pooled text projections compatible with transformer time-text embedding.

        Some diffusers builds expect `pooled_projections` to have a smaller feature
        size (for example 768) even when encoder hidden states are much larger.
        """

        pooled = text_embeds.mean(dim=1)

        expected_dim = None
        time_text_embed = getattr(self.transformer, "time_text_embed", None)
        text_embedder = getattr(time_text_embed, "text_embedder", None)
        linear_1 = getattr(text_embedder, "linear_1", None)
        if linear_1 is not None and hasattr(linear_1, "in_features"):
            expected_dim = int(linear_1.in_features)

        if expected_dim is None or expected_dim <= 0 or pooled.shape[-1] == expected_dim:
            return pooled

        if pooled.shape[-1] > expected_dim:
            return pooled[..., :expected_dim]

        pad_width = expected_dim - pooled.shape[-1]
        return torch.nn.functional.pad(pooled, (0, pad_width))

    # =========================================================================
    # Block Swapping (Memory Optimization)
    # =========================================================================

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Enable block swapping for VRAM efficiency.

        Args:
            blocks_to_swap: Number of blocks to swap (capped at max_swappable)
            device: Target device (default: cuda)
        """
        try:
            from ..training.block_swap import ModelOffloader
        except ImportError:
            print("  ⚠️ Block swap not available - module not implemented")
            return

        if device is None:
            device = torch.device('cuda')

        # Cap at maximum swappable
        blocks_to_swap = min(blocks_to_swap, self.max_swappable_blocks)

        transformer = self.transformer
        self.block_swap_offloaders = []

        # Unwrap transformer if wrapped
        unwrapped_transformer = transformer
        while hasattr(unwrapped_transformer, 'base_model') or hasattr(unwrapped_transformer, 'model'):
            if hasattr(unwrapped_transformer, 'base_model'):
                unwrapped_transformer = unwrapped_transformer.base_model
            elif hasattr(unwrapped_transformer, 'model'):
                unwrapped_transformer = unwrapped_transformer.model

        # FLUX.2 has transformer_blocks and single_transformer_blocks
        mmdit_blocks = getattr(unwrapped_transformer, 'transformer_blocks', None)
        single_blocks = getattr(unwrapped_transformer, 'single_transformer_blocks', None)

        # Handle double-stream blocks
        if mmdit_blocks is not None and blocks_to_swap > 0:
            num = len(mmdit_blocks)
            swap = min(blocks_to_swap, max(0, num - 1))
            if swap > 0:
                offloader = ModelOffloader(
                    model=transformer,
                    blocks_to_swap=swap,
                    device=device,
                    block_attr='transformer_blocks',
                    supports_backward=True,
                    debug=False,
                )
                self.block_swap_offloaders.append(offloader)
                blocks_to_swap -= swap

        # Handle single-stream blocks
        if single_blocks is not None and blocks_to_swap > 0:
            num = len(single_blocks)
            swap = min(blocks_to_swap, max(0, num - 1))
            if swap > 0:
                offloader = ModelOffloader(
                    model=transformer,
                    blocks_to_swap=swap,
                    device=device,
                    block_attr='single_transformer_blocks',
                    supports_backward=True,
                    debug=False,
                )
                self.block_swap_offloaders.append(offloader)

        # Move non-block params to GPU
        if self.block_swap_offloaders:
            if mmdit_blocks is not None:
                unwrapped_transformer.transformer_blocks = None
            if single_blocks is not None:
                unwrapped_transformer.single_transformer_blocks = None

            transformer.to(device)

            if mmdit_blocks is not None:
                unwrapped_transformer.transformer_blocks = mmdit_blocks
            if single_blocks is not None:
                unwrapped_transformer.single_transformer_blocks = single_blocks
        else:
            transformer.to(device)

        # Enable all offloaders
        for offloader in self.block_swap_offloaders:
            offloader.enable()

        print(f"  🔄 FLUX.2 Klein block swap enabled: {len(self.block_swap_offloaders)} offloaders")

    def prepare_block_swap_training(self) -> None:
        """Prepare for training with block swapping enabled."""
        if hasattr(self, 'block_swap_offloaders'):
            for offloader in self.block_swap_offloaders:
                offloader.prepare_for_forward()

    def disable_block_swap(self) -> None:
        """Disable block swapping and restore original forwards."""
        if hasattr(self, 'block_swap_offloaders'):
            for offloader in self.block_swap_offloaders:
                offloader.disable()
                offloader.cleanup()
            self.block_swap_offloaders = []

    # =========================================================================
    # Device Movement
    # =========================================================================

    def to(
        self,
        device: Union[str, torch.device],
        training_only: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Flux2KleinModel":
        """Move model components to device."""
        if isinstance(device, str):
            device = torch.device(device)

        self._train_device = device
        if dtype is not None:
            self._train_dtype = dtype

        # Move transformer (respecting block swap)
        if hasattr(self, 'block_swap_offloaders') and self.block_swap_offloaders:
            # Block swap active - careful movement
            transformer = self.transformer
            unwrapped = transformer
            while hasattr(unwrapped, 'base_model') or hasattr(unwrapped, 'model'):
                if hasattr(unwrapped, 'base_model'):
                    unwrapped = unwrapped.base_model
                elif hasattr(unwrapped, 'model'):
                    unwrapped = unwrapped.model

            mmdit = getattr(unwrapped, 'transformer_blocks', None)
            single = getattr(unwrapped, 'single_transformer_blocks', None)

            if mmdit is not None:
                unwrapped.transformer_blocks = None
            if single is not None:
                unwrapped.single_transformer_blocks = None

            transformer.to(device)

            if mmdit is not None:
                unwrapped.transformer_blocks = mmdit
            if single is not None:
                unwrapped.single_transformer_blocks = single
        else:
            self.transformer.to(device)

        if dtype is not None:
            self.transformer.to(dtype=dtype)

        # Move other components
        if not training_only:
            if self.vae is not None:
                self.vae.to(device)
            if self.text_encoder is not None:
                self.text_encoder.to(device)

        # Move embeddings
        for emb in self.embeddings:
            emb.to(device, dtype)
        for emb in self.output_embeddings:
            emb.to(device, dtype)

        return self

    # =========================================================================
    # Abstract Method Implementations (Required by BaseModel)
    # =========================================================================

    def get_trainable_parameters(self):
        """Return parameters to optimize (transformer parameters for LoRA).

        Returns:
            Iterator over transformer parameters that should be passed to optimizer.
        """
        return self.transformer.parameters()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for Flux 2 Klein training.

        Implements flow matching training for Klein models:
        1. Get latent from batch (32 channels from Klein VAE)
        2. Patchify latents (32 → 128 channels via pixel shuffle)
        3. Normalize using VAE batch norm statistics
        4. Sample timestep with resolution-dependent shift
        5. Create noise and interpolate with scaled latent
        6. Pack latents for transformer
        7. Create position IDs
        8. Transformer forward (no guidance for Klein)
        9. Unpack output
        10. Calculate flow = noise - scaled_latent
        11. Return dict for Trainer

        Args:
            batch: Dict with:
                - 'latent_image': VAE-encoded image latents [B, 32, H, W]
                - 'text_encoder_hidden_state': Pre-computed embeddings (optional)
                - 'tokens', 'tokens_mask': For on-the-fly encoding

        Returns:
            Dict with keys: 'loss_type', 'timestep', 'predicted', 'target'
        """
        # 1. Get latent from batch
        latent = batch["latent_image"]
        batch_size = latent.shape[0]
        device = latent.device
        dtype = latent.dtype

        # 2. Patchify latents: [B, 32, H, W] → [B, 128, H/2, W/2]
        patchified_latent = self.patchify_latents(latent.float())
        latent_height = patchified_latent.shape[-2]
        latent_width = patchified_latent.shape[-1]

        # 3. Normalize using VAE batch norm
        scaled_latent = self.normalize_latents(patchified_latent)

        # 4. Sample timestep with resolution-dependent shift
        # Shift formula:
        # t_scaled = u * N, then t_shifted = N * shift * t_scaled / ((shift - 1) * t_scaled + N)
        shift = self.calculate_timestep_shift(latent_height * 2, latent_width * 2)
        num_train_timesteps = 1000  # Default scheduler timesteps

        u = torch.rand(batch_size, device=device, dtype=dtype)
        # Apply proper shift transformation
        t_scaled = u * num_train_timesteps
        timestep = num_train_timesteps * shift * t_scaled / ((shift - 1) * t_scaled + num_train_timesteps)
        timestep_int = timestep.long().clamp(0, num_train_timesteps - 1)

        # 5. Create noise and add to scaled latent
        # sigma = (timestep + 1) / num_train_timesteps (never 0)
        noise = torch.randn_like(scaled_latent)
        sigma = ((timestep_int.float() + 1) / num_train_timesteps).view(-1, 1, 1, 1)
        noisy_latent = sigma * noise + (1 - sigma) * scaled_latent

        # 6. Pack latents for transformer: [B, 128, H, W] → [B, H*W, 128]
        packed_latent, img_ids = self.pack_latents(noisy_latent)

        # 7. Get or compute text embeddings
        if "text_encoder_hidden_state" in batch:
            text_encoder_output = batch["text_encoder_hidden_state"]
        else:
            # On-the-fly encoding
            text = batch.get("prompt", batch.get("caption", [""] * batch_size))
            if isinstance(text, str):
                text = [text]
            text_encoder_output = self.encode_prompt(text)
            text_encoder_output = text_encoder_output.to(device=device, dtype=dtype)

        # Pack text embeddings
        packed_text, txt_ids = self.pack_text(text_encoder_output)
        pooled_projections = self.pooled_text_projection(packed_text)
        guidance = torch.ones(batch_size, device=device, dtype=self.transformer.dtype)

        # 8. Transformer forward (NO guidance for Klein)
        transformer_output = self.transformer(
            hidden_states=packed_latent.to(dtype=self.transformer.dtype),
            timestep=timestep_int / 1000,  # discrete timestep normalized
            guidance=guidance,
            encoder_hidden_states=packed_text.to(dtype=self.transformer.dtype),
            pooled_projections=pooled_projections.to(dtype=self.transformer.dtype),
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=True,
        ).sample

        # 9. Unpack output: [B, H*W, 128] → [B, 128, H, W]
        predicted = self.unpack_latents(transformer_output, latent_height, latent_width)

        # 10. Calculate flow = noise - scaled_latent
        target = noise - scaled_latent

        # Cast target to match predicted dtype
        target = target.to(dtype=predicted.dtype)

        # 11. Return dict for Trainer
        return {
            "loss_type": "flow",
            "timestep": timestep_int,
            "predicted": predicted,
            "target": target,
        }


class Flux2KleinModelLoader:
    """Loads FLUX.2 Klein models from HuggingFace or local path - NEVER downloads."""

    @staticmethod
    def load(
        model_path: str,
        model_type: Optional["ModelType"] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        use_flux2_transformer: bool = True,
    ) -> Flux2KleinModel:
        """
        Load FLUX.2 Klein model from local path or HF cache.

        SAFETY: Will NEVER download. Fails if model not found.

        Args:
            model_path: HF model ID or local path
            model_type: FLUX_2_KLEIN_* variant (auto-detected if None)
            dtype: Weight dtype
            device: Target device
            use_flux2_transformer: If True (default), use custom Flux2Transformer2DModel
                                   which is compatible with FLUX.2 Klein weights.
                                   If False, use diffusers FluxTransformer2DModel (FLUX.1).

        Returns:
            Flux2KleinModel instance
        """
        from diffusers import DiffusionPipeline
        from transformers import AutoTokenizer, AutoModel
        # ModelType imported at top of file

        loader_type = "Flux2Transformer2DModel" if use_flux2_transformer else "diffusers"
        print(f"  Loading FLUX.2 Klein from {model_path} (transformer: {loader_type})...")

        # Auto-detect model type if not specified
        if model_type is None:
            path_lower = model_path.lower()
            if "4b" in path_lower:
                if "base" in path_lower:
                    model_type = ModelType.FLUX_2_KLEIN_4B_BASE
                else:
                    model_type = ModelType.FLUX_2_KLEIN_4B
            else:
                if "base" in path_lower:
                    model_type = ModelType.FLUX_2_KLEIN_9B_BASE
                else:
                    model_type = ModelType.FLUX_2_KLEIN_9B

        # Try loading via DiffusionPipeline first
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
            )

            # Extract Qwen3 text encoder from pipeline or load separately
            text_encoder = getattr(pipe, 'text_encoder', None)
            tokenizer = getattr(pipe, 'tokenizer', None)

            # If text encoder not in pipeline, try loading from subfolder
            if text_encoder is None:
                text_encoder_path = Path(model_path) / "text_encoder"
                if text_encoder_path.exists():
                    print(f"    Loading Qwen3 from {text_encoder_path}...")
                    text_encoder = AutoModel.from_pretrained(
                        str(text_encoder_path),
                        torch_dtype=dtype,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(text_encoder_path),
                        local_files_only=True,
                        trust_remote_code=True,
                    )

            model = Flux2KleinModel(
                transformer=pipe.transformer,
                vae=pipe.vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=pipe.scheduler,
                model_type=model_type,
            )

            del pipe
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Pipeline loading failed: {e}")
            print(f"    Attempting component-by-component load...")
            model = Flux2KleinModelLoader._load_components(
                model_path, model_type, dtype, use_flux2_transformer
            )

        variant_name = {
            ModelType.FLUX_2_KLEIN_4B: "Klein 4B",
            ModelType.FLUX_2_KLEIN_9B: "Klein 9B",
            ModelType.FLUX_2_KLEIN_4B_BASE: "Klein 4B Base",
            ModelType.FLUX_2_KLEIN_9B_BASE: "Klein 9B Base",
        }.get(model_type, "Klein")

        print(f"  ✅ Loaded FLUX.2 {variant_name}")
        return model

    @staticmethod
    def _load_components(
        model_path: str,
        model_type: "ModelType",
        dtype: torch.dtype,
        use_flux2_transformer: bool = True,
    ) -> Flux2KleinModel:
        """Load FLUX.2 Klein from separate component files."""
        from pathlib import Path
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, AutoModel
        from diffusers import FlowMatchEulerDiscreteScheduler

        path = Path(model_path)

        # Load transformer
        transformer_path = path / "transformer"
        if not transformer_path.exists():
            transformer_path = path / "model"

        print(f"    Loading transformer from {transformer_path}...")

        if use_flux2_transformer:
            # Use custom Flux2Transformer2DModel (compatible with FLUX.2 weights)
            from .flux2_transformer import Flux2Transformer2DModel
            transformer = Flux2Transformer2DModel.from_pretrained_klein(
                str(transformer_path),
                torch_dtype=dtype,
                device="cpu",
            )
        else:
            # Use diffusers FluxTransformer2DModel (FLUX.1 architecture)
            try:
                from diffusers import FluxTransformer2DModel
                transformer = FluxTransformer2DModel.from_pretrained(
                    str(transformer_path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            except Exception:
                # Try loading as safetensors
                st_file = list(transformer_path.glob("*.safetensors"))
                if st_file:
                    state_dict = load_file(str(st_file[0]))
                    from diffusers import FluxTransformer2DModel
                    transformer = FluxTransformer2DModel()
                    transformer.load_state_dict(state_dict, strict=False)
                    transformer = transformer.to(dtype)
                else:
                    raise FileNotFoundError(f"No transformer found at {transformer_path}")

        # Load VAE - MUST use AutoencoderKLFlux2 for correct batch norm structure
        vae_path = path / "vae"
        print(f"    Loading VAE from {vae_path}...")
        try:
            from diffusers import AutoencoderKLFlux2
            vae = AutoencoderKLFlux2.from_pretrained(
                str(vae_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as e:
            print(f"      Warning: Could not load VAE with AutoencoderKLFlux2: {e}")
            # Fallback to regular AutoencoderKL if Flux2-specific not available
            try:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    str(vae_path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                print("      Warning: Using fallback AutoencoderKL - batch norm may differ")
            except Exception as e2:
                print(f"      Warning: Could not load VAE: {e2}")
                vae = None

        # Load Qwen3 text encoder
        text_encoder_path = path / "text_encoder"
        tokenizer_path = path / "tokenizer"
        print(f"    Loading Qwen3 from {text_encoder_path}...")
        try:
            text_encoder = AutoModel.from_pretrained(
                str(text_encoder_path),
                torch_dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"      Warning: Could not load text encoder: {e}")
            text_encoder = None

        # Load tokenizer from separate folder
        print(f"    Loading tokenizer from {tokenizer_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"      Warning: Could not load tokenizer: {e}")
            tokenizer = None

        # Create scheduler
        scheduler = FlowMatchEulerDiscreteScheduler()

        return Flux2KleinModel(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            model_type=model_type,
        )

    @staticmethod
    def load_transformer_only(
        transformer_path: str,
        model_type: Optional["ModelType"] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_flux2_transformer: bool = True,
    ) -> Flux2KleinModel:
        """
        Load ONLY transformer for training (after caching VAE/text).

        This is the VRAM-efficient path when using cached latents and embeddings.

        Args:
            transformer_path: Path to transformer folder or safetensors file
            model_type: FLUX_2_KLEIN_* variant (auto-detected if None)
            dtype: Weight dtype
            use_flux2_transformer: If True (default), use custom Flux2Transformer2DModel
                                   which is compatible with FLUX.2 Klein weights.
                                   If False, use diffusers FluxTransformer2DModel (FLUX.1).
        """
        from pathlib import Path
        from safetensors.torch import load_file
        from diffusers import FlowMatchEulerDiscreteScheduler
        # ModelType imported at top of file

        path = Path(transformer_path)
        if not path.exists():
            raise FileNotFoundError(f"Transformer not found: {path}")

        loader_type = "Flux2Transformer2DModel" if use_flux2_transformer else "diffusers"
        print(f"  Loading FLUX.2 Klein transformer only: {path.name} ({loader_type})")
        print(f"    (VAE/text encoder NOT loaded - using cache)")

        # Determine model type
        if model_type is None:
            model_type = ModelType.FLUX_2_KLEIN_9B

        # Load transformer
        if use_flux2_transformer:
            # Use custom Flux2Transformer2DModel (compatible with FLUX.2 weights)
            from .flux2_transformer import Flux2Transformer2DModel
            transformer = Flux2Transformer2DModel.from_pretrained_klein(
                str(path),
                torch_dtype=dtype,
                device="cpu",
            )
        else:
            # Use diffusers FluxTransformer2DModel (FLUX.1 architecture)
            from diffusers import FluxTransformer2DModel
            if path.is_dir():
                transformer = FluxTransformer2DModel.from_pretrained(
                    str(path),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            else:
                state_dict = load_file(str(path))
                transformer = FluxTransformer2DModel()
                transformer.load_state_dict(state_dict, strict=False)
                transformer = transformer.to(dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()

        model = Flux2KleinModel(
            transformer=transformer,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            scheduler=scheduler,
            model_type=model_type,
        )

        variant = "4B" if model_type in (ModelType.FLUX_2_KLEIN_4B, ModelType.FLUX_2_KLEIN_4B_BASE) else "9B"
        print(f"  ✅ FLUX.2 Klein {variant} transformer loaded (cache mode)")
        return model


class Flux2KleinSampler:
    """
    Generates samples using FLUX.2 Klein models.

    Handles both distilled (4 steps) and base (50 steps) variants.
    Uses flow matching sampling with Euler integration.

    Key differences from FLUX.1:
    - Single Qwen3 text encoder (stacked layers)
    - 32-channel VAE with patchification (32 → 128 channels)
    - NO guidance embeddings (Klein ignores guidance_scale)
    """

    def __init__(
        self,
        model: Flux2KleinModel,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = -1,
    ) -> Tensor:
        """
        Generate a sample image using flow matching.

        Args:
            prompt: Text prompt
            width: Image width (must be divisible by 16)
            height: Image height (must be divisible by 16)
            num_steps: Number of diffusion steps (auto-detected based on variant)
            guidance_scale: CFG scale (ignored for Klein, kept for API compatibility)
            seed: Random seed (-1 for random)

        Returns:
            Image tensor [1, 3, H, W] in [0, 1] range
        """
        # Auto-detect steps based on variant
        if num_steps is None:
            num_steps = 4 if self.model.is_distilled() else 50

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Move components to device
        self.model.transformer.to(self.device)
        if self.model.vae is not None:
            self.model.vae.to(self.device)
        if self.model.text_encoder is not None:
            self.model.text_encoder.to(self.device)

        # Encode prompt using Qwen3 stacked layers
        prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            device=self.device,
            max_sequence_length=512,
        )

        # Prepare latents (32 channels before patchification)
        latents = self.model.prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        # Patchify: 32 → 128 channels
        latents = self.model.patchify_latents(latents)

        # Normalize using VAE batch norm statistics
        latents = self.model.normalize_latents(latents)

        # Pack latents for transformer: [B, 128, H, W] → [B, H*W, 128]
        packed_latents, img_ids = self.model.pack_latents(latents)

        # Pack text embeddings
        packed_text, txt_ids = self.model.pack_text(prompt_embeds)
        pooled_projections = self.model.pooled_text_projection(packed_text)

        # Get dimensions for unpacking later
        _, _, packed_h, packed_w = latents.shape

        # Use scheduler for proper timesteps
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_steps, device=self.device)

        # Denoising loop
        for i, t in enumerate(scheduler.timesteps):
            # Scheduler provides discrete timesteps, but transformer expects
            # continuous sigma values (0-1 range)
            sigma = scheduler.sigmas[i]

            # Forward through transformer
            # FLUX.2 Klein transformer expects:
            # - hidden_states: packed latents [B, N, C]
            # - encoder_hidden_states: text embeddings [B, L, D]
            # - timestep: sigma value (0-1 range for flow matching)
            # - img_ids: image position IDs
            # - txt_ids: text position IDs
            # - guidance: None for Klein (no CFG)
            timestep = sigma.expand(packed_latents.shape[0]).to(self.dtype)
            guidance = torch.ones(packed_latents.shape[0], device=self.device, dtype=self.dtype)

            with torch.autocast(device_type='cuda', dtype=self.dtype):
                velocity = self.model.transformer(
                    hidden_states=packed_latents,
                    timestep=timestep,
                    encoder_hidden_states=packed_text,
                    pooled_projections=pooled_projections,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]

            # Use scheduler step
            packed_latents = scheduler.step(
                velocity, t, packed_latents, return_dict=False
            )[0]

        # Unpack latents: [B, H*W, 128] → [B, 128, H, W]
        latents = self.model.unpack_latents(packed_latents, packed_h, packed_w)

        # Denormalize
        latents = self.model.denormalize_latents(latents)

        # Unpatchify: 128 → 32 channels
        latents = self.model.unpatchify_latents(latents)

        # Decode to image
        # Cast VAE to float32 for quality decode, then cast back
        vae_dtype = next(self.model.vae.parameters()).dtype
        self.model.vae.to(torch.float32)
        image = self.model.vae.decode(latents.float(), return_dict=False)[0]
        self.model.vae.to(vae_dtype)

        # Postprocess: [-1, 1] → [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def sample_with_lora(
        self,
        prompt: str,
        lora_path: str,
        lora_scale: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """
        Generate image with trained LoRA applied.

        Args:
            prompt: Text prompt
            lora_path: Path to LoRA safetensors file
            lora_scale: LoRA weight scale (0.0-1.0)
            **kwargs: Additional sampling arguments

        Returns:
            Image tensor [1, 3, H, W] in [0, 1] range
        """
        from ..adapters import create_adapter, detect_adapter_type, detect_rank
        from safetensors.torch import load_file

        # Load state dict and detect adapter type
        state_dict = load_file(lora_path)
        adapter_type = detect_adapter_type(state_dict)
        rank = detect_rank(state_dict)

        print(f"  Loading {adapter_type.value} adapter (rank={rank}) from {lora_path}")

        # Create and inject adapter
        adapter = create_adapter(
            adapter_type=adapter_type,
            rank=rank,
            model_type="flux_2_klein",
            device=self.device,
            dtype=self.dtype,
        )

        # Inject adapter into transformer
        self.model.transformer = adapter.inject(self.model.transformer)

        # Load weights
        adapter.load(lora_path)

        # Set adapter scale
        if hasattr(adapter, 'set_adapter_scale'):
            adapter.set_adapter_scale(lora_scale)

        try:
            # Generate image
            image = self.sample(prompt, **kwargs)
        finally:
            # Cleanup: unload adapter
            try:
                adapter.unmerge()
            except Exception:
                pass

        return image
