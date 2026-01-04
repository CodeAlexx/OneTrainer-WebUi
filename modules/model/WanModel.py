"""
WanModel - Wan 2.1/2.2 Text-to-Video and Image-to-Video Model

Based on diffusion-pipe's implementation:
- T5 text encoder (umt5-xxl)
- Custom 3D VAE
- Custom transformer (WanModel)
- Optional CLIP for i2v variants

Block swapping support ported from diffusion-pipe for 24GB VRAM training.
"""

from contextlib import nullcontext
from random import Random
from typing import TYPE_CHECKING

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from modules.util.BlockSwapOffloader import ModelOffloader


class WanModelEmbedding:
    """Embedding for Wan model - T5 text encoder embeddings."""
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )


class WanModel(BaseModel):
    """
    Wan 2.1/2.2 Video Generation Model
    
    Supports:
    - T2V (text-to-video)
    - I2V (image-to-video) with CLIP conditioning
    - Wan 2.2 dual transformer (high noise + low noise stages)
    """
    
    # Base model data
    tokenizer: object | None  # T5 tokenizer
    noise_scheduler: object | None  # Flow matching scheduler
    text_encoder: nn.Module | None  # T5 encoder
    vae: nn.Module | None  # Custom 3D VAE
    transformer: nn.Module | None  # WanModel transformer (or DualWanTransformer3DModel for 2.2)
    clip: nn.Module | None  # CLIP for i2v (optional)
    
    # Dual transformer support (Wan 2.2)
    is_dual_transformer: bool  # True if using DualWanTransformer3DModel
    low_vram: bool  # True to offload inactive transformer to CPU
    
    # Original copies
    orig_tokenizer: object | None
    
    # Autocast context
    transformer_autocast_context: torch.autocast | nullcontext
    
    transformer_train_dtype: DataType
    
    # Layer offload conductors
    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None
    
    # Persistent embedding training data
    embedding: WanModelEmbedding | None
    additional_embeddings: list[WanModelEmbedding] | None
    
    # Persistent LoRA training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None
    
    # Model configuration
    model_type_variant: str  # 't2v', 'i2v', 'i2v_v2', 'flf2v', 'ti2v'
    text_len: int
    framerate: int

    # Block swapping for low VRAM training (ported from diffusion-pipe)
    block_swap_offloader: "ModelOffloader | None"
    blocks_to_swap: int
    _original_block_forwards: list  # Store original forward methods

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.clip = None

        # Dual transformer (Wan 2.2)
        self.is_dual_transformer = False
        self.low_vram = False

        self.orig_tokenizer = None

        self.transformer_autocast_context = nullcontext()

        self.transformer_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

        # Set variant based on model type
        self.model_type_variant = 'i2v' if model_type.is_wan_i2v() else 't2v'
        self.text_len = 512
        self.framerate = 16  # Default, may change for 2.2 models

        # Block swapping (disabled by default)
        self.block_swap_offloader = None
        self.blocks_to_swap = 0
        self._original_block_forwards = []

    
    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]
    
    def all_embeddings(self) -> list[WanModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])
    
    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [emb.text_encoder_embedding for emb in self.additional_embeddings] \
               + ([self.embedding.text_encoder_embedding] if self.embedding is not None else [])
    
    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)
        if self.clip is not None:
            self.clip.to(device=device)
    
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
        else:
            self.transformer.to(device=device)
        
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)
    
    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
    
    def eval(self):
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.clip is not None:
            self.clip.eval()
        if self.transformer is not None:
            self.transformer.eval()
    
    def create_pipeline(self):
        """
        Wan doesn't use diffusers pipeline, returns None.
        Inference is handled differently.
        """
        return None
    
    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode text using T5.
        
        Returns:
            text_embeddings: Text embeddings from T5
            seq_lens: Sequence lengths for attention masking
        """
        if text_encoder_output is not None:
            # Use cached embeddings
            # Assume seq_lens is also cached
            return text_encoder_output, None
        
        if self.text_encoder is None or self.tokenizer is None:
            # Return zeros if no encoder
            return (
                torch.zeros(
                    size=(batch_size, self.text_len, 4096),  # T5-XXL hidden size
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                ),
                torch.ones(batch_size, dtype=torch.long, device=train_device)
            )
        
        # Tokenize
        if isinstance(text, str):
            text = [text]
        
        # Handle different tokenizer interfaces
        if hasattr(self.tokenizer, '__call__'):
            # Check if it's an HF tokenizer by looking for encode method
            if hasattr(self.tokenizer, 'encode') and hasattr(self.tokenizer, 'pad_token_id'):
                # HuggingFace tokenizer
                tokenized = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=self.text_len,
                    add_special_tokens=True,
                )
                tokens = tokenized['input_ids']
                mask = tokenized['attention_mask']
            else:
                # Custom tokenizer (diffusion-pipe style)
                tokens, mask = self.tokenizer(text, return_mask=True, add_special_tokens=True)
        else:
            raise ValueError(f"Unknown tokenizer type: {type(self.tokenizer)}")
        
        tokens = tokens.to(self.text_encoder.device)
        mask = mask.to(self.text_encoder.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        
        # Encode with T5 - handle different encoder interfaces
        with torch.autocast(device_type=self.text_encoder.device.type, dtype=self.train_dtype.torch_dtype()):
            # Check if this is an HF model (has forward that takes input_ids)
            if hasattr(self.text_encoder, 'config'):
                # HuggingFace model
                outputs = self.text_encoder(input_ids=tokens, attention_mask=mask)
                text_embeddings = outputs.last_hidden_state
            else:
                # Custom model (diffusion-pipe style)
                text_embeddings = self.text_encoder(tokens, mask)
        
        # Apply dropout if specified
        if text_encoder_dropout_probability is not None and rand is not None:
            dropout_mask = torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device
            ).float()
            text_embeddings = text_embeddings * dropout_mask[:, None, None]
        
        return text_embeddings, seq_lens
    
    def encode_image_for_i2v(
            self,
            images: Tensor,
    ) -> Tensor:
        """
        Encode first frame for i2v using CLIP.
        
        Args:
            images: First frame(s) of video, shape (B, C, H, W)
        
        Returns:
            clip_embeddings: CLIP features for conditioning
        """
        if self.clip is None:
            return None
        
        with torch.no_grad():
            clip_context = self.clip.visual(images.to(self.clip.device, self.clip.dtype))
        
        return clip_context
    
    def vae_encode(self, video: Tensor) -> Tensor:
        """
        Encode video to latents using 3D VAE.
        
        Args:
            video: Input video, shape (B, C, F, H, W)
        
        Returns:
            latents: Video latents
        """
        if self.vae is None:
            raise ValueError("VAE not loaded")
        
        latents_list = self.vae.encode(video)
        return torch.stack(latents_list)
    
    def vae_decode(self, latents: Tensor) -> Tensor:
        """
        Decode latents to video using 3D VAE.

        Args:
            latents: Video latents

        Returns:
            video: Decoded video, shape (B, C, F, H, W)
        """
        if self.vae is None:
            raise ValueError("VAE not loaded")

        video_list = self.vae.decode(latents)
        return torch.stack(video_list)

    def enable_block_swap(
            self,
            blocks_to_swap: int,
            device: torch.device = None,
            reentrant_activation_checkpointing: bool = False,
    ):
        """
        Enable block swapping for low VRAM training.

        Ported from diffusion-pipe for compatibility with 24GB training.

        Args:
            blocks_to_swap: Number of transformer blocks to swap between GPU and CPU.
                           For Wan 14B (40 blocks), use 32 for ~22GB VRAM.
            device: Target device (default: cuda)
            reentrant_activation_checkpointing: Use reentrant checkpointing mode.

        Example:
            # Enable block swapping for 24GB GPU training
            model.enable_block_swap(blocks_to_swap=32)
            model.prepare_block_swap_training()
        """
        from modules.util.BlockSwapOffloader import ModelOffloader

        if device is None:
            device = torch.device('cuda')

        # Get transformer blocks
        transformer = self.transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        elif hasattr(transformer, 'transformer_blocks'):
            blocks = transformer.transformer_blocks
        else:
            raise ValueError("Could not find transformer blocks. Expected 'blocks' or 'transformer_blocks' attribute.")

        num_blocks = len(blocks)
        assert blocks_to_swap <= num_blocks - 2, \
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'

        self.blocks_to_swap = blocks_to_swap

        # Create offloader
        self.block_swap_offloader = ModelOffloader(
            'WanTransformerBlock',
            list(blocks),
            num_blocks,
            blocks_to_swap,
            supports_backward=True,
            device=device,
            reentrant_activation_checkpointing=reentrant_activation_checkpointing,
            debug=False,
        )

        # Detach blocks from model, move non-block params to GPU
        if hasattr(transformer, 'blocks'):
            transformer.blocks = None
            transformer.to(device)
            transformer.blocks = blocks
        else:
            transformer.transformer_blocks = None
            transformer.to(device)
            transformer.transformer_blocks = blocks

        # Wrap block forward methods to use offloader
        self._wrap_blocks_for_offloading(blocks)

        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def _wrap_blocks_for_offloading(self, blocks: nn.ModuleList):
        """Wrap block forward methods to call offloader wait/submit."""
        self._original_block_forwards = []

        for block_idx, block in enumerate(blocks):
            # Store original forward
            original_forward = block.forward
            self._original_block_forwards.append(original_forward)

            # Create wrapped forward that uses offloader
            def make_wrapped_forward(orig_forward, idx):
                def wrapped_forward(*args, **kwargs):
                    # Wait for this block to be on GPU
                    if self.block_swap_offloader is not None:
                        self.block_swap_offloader.wait_for_block(idx)

                    # Call original forward
                    result = orig_forward(*args, **kwargs)

                    # Schedule next block transfer
                    if self.block_swap_offloader is not None:
                        self.block_swap_offloader.submit_move_blocks_forward(idx)

                    return result
                return wrapped_forward

            # Replace forward method
            block.forward = make_wrapped_forward(original_forward, block_idx)

    def prepare_block_swap_training(self):
        """Prepare for training with block swapping enabled."""
        if self.block_swap_offloader is None:
            return

        self.block_swap_offloader.enable_block_swap()
        self.block_swap_offloader.set_forward_only(False)
        self.block_swap_offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap: bool = False):
        """Prepare for inference with block swapping."""
        if self.block_swap_offloader is None:
            return

        if disable_block_swap:
            self.block_swap_offloader.disable_block_swap()
        self.block_swap_offloader.set_forward_only(True)
        self.block_swap_offloader.prepare_block_devices_before_forward()

    def disable_block_swap(self):
        """Disable block swapping and restore original block forwards."""
        if self.block_swap_offloader is None:
            return

        # Get transformer blocks
        transformer = self.transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        elif hasattr(transformer, 'transformer_blocks'):
            blocks = transformer.transformer_blocks
        else:
            return

        # Restore original forward methods
        for block, original_forward in zip(blocks, self._original_block_forwards):
            block.forward = original_forward

        self.block_swap_offloader = None
        self.blocks_to_swap = 0
        self._original_block_forwards = []
