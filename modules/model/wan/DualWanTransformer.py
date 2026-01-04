"""
DualWanTransformer3DModel - Wrapper for Wan 2.2's dual transformer architecture

Wan 2.2 14B uses two transformers that split by timestep:
- transformer_1: High noise (timesteps 1000-875, ratio > 0.875)
- transformer_2: Low noise (timesteps 875-0, ratio <= 0.875)

Based on ai-toolkit's implementation.
"""

from typing import Any, Dict, Optional, Union
import torch
from torch import nn


class DualWanTransformer3DModel(nn.Module):
    """
    Wrapper that routes to the correct transformer based on timestep.
    
    For Wan 2.2 14B:
    - High noise stage (transformer_1): timesteps > boundary (875)
    - Low noise stage (transformer_2): timesteps <= boundary (875)
    """
    
    def __init__(
        self,
        transformer_1: nn.Module,
        transformer_2: nn.Module,
        boundary_ratio: float = 0.875,
        low_vram: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.transformer_1 = transformer_1
        self.transformer_2 = transformer_2
        self.boundary_ratio = boundary_ratio
        self.boundary = boundary_ratio * 1000  # Timesteps are 0-1000
        self.low_vram = low_vram
        self._device = device
        self._dtype = dtype
        self._active_transformer_name = "transformer_1"
        
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def config(self):
        """Return config from first transformer."""
        return self.transformer_1.config
    
    @property
    def active_transformer(self) -> nn.Module:
        """Return currently active transformer."""
        return getattr(self, self._active_transformer_name)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for both transformers."""
        if hasattr(self.transformer_1, 'enable_gradient_checkpointing'):
            self.transformer_1.enable_gradient_checkpointing()
        if hasattr(self.transformer_2, 'enable_gradient_checkpointing'):
            self.transformer_2.enable_gradient_checkpointing()
    
    def _get_transformer_for_timestep(self, timestep: torch.Tensor) -> str:
        """Determine which transformer to use based on mean timestep."""
        mean_t = timestep.float().mean().item()
        if mean_t > self.boundary:
            return "transformer_1"
        else:
            return "transformer_2"
    
    def _swap_transformer_if_needed(self, target_name: str):
        """Swap active transformer, moving to/from CPU if low_vram mode."""
        if target_name == self._active_transformer_name:
            return
            
        if self.low_vram:
            # Move current transformer to CPU
            current = getattr(self, self._active_transformer_name)
            current.to("cpu")
            torch.cuda.empty_cache()
            
            # Move target transformer to device
            target = getattr(self, target_name)
            target.to(self._device)
            
        self._active_transformer_name = target_name
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Forward pass that routes to the correct transformer based on timestep.
        """
        # Determine which transformer to use
        with torch.no_grad():
            target_name = self._get_transformer_for_timestep(timestep)
        
        # Swap if needed (handles low_vram offloading)
        self._swap_transformer_if_needed(target_name)
        
        transformer = self.active_transformer
        
        # Ensure transformer is on correct device
        if transformer.device != hidden_states.device:
            if self.low_vram:
                # Move other to CPU first
                other_name = "transformer_1" if target_name == "transformer_2" else "transformer_2"
                getattr(self, other_name).to("cpu")
                torch.cuda.empty_cache()
            transformer.to(hidden_states.device)
        
        # Forward through active transformer
        return transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            **kwargs
        )
    
    def to(self, *args, **kwargs):
        """
        Override to() - with torchao quantization, models CAN be moved.
        With low_vram mode, only move the active transformer to GPU.
        """
        # Extract target device from args/kwargs
        target_device = None
        if args:
            if isinstance(args[0], torch.device):
                target_device = args[0]
            elif isinstance(args[0], str):
                target_device = torch.device(args[0])
        if 'device' in kwargs:
            target_device = kwargs['device']
            if isinstance(target_device, str):
                target_device = torch.device(target_device)
        
        # Update internal device tracking
        if target_device:
            self._device = target_device
        
        # With torchao quantization, models CAN be moved
        if self.low_vram and target_device and 'cuda' in str(target_device):
            # Only move active transformer to GPU
            self.active_transformer.to(target_device)
        elif target_device:
            # Move all to target
            self.transformer_1.to(target_device)
            self.transformer_2.to(target_device)
        
        return self
    
    def train(self, mode: bool = True):
        """Set training mode for both transformers."""
        self.transformer_1.train(mode)
        self.transformer_2.train(mode)
        return self
    
    def eval(self):
        """Set eval mode for both transformers."""
        self.transformer_1.eval()
        self.transformer_2.eval()
        return self
    
    def requires_grad_(self, requires_grad: bool = True):
        """Set requires_grad for both transformers."""
        self.transformer_1.requires_grad_(requires_grad)
        self.transformer_2.requires_grad_(requires_grad)
        return self
    
    def parameters(self, recurse: bool = True):
        """Yield parameters from both transformers."""
        yield from self.transformer_1.parameters(recurse)
        yield from self.transformer_2.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Yield named parameters from both transformers."""
        yield from self.transformer_1.named_parameters(f"{prefix}transformer_1.", recurse)
        yield from self.transformer_2.named_parameters(f"{prefix}transformer_2.", recurse)
