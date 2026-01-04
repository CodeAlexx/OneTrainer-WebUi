"""
Musubi Block Swap Manager for memory-efficient training.

Streams transformer blocks between CPU and GPU during forward/backward passes
to reduce VRAM usage. Based on SimpleTuner's musubi_block_swap implementation.
"""

import logging
from typing import Iterable, List, Optional, Set

import torch
import torch.nn as nn

__all__ = ["MusubiBlockSwapManager"]


def _module_on_device(module: nn.Module, device: torch.device) -> bool:
    """Check if all parameters and buffers of a module are on the specified device."""
    target = torch.device(device)
    for tensor in module.parameters():
        # Compare device types (cuda vs cuda:0 should match)
        if tensor.device.type != target.type:
            return False
        # If target has specific index, check that too
        if target.index is not None and tensor.device.index != target.index:
            return False
    for tensor in module.buffers():
        if tensor.device.type != target.type:
            return False
        if target.index is not None and tensor.device.index != target.index:
            return False
    return True


class MusubiBlockSwapManager:
    """
    Streams a subset of transformer blocks between devices to reduce VRAM usage.

    Works for both forward and backward passes by registering hooks that
    automatically move blocks to GPU before processing and back to CPU after.
    """

    def __init__(
        self,
        block_indices: List[int],
        offload_device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.block_indices = set(block_indices)
        self.offload_device = offload_device
        self._logger = logger or logging.getLogger(__name__)
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hook_device: Optional[torch.device] = None
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._forward_hook_device: Optional[torch.device] = None

    @classmethod
    def build(
        cls,
        depth: int,
        blocks_to_swap: int,
        swap_device: str = "cpu",
        logger: Optional[logging.Logger] = None,
    ) -> Optional["MusubiBlockSwapManager"]:
        """
        Factory method to create a MusubiBlockSwapManager.

        Args:
            depth: Total number of transformer blocks
            blocks_to_swap: Number of blocks to swap (from the end)
            swap_device: Device to offload blocks to (default: "cpu")
            logger: Optional logger instance

        Returns:
            MusubiBlockSwapManager instance or None if blocks_to_swap is 0
        """
        if blocks_to_swap is None or blocks_to_swap == 0:
            return None
        if blocks_to_swap < 0:
            raise ValueError(f"blocks_to_swap must be non-negative, got {blocks_to_swap}")

        max_swappable_blocks = max(depth - 1, 0)
        if max_swappable_blocks == 0:
            return None

        if blocks_to_swap > max_swappable_blocks:
            if logger:
                logger.warning(
                    "Requested blocks_to_swap=%s but maximum is %s; clamping.",
                    blocks_to_swap, max_swappable_blocks
                )
            blocks_to_swap = max_swappable_blocks

        # Swap the last N blocks (they're typically less critical)
        block_indices = list(range(depth - blocks_to_swap, depth))

        try:
            offload_device = torch.device(swap_device)
        except Exception as exc:
            if logger:
                logger.warning("Failed to initialize block offload device: %s", exc)
            return None

        return cls(block_indices, offload_device, logger)

    def activate(
        self,
        blocks: Iterable[nn.Module],
        compute_device: torch.device,
        grad_enabled: bool
    ) -> bool:
        """
        Activate block swapping for a list of transformer blocks.

        This mode requires manual stream_in/stream_out calls in the forward loop.
        Backward hooks are registered automatically for training.

        Args:
            blocks: Iterable of transformer block modules
            compute_device: Device to use for computation (GPU)
            grad_enabled: Whether gradients are enabled (training mode)

        Returns:
            True if activation was successful
        """
        if compute_device == self.offload_device:
            return False

        blocks_list = list(blocks)
        self._ensure_backward_hooks(blocks_list, compute_device, grad_enabled)
        self.mark_blocks_for_offload(blocks_list)
        return True

    def activate_with_forward_hooks(
        self,
        blocks: Iterable[nn.Module],
        compute_device: torch.device,
        grad_enabled: bool
    ) -> bool:
        """
        Activate block swapping using forward hooks (for diffusers models).

        Use this when you can't modify the forward loop directly.
        Forward hooks handle stream_in/stream_out automatically.
        Backward hooks are also registered for training.

        Args:
            blocks: Iterable of transformer block modules
            compute_device: Device to use for computation (GPU)
            grad_enabled: Whether gradients are enabled (training mode)

        Returns:
            True if activation was successful
        """
        if compute_device == self.offload_device:
            return False

        blocks_list = list(blocks)
        self._ensure_forward_hooks(blocks_list, compute_device)
        self._ensure_backward_hooks(blocks_list, compute_device, grad_enabled)
        self.mark_blocks_for_offload(blocks_list)
        return True

    def is_managed_block(self, index: int) -> bool:
        """Check if a block index is managed by this swap manager."""
        return index in self.block_indices

    def stream_in(self, block: nn.Module, device: torch.device):
        """Move a block to the compute device (GPU)."""
        self._move_module(block, device)
        if not _module_on_device(block, device):
            self._logger.error(
                "stream_in failed: block not fully on %s after move.", device
            )

    def stream_out(self, block: nn.Module):
        """Move a block to the offload device (CPU)."""
        self._move_module(block, self.offload_device)

    def mark_blocks_for_offload(self, blocks: List[nn.Module]):
        """Move all managed blocks to the offload device."""
        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            self._move_module(blocks[idx], self.offload_device)

    def _clear_backward_hooks(self):
        """Remove all registered backward hooks."""
        for handle in self._backward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._backward_hooks.clear()
        self._backward_hook_device = None

    def _clear_forward_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._forward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._forward_hooks.clear()
        self._forward_hook_device = None

    def _ensure_forward_hooks(
        self,
        blocks: List[nn.Module],
        compute_device: torch.device
    ) -> None:
        """Register forward hooks to handle block streaming during forward pass."""
        if self._forward_hook_device == compute_device and self._forward_hooks:
            return

        self._clear_forward_hooks()

        for idx, block in enumerate(blocks):
            if not self.is_managed_block(idx):
                continue

            # Create closures for hooks
            def _make_pre_hook(blk, dev):
                def _pre_hook(_module, _args):
                    self.stream_in(blk, dev)
                    return None
                return _pre_hook

            def _make_post_hook(blk):
                def _post_hook(_module, _args, _output):
                    self.stream_out(blk)
                    return None
                return _post_hook

            # Register hooks for forward pass
            self._forward_hooks.append(
                block.register_forward_pre_hook(_make_pre_hook(block, compute_device))
            )
            self._forward_hooks.append(
                block.register_forward_hook(_make_post_hook(block))
            )

        self._forward_hook_device = compute_device

    def _ensure_backward_hooks(
        self,
        blocks: List[nn.Module],
        compute_device: torch.device,
        grad_enabled: bool
    ) -> None:
        """Register backward hooks to handle block streaming during backprop."""
        if not grad_enabled:
            return

        if self._backward_hook_device == compute_device and self._backward_hooks:
            return

        self._clear_backward_hooks()

        for idx, block in enumerate(blocks):
            if not self.is_managed_block(idx):
                continue

            # Create closures for hooks
            def _make_pre_hook(blk, dev):
                def _pre_hook(_module, _grad_output):
                    self.stream_in(blk, dev)
                    return None
                return _pre_hook

            def _make_post_hook(blk):
                def _post_hook(_module, _grad_input, _grad_output):
                    self.stream_out(blk)
                    return None
                return _post_hook

            # Register hooks for backward pass
            self._backward_hooks.append(
                block.register_full_backward_pre_hook(_make_pre_hook(block, compute_device))
            )
            self._backward_hooks.append(
                block.register_full_backward_hook(_make_post_hook(block))
            )

        self._backward_hook_device = compute_device

    def _move_module(self, module: nn.Module, device: torch.device):
        """Move a module to a device if not already there."""
        if _module_on_device(module, device):
            return
        with torch.no_grad():
            module.to(device)

    def cleanup(self):
        """Clean up hooks and resources."""
        self._clear_forward_hooks()
        self._clear_backward_hooks()
