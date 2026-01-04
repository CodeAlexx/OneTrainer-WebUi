"""
BlockSwapOffloader - Block swapping for memory-efficient training

Ported from diffusion-pipe/musubi-tuner for compatibility.
Enables training large models (Wan 14B, Flux, HunyuanVideo) on 24GB VRAM
by swapping transformer blocks between GPU and CPU.

Usage:
    offloader = ModelOffloader(
        'TransformerBlock', blocks, len(blocks), blocks_to_swap=32,
        supports_backward=True, device=torch.device('cuda'),
        reentrant_activation_checkpointing=False
    )
    offloader.enable_block_swap()
    offloader.prepare_block_devices_before_forward()
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional

import torch
import torch.nn as nn


def clean_memory_on_device(device: torch.device):
    """Clean memory on the specified device."""
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    """Synchronize device operations."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Efficiently swap weights between CPU and CUDA using streams.

    Skips LoRA parameters (must stay on GPU for optimizer step).
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        # Skip LoRA modules - they must stay on GPU for training
        if 'lora' in module_to_cuda_name.lower():
            continue
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # CUDA to CPU
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # CPU to CUDA
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """Swap weights for non-CUDA devices (not fully tested)."""
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # Device to CPU
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device(device)

    # CPU to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device(device)


def weights_to_device(layer: nn.Module, device: torch.device, verbose: bool = False):
    """Move layer weights to device, skipping LoRA params when moving to CPU."""
    for name, module in layer.named_modules():
        # Skip LoRA modules when moving to CPU - they must stay on GPU
        if device.type == 'cpu' and 'lora' in name.lower():
            if verbose:
                print(f"Skipping moving {name} to CPU")
            continue
        if hasattr(module, "weight") and module.weight is not None:
            if verbose and 'lora' in name.lower():
                print(f"Moving {name} to {device}")
            module.weight.data = module.weight.data.to(device, non_blocking=True)


class Offloader:
    """Base offloading class with thread pool for async transfers."""

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        debug: bool = False
    ):
        self.block_type = block_type
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.blocks_to_swap_tmp = None
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        """Swap weights between CPU and GPU."""
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, block_idx_to_cpu: int, block_idx_to_cuda: int):
        """Submit async block transfer job."""
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to GPU")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda

        block_to_cpu = self.blocks[block_idx_to_cpu]
        block_to_cuda = self.blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx: int):
        """Wait for block transfer to complete."""
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    Block swapping offloader for transformer models.

    Supports forward and backward pass offloading for training.

    Args:
        block_type: Name for logging (e.g., 'TransformerBlock')
        blocks: List of transformer block modules
        num_blocks: Total number of blocks
        blocks_to_swap: Number of blocks to keep on CPU (swap during forward/backward)
        supports_backward: Enable backward pass offloading (for training)
        device: Target device (usually cuda)
        reentrant_activation_checkpointing: Use reentrant checkpointing (affects swap timing)
        debug: Enable debug logging

    Example:
        >>> offloader = ModelOffloader(
        ...     'WanBlock', transformer.blocks, 40, blocks_to_swap=32,
        ...     supports_backward=True, device=torch.device('cuda'),
        ...     reentrant_activation_checkpointing=False
        ... )
        >>> offloader.enable_block_swap()
        >>> offloader.prepare_block_devices_before_forward()

        # During forward pass (called by block wrapper):
        >>> offloader.wait_for_block(block_idx)
        >>> output = block(inputs)
        >>> offloader.submit_move_blocks_forward(block_idx)
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        reentrant_activation_checkpointing: bool,
        debug: bool = False,
    ):
        super().__init__(block_type, blocks, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.reentrant_activation_checkpointing = reentrant_activation_checkpointing

        if self.supports_backward:
            # Register backward hooks for block swapping during backward pass
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def disable_block_swap(self):
        """Temporarily disable block swapping."""
        self.blocks_to_swap_tmp = self.blocks_to_swap
        self.blocks_to_swap = None

    def enable_block_swap(self):
        """Enable block swapping (restore from disabled state)."""
        if self.blocks_to_swap_tmp is not None:
            self.blocks_to_swap = self.blocks_to_swap_tmp

    def set_forward_only(self, forward_only: bool):
        """Set forward-only mode (for inference)."""
        self.forward_only = forward_only

    def __del__(self):
        """Clean up backward hooks."""
        if hasattr(self, 'supports_backward') and self.supports_backward:
            if hasattr(self, 'remove_handles'):
                for handle in self.remove_handles:
                    handle.remove()

    def create_backward_hook(self, block_index: int) -> Optional[callable]:
        """Create backward hook for block swapping during backward pass."""
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self):
        """
        Prepare block devices before forward pass.

        Moves first (num_blocks - blocks_to_swap) blocks to GPU,
        and last blocks_to_swap blocks to CPU (weights only).
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            # No swapping - all blocks on GPU
            for block in self.blocks:
                block.to(self.device)
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        # First blocks stay on GPU
        for b in self.blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weights_to_device(b, self.device)

        # Last blocks: structure on GPU, weights on CPU
        for b in self.blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # Move structure to GPU
            weights_to_device(b, torch.device('cpu'))  # Move weights to CPU

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        """Wait for block to be transferred to GPU (called before forward)."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if self.reentrant_activation_checkpointing and torch.is_grad_enabled():
            # Second forward pass with reentrant checkpointing - skip
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, block_idx: int):
        """Submit block transfer after forward (pipelined with next block)."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.reentrant_activation_checkpointing and torch.is_grad_enabled():
            # Second forward pass with reentrant checkpointing - skip
            return

        # In backward mode, swap additional blocks during backward pass
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # Wrap for forward-only
        self._submit_move_blocks(block_idx_to_cpu, block_idx_to_cuda)
