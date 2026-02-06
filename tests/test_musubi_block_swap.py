"""
Test Musubi Block Swap Manager functionality.
"""

import sys
sys.path.insert(0, '/home/alex/OneTrainer')

import torch
import torch.nn as nn
from modules.util.musubi_block_swap import MusubiBlockSwapManager, _module_on_device


def test_module_on_device():
    """Test _module_on_device helper function."""
    model = nn.Linear(10, 10)
    model.to('cpu')
    assert _module_on_device(model, 'cpu'), "Model should be on CPU"

    if torch.cuda.is_available():
        model.to('cuda')
        assert _module_on_device(model, 'cuda'), "Model should be on CUDA"
        assert not _module_on_device(model, 'cpu'), "Model should not be on CPU"

    print("✓ test_module_on_device passed")


def test_build_manager():
    """Test MusubiBlockSwapManager.build() factory method."""
    # Test with 0 blocks to swap - should return None
    manager = MusubiBlockSwapManager.build(depth=10, blocks_to_swap=0)
    assert manager is None, "Should return None for 0 blocks"

    # Test with valid blocks to swap
    manager = MusubiBlockSwapManager.build(depth=10, blocks_to_swap=5)
    assert manager is not None, "Should return manager for 5 blocks"
    assert len(manager.block_indices) == 5, "Should have 5 block indices"
    assert manager.block_indices == {5, 6, 7, 8, 9}, "Should swap last 5 blocks"

    # Test clamping when requesting more than available
    manager = MusubiBlockSwapManager.build(depth=10, blocks_to_swap=20)
    assert manager is not None
    assert len(manager.block_indices) == 9, "Should clamp to max swappable (depth-1)"

    print("✓ test_build_manager passed")


def test_is_managed_block():
    """Test is_managed_block method."""
    manager = MusubiBlockSwapManager.build(depth=10, blocks_to_swap=3)

    # Last 3 blocks should be managed (7, 8, 9)
    assert not manager.is_managed_block(0), "Block 0 should not be managed"
    assert not manager.is_managed_block(6), "Block 6 should not be managed"
    assert manager.is_managed_block(7), "Block 7 should be managed"
    assert manager.is_managed_block(8), "Block 8 should be managed"
    assert manager.is_managed_block(9), "Block 9 should be managed"

    print("✓ test_is_managed_block passed")


def test_stream_in_out():
    """Test stream_in and stream_out methods."""
    if not torch.cuda.is_available():
        print("⚠ Skipping CUDA test (no GPU available)")
        return

    manager = MusubiBlockSwapManager.build(depth=5, blocks_to_swap=2)

    # Create a simple module
    block = nn.Linear(10, 10)
    block.to('cpu')

    # Stream in to GPU
    manager.stream_in(block, torch.device('cuda'))
    assert _module_on_device(block, 'cuda'), "Block should be on CUDA after stream_in"

    # Stream out to CPU
    manager.stream_out(block)
    assert _module_on_device(block, 'cpu'), "Block should be on CPU after stream_out"

    print("✓ test_stream_in_out passed")


def test_activate_with_forward_hooks():
    """Test activate_with_forward_hooks for ZImage/Qwen style models."""
    if not torch.cuda.is_available():
        print("⚠ Skipping CUDA forward hooks test (no GPU available)")
        return

    manager = MusubiBlockSwapManager.build(depth=5, blocks_to_swap=2)

    # Create blocks
    blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
    blocks.to('cuda')  # Start on GPU

    # Activate with forward hooks
    success = manager.activate_with_forward_hooks(
        blocks,
        torch.device('cuda'),
        grad_enabled=True
    )
    assert success, "Activation should succeed"

    # Check that managed blocks are offloaded
    for i, block in enumerate(blocks):
        if manager.is_managed_block(i):
            assert _module_on_device(block, 'cpu'), f"Block {i} should be offloaded to CPU"
        else:
            assert _module_on_device(block, 'cuda'), f"Block {i} should remain on CUDA"

    # Test forward pass - hooks should move blocks in/out
    x = torch.randn(2, 10, device='cuda')
    for i, block in enumerate(blocks):
        x = block(x)  # Forward hooks should stream block in before, out after

    # After forward, managed blocks should be back on CPU
    for i, block in enumerate(blocks):
        if manager.is_managed_block(i):
            assert _module_on_device(block, 'cpu'), f"Block {i} should be on CPU after forward"

    # Cleanup
    manager.cleanup()

    print("✓ test_activate_with_forward_hooks passed")


def test_backward_hooks():
    """Test backward hooks for training."""
    if not torch.cuda.is_available():
        print("⚠ Skipping CUDA backward hooks test (no GPU available)")
        return

    manager = MusubiBlockSwapManager.build(depth=3, blocks_to_swap=1)

    # Create blocks with requires_grad - start on GPU
    blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
    blocks.to('cuda')

    # Activate with forward hooks (which also registers backward hooks)
    # This will offload managed blocks (block 2) to CPU
    success = manager.activate_with_forward_hooks(
        blocks,
        torch.device('cuda'),
        grad_enabled=True
    )
    assert success

    # Verify block 2 is offloaded to CPU, blocks 0,1 stay on GPU
    assert _module_on_device(blocks[0], 'cuda'), "Block 0 should be on GPU"
    assert _module_on_device(blocks[1], 'cuda'), "Block 1 should be on GPU"
    assert _module_on_device(blocks[2], 'cpu'), "Block 2 should be offloaded to CPU"

    # Forward pass - hooks should move block 2 to GPU during its forward
    x = torch.randn(2, 10, device='cuda', requires_grad=True)
    for block in blocks:
        x = block(x)

    # After forward, block 2 should be back on CPU
    assert _module_on_device(blocks[2], 'cpu'), "Block 2 should be on CPU after forward"

    # Backward pass - hooks should handle block movement for gradients
    loss = x.sum()
    loss.backward()

    # After backward, block 2 should be back on CPU
    assert _module_on_device(blocks[2], 'cpu'), "Block 2 should be on CPU after backward"

    # Cleanup
    manager.cleanup()

    print("✓ test_backward_hooks passed")


def run_all_tests():
    """Run all tests."""
    print("\n=== Testing MusubiBlockSwapManager ===\n")

    test_module_on_device()
    test_build_manager()
    test_is_managed_block()
    test_stream_in_out()
    test_activate_with_forward_hooks()
    test_backward_hooks()

    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
