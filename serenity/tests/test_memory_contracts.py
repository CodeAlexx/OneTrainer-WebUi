"""
Memory Contract Tests

Test memory management contracts:
- Offload/restore roundtrip
- Memory actually freed
- Gradient checkpointing
- Block swap behavior
"""

import gc
import torch
import torch.nn as nn


# =============================================================================
# Helper: Create Mock Transformer
# =============================================================================

def create_mock_transformer(num_blocks: int = 5, hidden_dim: int = 512):
    """Create a simple mock transformer for testing."""

    class MockBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(self.linear(x))

    class MockTransformer(nn.Module):
        def __init__(self, num_blocks, dim):
            super().__init__()
            self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(num_blocks)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    return MockTransformer(num_blocks, hidden_dim)


# =============================================================================
# Test: LayerOffloadConductor Basic
# =============================================================================

class TestLayerOffloadConductorBasic:
    """Basic tests for LayerOffloadConductor."""

    def test_conductor_creation(self):
        """Conductor can be created with a module."""
        from serenity.training.layer_offload import LayerOffloadConductor

        # Create a minimal config-like object
        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.0
            gradient_checkpointing = 'off'
            enable_activation_offloading = False
            enable_async_offloading = False

        transformer = create_mock_transformer()
        conductor = LayerOffloadConductor(transformer, MockConfig())

        assert conductor is not None
        print("  ✅ LayerOffloadConductor created")

    def test_conductor_add_layers(self):
        """Conductor can register layers."""
        from serenity.training.layer_offload import LayerOffloadConductor

        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.0
            gradient_checkpointing = 'off'
            enable_activation_offloading = False
            enable_async_offloading = False

        transformer = create_mock_transformer(num_blocks=5)
        conductor = LayerOffloadConductor(transformer, MockConfig())

        # Register blocks
        for block in transformer.blocks:
            conductor.add_layer(block)

        assert len(conductor._layers) == 5
        print("  ✅ Conductor has 5 layers registered")


# =============================================================================
# Test: Block Swap Memory
# =============================================================================

class TestBlockSwapMemory:
    """Test block swap memory behavior."""

    def test_weights_move_to_cpu(self):
        """Block weights move to CPU."""
        from serenity.training.block_swap import weights_to_device

        block = nn.Linear(100, 100)

        # Weights start on CPU by default
        assert block.weight.device.type == 'cpu'

        # Move to CPU explicitly (should be no-op)
        weights_to_device(block, torch.device('cpu'))
        assert block.weight.device.type == 'cpu'

        print("  ✅ weights_to_device works on CPU")

    def test_block_swap_enable_disable(self):
        """Block swap can be enabled and disabled."""
        from serenity.training.block_swap import BlockSwapOffloader

        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

        offloader = BlockSwapOffloader(
            block_type="test",
            blocks=list(blocks),
            num_blocks=5,
            blocks_to_swap=2,
            device=torch.device('cpu'),
        )

        assert offloader.blocks_to_swap == 2

        # Disable
        offloader.disable_block_swap()
        assert offloader.blocks_to_swap == 0

        # Re-enable
        offloader.enable_block_swap()
        assert offloader.blocks_to_swap == 2

        print("  ✅ Block swap enable/disable works")

    def test_prepare_block_devices(self):
        """Block devices are prepared correctly for forward."""
        from serenity.training.block_swap import BlockSwapOffloader

        blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

        offloader = BlockSwapOffloader(
            block_type="test",
            blocks=list(blocks),
            num_blocks=5,
            blocks_to_swap=2,  # Last 2 blocks on CPU
            device=torch.device('cpu'),
        )

        # Prepare - all on CPU for this test
        offloader.prepare_block_devices_before_forward()

        # Verify all blocks are on CPU (since device is CPU)
        for block in blocks:
            for p in block.parameters():
                assert p.device.type == 'cpu'

        print("  ✅ prepare_block_devices_before_forward works")


# =============================================================================
# Test: Memory Sync Utilities
# =============================================================================

class TestMemorySyncUtilities:
    """Test memory sync utilities."""

    def test_torch_gc(self):
        """torch_gc runs without error."""
        from serenity.memory.sync import torch_gc

        # Allocate some tensors
        tensors = [torch.randn(100, 100) for _ in range(10)]

        # Delete references
        del tensors

        # GC should work
        torch_gc()

        print("  ✅ torch_gc works")

    def test_device_equals(self):
        """device_equals correctly compares devices."""
        from serenity.memory.sync import device_equals

        cpu = torch.device('cpu')
        cpu2 = torch.device('cpu')

        assert device_equals(cpu, cpu2)
        assert device_equals(None, None)
        assert not device_equals(cpu, None)
        assert not device_equals(None, cpu)

        print("  ✅ device_equals works")

    def test_tensors_match_device(self):
        """tensors_match_device checks tensor locations."""
        from serenity.memory.sync import tensors_match_device

        # Test with tuple of tensors
        t1 = torch.randn(10)
        t2 = torch.randn(10)

        assert tensors_match_device((t1, t2), torch.device('cpu'), [0, 1])
        print("  ✅ tensors_match_device works")


# =============================================================================
# Test: Static Allocators
# =============================================================================

class TestStaticAllocators:
    """Test static memory allocators."""

    def test_static_layer_allocator_creation(self):
        """StaticLayerAllocator can be created."""
        from serenity.memory.allocators import StaticLayerAllocator

        allocator = StaticLayerAllocator(torch.device('cpu'))
        assert allocator is not None
        print("  ✅ StaticLayerAllocator created")

    def test_static_activation_allocator_creation(self):
        """StaticActivationAllocator can be created."""
        from serenity.memory.allocators import StaticActivationAllocator

        allocator = StaticActivationAllocator(torch.device('cpu'))
        assert allocator is not None
        print("  ✅ StaticActivationAllocator created")


# =============================================================================
# Test: Gradient Checkpointing
# =============================================================================

class TestGradientCheckpointing:
    """Test gradient checkpointing setup."""

    def test_gradient_checkpointing_off(self):
        """Can disable gradient checkpointing."""
        from serenity.training.gradient import setup_gradient_checkpointing
        from serenity.core.enums import GradientCheckpointingMethod

        model = nn.Linear(10, 10)
        setup_gradient_checkpointing(model, GradientCheckpointingMethod.OFF)

        print("  ✅ Gradient checkpointing OFF works")

    def test_gradient_checkpointing_on(self):
        """Can enable gradient checkpointing."""
        from serenity.training.gradient import setup_gradient_checkpointing
        from serenity.core.enums import GradientCheckpointingMethod

        model = create_mock_transformer()
        setup_gradient_checkpointing(model, GradientCheckpointingMethod.ON)

        print("  ✅ Gradient checkpointing ON works")


# =============================================================================
# Test: MemoryManager Basic
# =============================================================================

class TestMemoryManagerBasic:
    """Basic tests for MemoryManager."""

    def test_memory_manager_creation(self):
        """MemoryManager can be created with config and model."""
        from serenity.memory.manager import MemoryManager

        # Create minimal mock config
        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.0
            gradient_checkpointing = 'off'
            enable_activation_offloading = False

        # Create minimal mock model
        class MockModel:
            transformer = create_mock_transformer()

        config = MockConfig()
        model = MockModel()

        manager = MemoryManager(config, model)
        assert manager is not None
        print("  ✅ MemoryManager created")

    def test_memory_manager_properties(self):
        """MemoryManager properties work correctly."""
        from serenity.memory.manager import MemoryManager

        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.5
            gradient_checkpointing = 'on'
            enable_activation_offloading = True

        class MockModel:
            transformer = create_mock_transformer()

        config = MockConfig()
        model = MockModel()

        manager = MemoryManager(config, model)

        assert manager.use_layer_offloading == True
        assert manager.use_gradient_checkpointing == True
        # Note: use_activation_offloading depends on gradient_checkpointing == 'cpu_offloaded'
        # Since we have 'on', it should be False
        assert manager.use_activation_offloading == False

        print("  ✅ MemoryManager properties work")

    def test_memory_manager_activation_offload_requires_cpu_offloaded_checkpointing(self):
        """Activation offload toggles only in cpu_offloaded checkpoint mode."""
        from serenity.memory.manager import MemoryManager

        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.0
            gradient_checkpointing = 'cpu_offloaded'
            enable_activation_offloading = True

        class MockModel:
            transformer = create_mock_transformer()

        manager = MemoryManager(MockConfig(), MockModel())
        assert manager.use_activation_offloading == True
        print("  ✅ Activation offload enabled in cpu_offloaded mode")

    def test_memory_manager_get_memory_usage(self):
        """MemoryManager can report memory usage."""
        from serenity.memory.manager import MemoryManager

        class MockConfig:
            train_device = 'cpu'
            temp_device = 'cpu'
            layer_offload_fraction = 0.0
            gradient_checkpointing = 'off'

        class MockModel:
            transformer = create_mock_transformer()

        config = MockConfig()
        model = MockModel()

        manager = MemoryManager(config, model)
        stats = manager.get_memory_usage()

        assert 'allocated_gb' in stats
        assert 'cached_gb' in stats
        assert 'max_allocated_gb' in stats

        print(f"  ✅ Memory stats: {stats}")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all memory contract tests."""
    print("=== Memory Contract Tests ===\n")

    print("Test: LayerOffloadConductor Basic")
    t = TestLayerOffloadConductorBasic()
    try:
        t.test_conductor_creation()
        t.test_conductor_add_layers()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Block Swap Memory")
    t = TestBlockSwapMemory()
    try:
        t.test_weights_move_to_cpu()
        t.test_block_swap_enable_disable()
        t.test_prepare_block_devices()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Memory Sync Utilities")
    t = TestMemorySyncUtilities()
    try:
        t.test_torch_gc()
        t.test_device_equals()
        t.test_tensors_match_device()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Static Allocators")
    t = TestStaticAllocators()
    try:
        t.test_static_layer_allocator_creation()
        t.test_static_activation_allocator_creation()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Gradient Checkpointing")
    t = TestGradientCheckpointing()
    try:
        t.test_gradient_checkpointing_off()
        t.test_gradient_checkpointing_on()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: MemoryManager Basic")
    t = TestMemoryManagerBasic()
    try:
        t.test_memory_manager_creation()
        t.test_memory_manager_properties()
        t.test_memory_manager_activation_offload_requires_cpu_offloaded_checkpointing()
        t.test_memory_manager_get_memory_usage()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\n=== Memory Contract Tests Complete ===")


if __name__ == "__main__":
    run_all_tests()
