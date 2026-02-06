"""Integration tests for native layer offload strategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from eritrainer.memory.checkpoint_layer import OffloadCheckpointLayer
from eritrainer.memory.strategy import LayerOffloadStrategy, MemoryConfig


class MockTransformerLayer(nn.Module):
    """Simple transformer-like block."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.linear(x)))


class MockTransformer(nn.Module):
    """Transformer container with a supported layer attribute name."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([MockTransformerLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_blocks:
            x = layer(x)
        return x


class MockModel:
    """Minimal model contract with transformer attr."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        self._transformer = MockTransformer(num_layers=num_layers, hidden_size=hidden_size)

    @property
    def transformer(self) -> nn.Module:
        return self._transformer

    def to(self, device):
        self._transformer.to(device)
        return self


class TestConductorActivation:
    def test_conductor_created_during_setup(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.5,
            gradient_checkpointing="on",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        assert strategy.conductor is not None
        strategy.cleanup()

    def test_before_layer_called_during_forward(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.5,
            gradient_checkpointing="on",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        assert strategy.conductor is not None

        call_count = {"value": 0}
        original_before = strategy.conductor.before_layer

        def counted_before(layer_idx, call_id, activations=None):
            call_count["value"] += 1
            return original_before(layer_idx, call_id, activations)

        strategy.conductor.before_layer = counted_before  # type: ignore[assignment]

        x = torch.randn(2, 64, requires_grad=True)
        with strategy.forward_context():
            y = model.transformer(x)
            loss = y.mean()
            loss.backward()

        assert call_count["value"] >= 4
        strategy.cleanup()

    def test_layers_wrapped_with_offload_checkpoint(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.25,
            gradient_checkpointing="on",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)

        assert isinstance(model.transformer.transformer_blocks[0], OffloadCheckpointLayer)
        strategy.cleanup()

    def test_forward_context_integration(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.5,
            gradient_checkpointing="on",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)

        ctx = strategy.forward_context()
        assert ctx is not None
        with ctx:
            _ = model.transformer(torch.randn(2, 64))
        strategy.cleanup()

    def test_cleanup_idempotent(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.5,
            gradient_checkpointing="on",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        strategy.cleanup()
        strategy.cleanup()
        assert strategy.conductor is None

    def test_zero_offload_fraction_skips_setup(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.0,
            gradient_checkpointing="off",
            enable_activation_offloading=False,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        assert strategy.conductor is None

    def test_activation_offload_without_layer_offload_still_sets_conductor(self):
        model = MockModel(num_layers=4)
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.0,
            gradient_checkpointing="cpu_offloaded",
            enable_activation_offloading=True,
            train_device="cpu",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        assert strategy.conductor is not None
        strategy.cleanup()

    def test_last_layer_activation_not_offloaded(self, monkeypatch):
        from eritrainer.memory.conductor import LayerOffloadConductor

        model = MockModel(num_layers=2)
        conductor = LayerOffloadConductor(
            module=model.transformer,
            train_device=torch.device("cuda"),
            temp_device=torch.device("cpu"),
            layer_offload_fraction=0.0,
            offload_activations=True,
            enable_async=False,
        )
        for layer in model.transformer.transformer_blocks:
            conductor.add_layer(layer)

        move_calls = {"count": 0}

        def _fake_move(value, device, non_blocking=False):
            del device, non_blocking
            move_calls["count"] += 1
            return value

        monkeypatch.setattr("eritrainer.memory.conductor._move_structure", _fake_move)

        conductor._active = True
        _ = conductor.after_layer(0, 1, torch.randn(2, 64))
        _ = conductor.after_layer(1, 1, torch.randn(2, 64))

        # Only the non-final wrapped layer should offload activations.
        assert move_calls["count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device-movement validation")
class TestConductorCUDA:
    def test_cuda_layer_movement(self):
        model = MockModel(num_layers=6).to("cuda:0")
        config = MemoryConfig(
            strategy="layer_offload",
            layer_offload_fraction=0.5,
            gradient_checkpointing="cpu_offloaded",
            enable_activation_offloading=False,
            enable_async_offloading=False,
            train_device="cuda:0",
            temp_device="cpu",
        )
        strategy = LayerOffloadStrategy(config)
        strategy.setup(model)
        assert strategy.conductor is not None

        x = torch.randn(2, 64, device="cuda:0", requires_grad=True)
        with strategy.forward_context():
            y = model.transformer(x)
            y.mean().backward()

        offloaded = strategy.conductor.get_offloaded_layers()
        assert len(offloaded) > 0
        strategy.cleanup()
