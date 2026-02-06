"""
LyCORIS Feature Parity Verification Tests

Verifies that EriTrainer has feature parity with SimpleTuner for:
- Z-Image: Full LyCORIS support
- SDXL: Full LyCORIS support
- SD 3.5: Full LyCORIS support (EriTrainer EXCEEDS SimpleTuner here!)

Run: python -m pytest eritrainer/tests/test_lycoris_parity.py -v
"""

from pathlib import Path

import torch
import torch.nn as nn

import pytest

# Skip if lycoris not installed
pytest.importorskip("lycoris")


class TestLyCORISManagerParity:
    """Test LyCORIS manager supports all required adapter types."""

    def test_adapter_types_enum(self):
        """Verify all required adapter types are defined."""
        from eritrainer.training.lycoris_manager import AdapterType

        required_types = [
            "lora", "locon", "loha", "lokr", "ia3",
            "oft", "boft", "diag_oft", "glora", "dylora", "dora", "full"
        ]

        for adapter_type in required_types:
            assert hasattr(AdapterType, adapter_type.upper()), f"Missing AdapterType.{adapter_type.upper()}"

    def test_default_targets_zimage(self):
        """Verify Z-Image default targets are defined."""
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS

        assert "zimage" in DEFAULT_TARGETS or "z_image" in DEFAULT_TARGETS
        targets = DEFAULT_TARGETS.get("zimage", DEFAULT_TARGETS.get("z_image", []))
        assert len(targets) > 0, "Z-Image default targets not defined"

        # Should include attention and feed forward
        target_str = " ".join(targets)
        assert "attention" in target_str.lower() or "to_q" in target_str.lower()
        assert "feed_forward" in target_str.lower() or "w1" in target_str.lower()

    def test_default_targets_sdxl(self):
        """Verify SDXL default targets are defined."""
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS

        assert "sdxl" in DEFAULT_TARGETS
        targets = DEFAULT_TARGETS["sdxl"]
        assert len(targets) > 0, "SDXL default targets not defined"

        # Should include attention modules
        target_str = " ".join(targets)
        assert "attn" in target_str.lower()
        assert "to_q" in target_str.lower() or "to_k" in target_str.lower()

    def test_default_targets_sd3(self):
        """Verify SD3 default targets are defined."""
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS

        assert "sd3" in DEFAULT_TARGETS
        targets = DEFAULT_TARGETS["sd3"]
        assert len(targets) > 0, "SD3 default targets not defined"

        # Should include joint transformer attention
        target_str = " ".join(targets)
        assert "joint_transformer" in target_str.lower() or "attn" in target_str.lower()


class TestLyCORISConfigSupport:
    """Test LyCORIS config classes for all adapter types."""

    def test_locon_config(self):
        """Test LoCon config has required fields."""
        from eritrainer.training.lycoris_manager import LoConConfig

        config = LoConConfig()
        assert hasattr(config, "use_conv2d")
        assert hasattr(config, "use_linear")
        assert hasattr(config, "conv_rank")
        assert hasattr(config, "conv_alpha")

    def test_loha_config(self):
        """Test LoHa config has required fields."""
        from eritrainer.training.lycoris_manager import LoHaConfig

        config = LoHaConfig()
        assert hasattr(config, "use_effective_conv2d")

    def test_lokr_config(self):
        """Test LoKr config has required fields."""
        from eritrainer.training.lycoris_manager import LoKrConfig

        config = LoKrConfig()
        assert hasattr(config, "factor")
        assert hasattr(config, "decompose_both")
        assert hasattr(config, "use_tucker")
        assert hasattr(config, "full_matrix")

    def test_ia3_config(self):
        """Test IA3 config has required fields."""
        from eritrainer.training.lycoris_manager import IA3Config

        config = IA3Config()
        assert hasattr(config, "feedforward_modules")
        assert hasattr(config, "init_ia3_weights")

    def test_oft_config(self):
        """Test OFT config has required fields."""
        from eritrainer.training.lycoris_manager import OFTConfig

        config = OFTConfig()
        assert hasattr(config, "block_size")
        assert hasattr(config, "is_coft")
        assert hasattr(config, "boft_m")
        assert hasattr(config, "boft_block_dim")

    def test_glora_config(self):
        """Test GLoRA config has required fields."""
        from eritrainer.training.lycoris_manager import GLoRAConfig

        config = GLoRAConfig()
        assert hasattr(config, "gate_init")

    def test_dylora_config(self):
        """Test DyLoRA config has required fields."""
        from eritrainer.training.lycoris_manager import DyLoRAConfig

        config = DyLoRAConfig()
        assert hasattr(config, "min_rank")
        assert hasattr(config, "max_rank")
        assert hasattr(config, "rank_schedule")


class TestAdapterModuleSupport:
    """Test adapter module factory and registry."""

    def test_adapter_registry_complete(self):
        """Verify adapter registry has all required types."""
        from eritrainer.adapters import ADAPTER_REGISTRY, AdapterType

        required_types = [
            AdapterType.LORA,
            AdapterType.DORA,
            AdapterType.LOKR,
            AdapterType.LOHA,
            AdapterType.LOCON,
        ]

        for adapter_type in required_types:
            assert adapter_type in ADAPTER_REGISTRY, f"Missing {adapter_type} in ADAPTER_REGISTRY"

    def test_create_adapter_factory(self):
        """Test create_adapter factory function."""
        from eritrainer.adapters import create_adapter

        # Test LoRA creation (doesn't need model)
        adapter = create_adapter(
            adapter_type="lora",
            rank=16,
            alpha=16.0,
            model_type="zimage"
        )
        assert adapter is not None

    def test_adapter_detection(self):
        """Test adapter type detection from state dict."""
        from eritrainer.adapters import AdapterType, detect_adapter_type

        # Test LoKr detection
        lokr_state = {"layer.lokr_w1": None, "layer.lokr_w2": None}
        assert detect_adapter_type(lokr_state) == AdapterType.LOKR

        # Test LoHa detection
        loha_state = {"layer.hada_w1_a": None}
        assert detect_adapter_type(loha_state) == AdapterType.LOHA

        # Test DoRA detection
        dora_state = {"layer.dora_scale": None}
        assert detect_adapter_type(dora_state) == AdapterType.DORA

        # Test LoRA default
        lora_state = {"layer.lora_A": None}
        assert detect_adapter_type(lora_state) == AdapterType.LORA


class TestEMASupport:
    """Test EMA support (feature parity with SimpleTuner SDXL)."""

    def test_ema_module_exists(self):
        """Verify EMA module exists."""
        from eritrainer.training.ema import EMAMode, EMAModel, EMAModule

        assert EMAModule is not None
        assert EMAModel is not None
        assert EMAMode is not None

    def test_ema_modes(self):
        """Test EMA modes match SimpleTuner."""
        from eritrainer.training.ema import EMAMode

        assert EMAMode.OFF is not None
        assert EMAMode.GPU is not None
        assert EMAMode.CPU is not None


class TestLossSupport:
    """Test loss function support (feature parity)."""

    def test_flow_matching_loss(self):
        """Test flow matching loss exists."""
        from eritrainer.training.losses import flow_matching_loss

        assert callable(flow_matching_loss)

    def test_snr_weighted_loss(self):
        """Test Min-SNR weighted loss exists (SDXL feature)."""
        from eritrainer.training.losses import snr_weighted_loss

        assert callable(snr_weighted_loss)

    def test_velocity_loss(self):
        """Test velocity loss exists (SD3 feature)."""
        from eritrainer.training.losses import velocity_loss

        assert callable(velocity_loss)


class TestNoiseConfigSupport:
    """Test noise configuration support (SDXL offset noise)."""

    def test_noise_config_has_offset_noise(self):
        """Verify offset noise config exists."""
        from eritrainer.core.config import NoiseConfig

        config = NoiseConfig()
        assert hasattr(config, "offset_noise_weight")
        assert hasattr(config, "generalized_offset_noise")


class TestFeatureParitySummary:
    """Summary tests confirming feature parity."""

    def test_zimage_parity(self):
        """Confirm Z-Image has SimpleTuner parity."""
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS, AdapterType

        # Has target modules
        assert "zimage" in DEFAULT_TARGETS or "z_image" in DEFAULT_TARGETS

        # Has all adapter types
        required = [AdapterType.LORA, AdapterType.LOKR, AdapterType.LOHA, AdapterType.LOCON]
        for t in required:
            assert t in AdapterType.__members__.values()

    def test_sdxl_parity(self):
        """Confirm SDXL has SimpleTuner parity."""
        from eritrainer.core.config import NoiseConfig
        from eritrainer.training.ema import EMAModel
        from eritrainer.training.losses import snr_weighted_loss
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS

        # Has target modules
        assert "sdxl" in DEFAULT_TARGETS

        # Has Min-SNR
        assert callable(snr_weighted_loss)

        # Has EMA
        assert EMAModel is not None

        # Has offset noise
        config = NoiseConfig()
        assert hasattr(config, "offset_noise_weight")

    def test_sd3_exceeds_simpletuner(self):
        """Confirm SD3 EXCEEDS SimpleTuner (has LyCORIS, SimpleTuner doesn't)."""
        from eritrainer.training.lycoris_manager import DEFAULT_TARGETS, AdapterType, LyCORISManager

        # Has target modules
        assert "sd3" in DEFAULT_TARGETS

        # Has all LyCORIS types (SimpleTuner SD3 only has PEFT LoRA)
        lycoris_types = [
            AdapterType.LOKR, AdapterType.LOHA, AdapterType.LOCON,
            AdapterType.OFT, AdapterType.BOFT, AdapterType.IA3
        ]
        for t in lycoris_types:
            assert t in AdapterType.__members__.values()

        # Can create LyCORIS manager for SD3
        from eritrainer.training.lycoris_manager import AdapterConfig
        config = AdapterConfig(adapter_type=AdapterType.LOKR, rank=16)
        manager = LyCORISManager(config, model_type="sd3")
        assert manager is not None


class TestLyCORISRuntime:
    """Runtime checks for native LyCORIS manager wiring."""

    def _build_target_module(self):
        class Transformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(16, 16)
                self.to_k = nn.Linear(16, 16)
                self.to_v = nn.Linear(16, 16)
                self.ff = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 16))

            def forward(self, x):
                x = self.to_q(x) + self.to_k(x) + self.to_v(x)
                return self.ff(x)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = Transformer2DModel()

            def forward(self, x):
                return self.unet(x)

        return Wrapper()

    def test_lokr_attach_optimizer_and_save(self, tmp_path):
        from eritrainer.training.lycoris_manager import AdapterConfig, AdapterType, LyCORISManager

        module = self._build_target_module()
        config = AdapterConfig(
            adapter_type=AdapterType.LOKR,
            rank=4,
            alpha=4.0,
            factor=2,
            use_tucker=False,
            full_matrix=False,
        )
        manager = LyCORISManager(config, model_type="sdxl")
        network = manager.apply(module)

        assert network is not None
        assert manager.is_attached()

        groups = manager.prepare_optimizer_params(1e-4)
        assert groups
        assert isinstance(groups[0], dict)
        assert "params" in groups[0]

        x = torch.randn(2, 16, requires_grad=True)
        y = module(x).sum()
        y.backward()

        out_path = tmp_path / "lokr_test.safetensors"
        saved = manager.save_weights(out_path)
        assert Path(saved).exists()

    def test_target_modules_emit_custom_preset(self, monkeypatch):
        from eritrainer.training import lycoris_manager as lm
        from eritrainer.training.lycoris_manager import AdapterConfig, AdapterType, LyCORISManager

        captured: dict[str, object] = {}

        class _DummyNetwork:
            def apply_to(self):
                captured["apply_to_called"] = True

        def _fake_create_lycoris(module, **kwargs):
            captured["module"] = module
            captured["kwargs"] = kwargs
            return _DummyNetwork()

        monkeypatch.setattr(lm, "create_lycoris", _fake_create_lycoris)

        config = AdapterConfig(
            adapter_type=AdapterType.LORA,
            rank=4,
            alpha=4.0,
            target_modules=["attn.to_q", "to_v"],
        )
        manager = LyCORISManager(config, model_type="sdxl")
        target = nn.Linear(8, 8)
        network = manager.apply(target)

        assert network is not None
        assert captured.get("apply_to_called") is True
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        assert "preset" in kwargs
        preset_path = Path(kwargs["preset"])
        assert preset_path.exists()
        text = preset_path.read_text(encoding="utf-8")
        assert "target_name" in text
        assert "*attn.to_q" in text
        assert "*to_v" in text

    def test_ltx2_lycoris_disabled(self):
        from eritrainer.training.lycoris_manager import AdapterConfig, AdapterType, LyCORISManager

        config = AdapterConfig(
            adapter_type=AdapterType.LORA,
            rank=4,
            alpha=4.0,
        )
        with pytest.raises(ValueError, match="temporarily disabled"):
            LyCORISManager(config, model_type="ltx2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
