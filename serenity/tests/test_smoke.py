"""
Smoke Tests - Quick validation that core components can be imported and instantiated.

Run with: python -m pytest serenity/tests/test_smoke.py -v
"""

import pytest
import torch


# =============================================================================
# Test 1: Import Tests - ACTUAL MODULES
# =============================================================================

class TestImports:
    """Verify all key modules can be imported."""

    def test_import_interfaces(self):
        """Core interfaces import."""
        from serenity.core.interfaces import ModelType, BaseModel
        assert hasattr(ModelType, 'FLUX_2_KLEIN_4B')
        assert hasattr(ModelType, 'FLUX_2_KLEIN_9B')
        assert hasattr(ModelType, 'ZIMAGE')
        assert hasattr(ModelType, 'SDXL')

    def test_import_all_models(self):
        """All model classes import."""
        from serenity.models import (
            Flux1Model,
            Flux2Model,
            Flux2KleinModel,
            FluxKleinModel,
            ZImageModel,
            SD15Model,
            SDXLModel,
        )
        assert Flux1Model is not None
        assert Flux2Model is not None
        assert Flux2KleinModel is not None
        assert FluxKleinModel is not None
        assert ZImageModel is not None
        assert SD15Model is not None
        assert SDXLModel is not None

    def test_import_flux2_klein_model(self):
        """FLUX.2 Klein model imports."""
        from serenity.models.flux2_klein import Flux2KleinModel, Flux2KleinModelLoader
        assert Flux2KleinModel is not None
        assert Flux2KleinModelLoader is not None

    def test_import_trainer(self):
        """Trainer class imports."""
        from serenity.training.trainer import Trainer
        assert Trainer is not None

    def test_import_config(self):
        """Config module imports."""
        from serenity.core.config import TrainerConfig, load_config
        assert TrainerConfig is not None
        assert load_config is not None

    def test_import_sampling(self):
        """Sampling module imports."""
        from serenity.sampling import (
            create_sampler,
            FluxSampler,
            Flux2Sampler,
            ZImageSampler,
            SD15Sampler,
            SDXLSampler,
        )
        assert create_sampler is not None
        assert FluxSampler is not None
        assert Flux2Sampler is not None
        assert ZImageSampler is not None

    def test_import_adapters(self):
        """Adapter module imports."""
        from serenity.adapters import create_adapter
        from serenity.adapters.base import AdapterProtocol
        assert create_adapter is not None
        assert AdapterProtocol is not None

    def test_import_data_pipeline(self):
        """Data pipeline imports."""
        from serenity.data import (
            BucketBatchSampler,
            EriDataset,
            create_dataloader,
            CacheManager,
        )
        assert BucketBatchSampler is not None
        assert EriDataset is not None
        assert create_dataloader is not None

    def test_import_cli(self):
        """CLI imports."""
        from serenity.cli.commands import train_command
        assert train_command is not None


# =============================================================================
# Test 2: Model Type Enum
# =============================================================================

class TestModelTypeEnum:
    """Verify ModelType enum has all required values."""

    def test_flux_1_types(self):
        """Flux 1 types exist."""
        from serenity.core.interfaces import ModelType
        assert ModelType.FLUX_DEV.value == "flux_dev"
        assert ModelType.FLUX_SCHNELL.value == "flux_schnell"

    def test_flux_2_types(self):
        """Flux 2 types exist."""
        from serenity.core.interfaces import ModelType
        assert ModelType.FLUX_2_DEV.value == "flux_2_dev"
        assert ModelType.FLUX_2_KLEIN.value == "flux_2_klein"
        assert ModelType.FLUX_2_KLEIN_4B.value == "flux_2_klein_4b"
        assert ModelType.FLUX_2_KLEIN_9B.value == "flux_2_klein_9b"
        assert ModelType.FLUX_2_KLEIN_4B_BASE.value == "flux_2_klein_4b_base"
        assert ModelType.FLUX_2_KLEIN_9B_BASE.value == "flux_2_klein_9b_base"

    def test_sd_types(self):
        """SD types exist."""
        from serenity.core.interfaces import ModelType
        assert ModelType.SD15.value == "sd15"
        assert ModelType.SDXL.value == "sdxl"

    def test_other_types(self):
        """Other model types exist."""
        from serenity.core.interfaces import ModelType
        assert ModelType.ZIMAGE.value == "zimage"


# =============================================================================
# Test 3: Flux2KleinModel Structure
# =============================================================================

class TestFlux2KleinModelStructure:
    """Test Flux2KleinModel class structure without loading weights."""

    def test_flux2_klein_config_values(self):
        """FLUX2_KLEIN_CONFIG has correct values."""
        from serenity.models.flux2_klein import FLUX2_KLEIN_CONFIG

        # Klein 4B
        config_4b = FLUX2_KLEIN_CONFIG['klein-4b']
        assert config_4b['double_blocks'] == 5
        assert config_4b['single_blocks'] == 20
        assert config_4b['total_blocks'] == 25
        assert config_4b['max_swappable'] == 24
        assert config_4b['joint_attention_dim'] == 7680
        assert config_4b['text_encoder_layers'] == (9, 18, 27)

        # Klein 9B
        config_9b = FLUX2_KLEIN_CONFIG['klein-9b']
        assert config_9b['double_blocks'] == 8
        assert config_9b['single_blocks'] == 24
        assert config_9b['total_blocks'] == 32
        assert config_9b['max_swappable'] == 31
        assert config_9b['joint_attention_dim'] == 12288


# =============================================================================
# Test 4: Training Utilities
# =============================================================================

class TestTrainingUtilities:
    """Test training utility functions."""

    def test_torch_gc(self):
        """torch_gc utility works."""
        from serenity.training.torch_util import torch_gc
        # Should not raise
        torch_gc()

    def test_device_equals(self):
        """device_equals utility works."""
        from serenity.training.torch_util import device_equals

        cpu = torch.device('cpu')
        assert device_equals(cpu, cpu)
        assert device_equals(cpu, torch.device('cpu'))

    def test_loss_functions(self):
        """Loss functions can be imported."""
        from serenity.training.loss import compute_loss
        assert compute_loss is not None


# =============================================================================
# Test 5: Sampling Factory
# =============================================================================

class TestSamplingFactory:
    """Test sampling factory function dispatches correctly."""

    def test_create_sampler_factory_exists(self):
        """create_sampler factory function exists."""
        from serenity.sampling import create_sampler
        assert callable(create_sampler)


# =============================================================================
# Test 6: Data Pipeline Components
# =============================================================================

class TestDataPipeline:
    """Test data pipeline components."""

    def test_bucket_types(self):
        """Bucket types exist."""
        from serenity.data import Bucket, BucketManager
        assert Bucket is not None
        assert BucketManager is not None

    def test_concept_scanner(self):
        """ConceptScanner exists."""
        from serenity.data import ConceptScanner
        assert ConceptScanner is not None

    def test_caption_loader(self):
        """CaptionLoader exists."""
        from serenity.data import CaptionLoader
        assert CaptionLoader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
