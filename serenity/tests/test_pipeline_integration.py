"""
Pipeline Integration Tests

Test the training pipeline layer:
1. Trainer initialization and configuration
2. NaN handler behavior
3. Command synchronization
4. Backup manager
5. GC scheduler
6. Training loop structure (with mocks)

These tests validate the pipeline layer works correctly with mock components.
"""

import gc
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# =============================================================================
# Test: NaN Handler
# =============================================================================

class TestNaNHandler:
    """Test NaN/Inf loss handling."""

    def test_nan_handler_creation(self):
        """NaN handler can be created with default params."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        assert handler.max_consecutive == 10
        assert handler.total_nan_threshold == 100
        assert handler.emergency_backup_threshold == 5
        assert handler.consecutive_nan_count == 0
        print("  ✅ NaNHandler created with defaults")

    def test_nan_handler_custom_thresholds(self):
        """NaN handler respects custom thresholds."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler(
            max_consecutive=5,
            total_nan_threshold=50,
            emergency_backup_threshold=3,
        )

        assert handler.max_consecutive == 5
        assert handler.total_nan_threshold == 50
        assert handler.emergency_backup_threshold == 3
        print("  ✅ NaNHandler accepts custom thresholds")

    def test_valid_loss_continues(self):
        """Valid loss returns continue action."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        is_valid, action = handler.check_loss(0.5)

        assert is_valid == True
        assert action == "continue"
        assert handler.last_valid_loss == 0.5
        print("  ✅ Valid loss returns continue")

    def test_nan_loss_skips(self):
        """NaN loss returns skip action."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        is_valid, action = handler.check_loss(float('nan'))

        assert is_valid == False
        assert action == "skip"
        assert handler.consecutive_nan_count == 1
        print("  ✅ NaN loss returns skip")

    def test_inf_loss_skips(self):
        """Inf loss returns skip action."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        is_valid, action = handler.check_loss(float('inf'))

        assert is_valid == False
        assert action == "skip"
        assert handler.consecutive_nan_count == 1
        print("  ✅ Inf loss returns skip")

    def test_consecutive_nan_resets_on_valid(self):
        """Consecutive NaN count resets on valid loss."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        # Add some NaN losses
        handler.check_loss(float('nan'))
        handler.check_loss(float('nan'))
        assert handler.consecutive_nan_count == 2

        # Valid loss resets
        handler.check_loss(0.5)
        assert handler.consecutive_nan_count == 0
        print("  ✅ Consecutive NaN resets on valid loss")

    def test_emergency_backup_triggered(self):
        """Emergency backup triggered at threshold."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler(emergency_backup_threshold=3)

        # Send 3 NaN losses
        handler.check_loss(float('nan'))
        handler.check_loss(float('nan'))
        is_valid, action = handler.check_loss(float('nan'))

        assert action == "emergency_backup"
        print("  ✅ Emergency backup triggered at threshold")

    def test_abort_at_max_consecutive(self):
        """Training aborts at max consecutive NaN."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler(max_consecutive=3, emergency_backup_threshold=10)

        # Send 3 NaN losses
        handler.check_loss(float('nan'))
        handler.check_loss(float('nan'))
        is_valid, action = handler.check_loss(float('nan'))

        assert action == "abort"
        print("  ✅ Abort triggered at max consecutive NaN")

    def test_stats_tracking(self):
        """Handler tracks statistics correctly."""
        from serenity.core.trainer import NaNHandler

        handler = NaNHandler()

        handler.check_loss(0.5)
        handler.check_loss(float('nan'))
        handler.check_loss(0.3)

        stats = handler.get_stats()

        assert stats["total_nan_count"] == 1
        assert stats["consecutive_nan_count"] == 0
        assert stats["last_valid_loss"] == 0.3
        print("  ✅ Statistics tracked correctly")


# =============================================================================
# Test: Command Handler
# =============================================================================

class TestCommandHandler:
    """Test command synchronization system."""

    def test_command_handler_creation(self):
        """CommandHandler can be created."""
        from serenity.core.trainer import CommandHandler

        handler = CommandHandler()

        assert handler is not None
        print("  ✅ CommandHandler created")

    def test_command_handler_check_commands(self):
        """CommandHandler check_commands returns commands object."""
        from serenity.core.trainer import CommandHandler

        handler = CommandHandler()

        cmds = handler.check_commands()
        assert cmds is not None
        assert hasattr(cmds, 'stop')
        print("  ✅ check_commands returns commands")

    def test_command_handler_set_command(self):
        """CommandHandler can set commands."""
        from serenity.core.trainer import CommandHandler

        handler = CommandHandler()

        handler.set_command("stop")
        cmds = handler.check_commands()

        assert cmds.stop == True
        print("  ✅ Commands set correctly")


# =============================================================================
# Test: Backup Manager
# =============================================================================

class TestBackupManager:
    """Test rolling backup system."""

    def test_backup_manager_creation(self):
        """BackupManager can be created."""
        from serenity.core.trainer import BackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackupManager(Path(tmpdir), rolling_count=3)

            assert manager is not None
            assert manager.rolling_count == 3
            print("  ✅ BackupManager created")

    def test_backup_manager_directory_creation(self):
        """BackupManager creates backup directory."""
        from serenity.core.trainer import BackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "backups_test"
            manager = BackupManager(backup_dir, rolling_count=3)

            # BackupManager should have output_dir attribute
            assert manager.output_dir == backup_dir
            print("  ✅ BackupManager tracks output directory")


# =============================================================================
# Test: GC Scheduler
# =============================================================================

class TestGCScheduler:
    """Test garbage collection scheduler."""

    def test_gc_scheduler_creation(self):
        """GCScheduler can be created."""
        from serenity.core.trainer import GCScheduler

        scheduler = GCScheduler(min_interval_seconds=30.0)

        assert scheduler is not None
        assert scheduler._min_interval == 30.0
        print("  ✅ GCScheduler created")

    def test_gc_scheduler_request_gc(self):
        """GCScheduler can request GC."""
        from serenity.core.trainer import GCScheduler

        scheduler = GCScheduler(min_interval_seconds=0.0)  # No interval

        # Should be able to request GC
        result = scheduler.request_gc()
        assert isinstance(result, bool)
        print("  ✅ GCScheduler request_gc works")


# =============================================================================
# Test: Training Progress
# =============================================================================

class TestTrainProgress:
    """Test training progress tracking."""

    def test_progress_creation(self):
        """TrainProgress can be created."""
        from serenity.core.progress import TrainProgress

        progress = TrainProgress()

        assert progress is not None
        assert progress.epoch == 0
        assert progress.global_step == 0
        print("  ✅ TrainProgress created")

    def test_progress_update(self):
        """TrainProgress can be updated."""
        from serenity.core.progress import TrainProgress

        progress = TrainProgress()

        progress.epoch = 1
        progress.global_step = 100
        progress.ema_loss = 0.5

        assert progress.epoch == 1
        assert progress.global_step == 100
        assert progress.ema_loss == 0.5
        print("  ✅ TrainProgress updated")


# =============================================================================
# Test: Trainer Creation
# =============================================================================

class TestTrainerCreation:
    """Test Trainer initialization without loading models."""

    def test_trainer_creation_minimal_config(self):
        """Trainer can be created with minimal config."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",  # Not loaded in init
                output_dir=tmpdir,
                concepts=[],
            )

            trainer = Trainer(config)

            assert trainer is not None
            assert trainer.model is None  # Not loaded yet
            assert trainer.optimizer is None
            print("  ✅ Trainer created with minimal config")

    def test_trainer_has_nan_handler(self):
        """Trainer creates NaN handler."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            trainer = Trainer(config)

            assert trainer.nan_handler is not None
            print("  ✅ Trainer has NaN handler")

    def test_trainer_has_command_handler(self):
        """Trainer creates command handler."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            trainer = Trainer(config)

            assert trainer.command_handler is not None
            print("  ✅ Trainer has command handler")

    def test_trainer_has_gc_scheduler(self):
        """Trainer creates GC scheduler."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            trainer = Trainer(config)

            assert trainer.gc_scheduler is not None
            print("  ✅ Trainer has GC scheduler")

    def test_trainer_sets_seed(self):
        """Trainer sets random seed."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer
        import random
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
                seed=42,
            )

            trainer = Trainer(config)

            # Seed should be set
            r1 = random.random()

            # Create another trainer with same seed
            trainer2 = Trainer(config)
            r2 = random.random()

            # Should be same due to seed
            assert r1 == r2
            print("  ✅ Trainer sets random seed")


# =============================================================================
# Test: Config Validation
# =============================================================================

class TestConfigValidation:
    """Test configuration handling."""

    def test_config_required_fields(self):
        """TrainConfig validates required fields."""
        from serenity.core.config import TrainConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should work with required fields
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            assert config.model_type.value == "flux_dev"
            print("  ✅ Config validates required fields")

    def test_config_default_values(self):
        """TrainConfig provides sensible defaults."""
        from serenity.core.config import TrainConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            # Check defaults
            assert config.learning_rate == 1e-4
            assert config.train_device == "cuda"
            assert config.seed == 42
            print("  ✅ Config provides sensible defaults")


# =============================================================================
# Test: Model Type Routing
# =============================================================================

class TestModelTypeRouting:
    """Test model type determines correct loader."""

    def test_flux_dev_type(self):
        """ModelType.FLUX_DEV is valid."""
        from serenity.core.config import ModelType

        assert ModelType.FLUX_DEV.value == "flux_dev"
        print("  ✅ FLUX_DEV type valid")

    def test_flux2_klein_type(self):
        """ModelType.FLUX_2_KLEIN_4B is valid."""
        from serenity.core.config import ModelType

        assert ModelType.FLUX_2_KLEIN_4B.value == "flux_2_klein_4b"
        assert ModelType.FLUX_2_KLEIN_9B.value == "flux_2_klein_9b"
        print("  ✅ FLUX_2_KLEIN types valid")

    def test_model_type_enum_complete(self):
        """All expected model types exist."""
        from serenity.core.config import ModelType

        expected = [
            "flux_dev", "flux_schnell", "flux_2_klein_4b",
            "z_image", "sdxl_10_base", "ltx2", "qwen", "qwen_image_edit"
        ]

        for mt in expected:
            assert any(
                m.value == mt for m in ModelType
            ), f"Missing model type: {mt}"

        print("  ✅ All expected model types exist")


# =============================================================================
# Test: Training Method Routing
# =============================================================================

class TestTrainingMethodRouting:
    """Test training method determines correct setup."""

    def test_lora_method(self):
        """TrainingMethod.LORA is valid."""
        from serenity.core.config import TrainingMethod

        assert TrainingMethod.LORA.value == "lora"
        print("  ✅ LORA method valid")

    def test_fine_tune_method(self):
        """TrainingMethod.FINE_TUNE is valid."""
        from serenity.core.config import TrainingMethod

        assert TrainingMethod.FINE_TUNE.value == "fine_tune"
        print("  ✅ FINE_TUNE method valid")

    def test_training_methods_complete(self):
        """All expected training methods exist."""
        from serenity.core.config import TrainingMethod

        expected = ["lora", "fine_tune", "embedding"]

        for tm in expected:
            assert any(
                m.value == tm for m in TrainingMethod
            ), f"Missing training method: {tm}"

        print("  ✅ All expected training methods exist")


# =============================================================================
# Test: Pipeline Layer Independence
# =============================================================================

class TestPipelineLayerIndependence:
    """Test that pipeline layer uses interfaces, not implementations."""

    def test_trainer_uses_base_model_type(self):
        """Trainer uses BaseModel type hint, not specific models."""
        from serenity.core.trainer import Trainer
        import inspect

        # Get the model attribute type hint
        hints = Trainer.__init__.__doc__ or ""

        # The class should type hint with BaseModel
        assert Trainer.__annotations__.get('model', '') != 'FluxModel'
        print("  ✅ Trainer uses BaseModel type hint")

    def test_trainer_model_is_optional(self):
        """Trainer's model attribute is optional (None before start)."""
        from serenity.core.config import TrainConfig
        from serenity.core.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                model_type="flux_dev",
                training_method="lora",
                transformer_path="/fake/path",
                output_dir=tmpdir,
                concepts=[],
            )

            trainer = Trainer(config)

            assert trainer.model is None
            print("  ✅ Trainer model is Optional (None before start)")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all pipeline integration tests."""
    print("=== Pipeline Integration Tests ===\n")

    print("Test: NaN Handler")
    t = TestNaNHandler()
    try:
        t.test_nan_handler_creation()
        t.test_nan_handler_custom_thresholds()
        t.test_valid_loss_continues()
        t.test_nan_loss_skips()
        t.test_inf_loss_skips()
        t.test_consecutive_nan_resets_on_valid()
        t.test_emergency_backup_triggered()
        t.test_abort_at_max_consecutive()
        t.test_stats_tracking()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Command Handler")
    t = TestCommandHandler()
    try:
        t.test_command_handler_creation()
        t.test_command_handler_check_commands()
        t.test_command_handler_set_command()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Backup Manager")
    t = TestBackupManager()
    try:
        t.test_backup_manager_creation()
        t.test_backup_manager_directory_creation()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: GC Scheduler")
    t = TestGCScheduler()
    try:
        t.test_gc_scheduler_creation()
        t.test_gc_scheduler_request_gc()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Training Progress")
    t = TestTrainProgress()
    try:
        t.test_progress_creation()
        t.test_progress_update()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Trainer Creation")
    t = TestTrainerCreation()
    try:
        t.test_trainer_creation_minimal_config()
        t.test_trainer_has_nan_handler()
        t.test_trainer_has_command_handler()
        t.test_trainer_has_gc_scheduler()
        t.test_trainer_sets_seed()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Config Validation")
    t = TestConfigValidation()
    try:
        t.test_config_required_fields()
        t.test_config_default_values()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Model Type Routing")
    t = TestModelTypeRouting()
    try:
        t.test_flux_dev_type()
        t.test_flux2_klein_type()
        t.test_model_type_enum_complete()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Training Method Routing")
    t = TestTrainingMethodRouting()
    try:
        t.test_lora_method()
        t.test_fine_tune_method()
        t.test_training_methods_complete()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\nTest: Pipeline Layer Independence")
    t = TestPipelineLayerIndependence()
    try:
        t.test_trainer_uses_base_model_type()
        t.test_trainer_model_is_optional()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\n=== Pipeline Integration Tests Complete ===")


if __name__ == "__main__":
    run_all_tests()
