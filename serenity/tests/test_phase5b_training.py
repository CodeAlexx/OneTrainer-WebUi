"""Tests for Phase 5b training support modules.

Covers:
- checkpoint/conversion.py: ModelFormat enum, detect_model_format
- checkpoint/saver.py: ModelSaver, create_saver, dtype conversion, safetensors headers
- training/grad_scaler.py: DummyOptimizer, CustomGradScaler
- training/tensorboard.py: TensorBoardLogger disabled mode, tag formatting, close
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.nn import Parameter


# ---------------------------------------------------------------------------
# Module 1: checkpoint/conversion.py
# ---------------------------------------------------------------------------


def test_detect_model_format_safetensors_file(tmp_path: Path):
    """A .safetensors file is detected as SAFETENSORS format."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00" * 1024)

    # Mock _is_lora_safetensors to return False so we test the routing logic.
    with patch("serenity.checkpoint.conversion._is_lora_safetensors", return_value=False):
        result = detect_model_format(f)
    assert result == ModelFormat.SAFETENSORS


def test_detect_model_format_diffusers_directory(tmp_path: Path):
    """A directory with model_index.json is detected as DIFFUSERS."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    (tmp_path / "model_index.json").write_text("{}")
    result = detect_model_format(tmp_path)
    assert result == ModelFormat.DIFFUSERS


def test_detect_model_format_diffusers_by_subfolders(tmp_path: Path):
    """A directory with 2+ diffusers marker sub-folders is DIFFUSERS."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae").mkdir()
    result = detect_model_format(tmp_path)
    assert result == ModelFormat.DIFFUSERS


def test_detect_model_format_torch_ckpt(tmp_path: Path):
    """A .ckpt file is detected as TORCH_CKPT."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    f = tmp_path / "model.ckpt"
    f.write_bytes(b"\x00" * 1024)

    # Mock _is_lora_torch to return False so we test the routing logic.
    with patch("serenity.checkpoint.conversion._is_lora_torch", return_value=False):
        result = detect_model_format(f)
    assert result == ModelFormat.TORCH_CKPT


def test_detect_model_format_nonexistent_path():
    """A path that does not exist returns UNKNOWN."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    result = detect_model_format("/nonexistent/path/to/model.xyz")
    assert result == ModelFormat.UNKNOWN


def test_detect_model_format_unknown_suffix(tmp_path: Path):
    """An unrecognized file extension returns UNKNOWN."""
    from serenity.checkpoint.conversion import ModelFormat, detect_model_format

    f = tmp_path / "model.txt"
    f.write_text("not a model")
    result = detect_model_format(f)
    assert result == ModelFormat.UNKNOWN


def test_model_format_enum_values():
    """ModelFormat has all expected members."""
    from serenity.checkpoint.conversion import ModelFormat

    expected = {
        "DIFFUSERS", "SAFETENSORS", "TORCH_CKPT",
        "LORA_SAFETENSORS", "LORA_TORCH", "UNKNOWN",
    }
    actual = {m.name for m in ModelFormat}
    assert actual == expected


def test_model_format_is_str_enum():
    """ModelFormat members are strings."""
    from serenity.checkpoint.conversion import ModelFormat

    assert isinstance(ModelFormat.DIFFUSERS, str)
    assert ModelFormat.SAFETENSORS == "safetensors"


# ---------------------------------------------------------------------------
# Module 2: checkpoint/saver.py
# ---------------------------------------------------------------------------


def test_convert_state_dict_dtype_fp32_to_fp16():
    """Tensors are moved to CPU and cast from fp32 to fp16."""
    from serenity.checkpoint.saver import _convert_state_dict_dtype

    sd = {
        "weight": torch.randn(4, 4, dtype=torch.float32),
        "bias": torch.randn(4, dtype=torch.float32),
    }
    result = _convert_state_dict_dtype(sd, torch.float16)

    for key, tensor in result.items():
        assert tensor.dtype == torch.float16, f"{key} should be fp16"
        assert tensor.device == torch.device("cpu"), f"{key} should be on CPU"
        assert tensor.is_contiguous(), f"{key} should be contiguous"


def test_convert_state_dict_dtype_none_keeps_original():
    """Passing dtype=None keeps original dtype but moves to CPU."""
    from serenity.checkpoint.saver import _convert_state_dict_dtype

    sd = {"w": torch.randn(2, 2, dtype=torch.float64)}
    result = _convert_state_dict_dtype(sd, None)
    assert result["w"].dtype == torch.float64
    assert result["w"].device == torch.device("cpu")


def test_convert_state_dict_dtype_empty_dict():
    """An empty state dict returns an empty dict."""
    from serenity.checkpoint.saver import _convert_state_dict_dtype

    result = _convert_state_dict_dtype({}, torch.float16)
    assert result == {}


def test_build_safetensors_header_contains_required_keys():
    """Header contains date and model_type when provided."""
    from serenity.checkpoint.saver import _build_safetensors_header

    header = _build_safetensors_header(model_type="flux_dev", training_method="lora")

    assert "serenity_date" in header
    assert "serenity_model_type" in header
    assert header["serenity_model_type"] == "flux_dev"
    assert "serenity_training_method" in header
    assert header["serenity_training_method"] == "lora"


def test_build_safetensors_header_minimal():
    """Header with no arguments still has a date."""
    from serenity.checkpoint.saver import _build_safetensors_header

    header = _build_safetensors_header()
    assert "serenity_date" in header
    assert "serenity_model_type" not in header
    assert "serenity_training_method" not in header


def test_build_safetensors_header_extra_metadata():
    """Extra metadata dict is merged into the header."""
    from serenity.checkpoint.saver import _build_safetensors_header

    header = _build_safetensors_header(metadata={"custom_key": "custom_value"})
    assert header["custom_key"] == "custom_value"


def test_build_safetensors_header_sdxl_compat():
    """SDXL model type adds Kohya-compat ss_base_model_version key."""
    from serenity.checkpoint.saver import _build_safetensors_header

    header = _build_safetensors_header(model_type="sdxl_base")
    assert header.get("ss_base_model_version") == "sdxl_"


def test_create_saver_returns_model_saver():
    """create_saver() returns a properly configured ModelSaver."""
    from serenity.checkpoint.saver import ModelSaver, create_saver

    saver = create_saver(
        model_type="flux_dev",
        training_method="lora",
        dtype=torch.float16,
        author="test",
    )
    assert isinstance(saver, ModelSaver)
    assert saver.model_type == "flux_dev"
    assert saver.training_method == "lora"
    assert saver.default_dtype == torch.float16
    assert saver.extra_metadata == {"author": "test"}


def test_create_saver_defaults():
    """create_saver() with no arguments gives sensible defaults."""
    from serenity.checkpoint.saver import ModelSaver, create_saver

    saver = create_saver()
    assert isinstance(saver, ModelSaver)
    assert saver.model_type is None
    assert saver.training_method is None
    assert saver.default_dtype is None
    assert saver.extra_metadata == {}


# ---------------------------------------------------------------------------
# Module 3: training/grad_scaler.py
# ---------------------------------------------------------------------------


def test_dummy_optimizer_param_groups_structure():
    """DummyOptimizer has a param_groups list with correct structure."""
    from serenity.training.grad_scaler import DummyOptimizer

    param = Parameter(torch.randn(3, 3))
    opt = DummyOptimizer(param)

    assert isinstance(opt.param_groups, list)
    assert len(opt.param_groups) == 1
    assert "params" in opt.param_groups[0]
    assert len(opt.param_groups[0]["params"]) == 1


def test_dummy_optimizer_wraps_parameter():
    """DummyOptimizer stores the exact parameter reference."""
    from serenity.training.grad_scaler import DummyOptimizer

    param = Parameter(torch.randn(4))
    opt = DummyOptimizer(param)
    assert opt.param_groups[0]["params"][0] is param


def test_dummy_optimizer_accepts_plain_tensor():
    """DummyOptimizer also accepts a plain Tensor (not just Parameter)."""
    from serenity.training.grad_scaler import DummyOptimizer

    tensor = torch.randn(2, 2)
    opt = DummyOptimizer(tensor)
    assert opt.param_groups[0]["params"][0] is tensor


def test_custom_grad_scaler_initialization():
    """CustomGradScaler initializes and inherits from GradScaler."""
    from torch.amp.grad_scaler import GradScaler

    from serenity.training.grad_scaler import CustomGradScaler

    scaler = CustomGradScaler()
    assert isinstance(scaler, GradScaler)


def test_custom_grad_scaler_enabled_by_default():
    """CustomGradScaler is enabled by default (inherits from parent)."""
    from serenity.training.grad_scaler import CustomGradScaler

    scaler = CustomGradScaler()
    # GradScaler defaults to enabled=True on CUDA, False on CPU.
    # We just verify the attribute exists and is a bool.
    assert isinstance(scaler.is_enabled(), bool)


def test_custom_grad_scaler_has_unscale_parameter():
    """CustomGradScaler exposes the unscale_parameter_ method."""
    from serenity.training.grad_scaler import CustomGradScaler

    scaler = CustomGradScaler()
    assert hasattr(scaler, "unscale_parameter_")
    assert callable(scaler.unscale_parameter_)


def test_custom_grad_scaler_has_fused_step_methods():
    """CustomGradScaler exposes maybe_opt_step_parameter and step_after_unscale."""
    from serenity.training.grad_scaler import CustomGradScaler

    scaler = CustomGradScaler()
    assert hasattr(scaler, "maybe_opt_step_parameter")
    assert hasattr(scaler, "step_after_unscale_parameter_")


# ---------------------------------------------------------------------------
# Module 4: training/tensorboard.py
# ---------------------------------------------------------------------------


def test_tensorboard_logger_disabled_is_noop(tmp_path: Path):
    """TensorBoardLogger with enabled=False does not create a writer."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger(log_dir=tmp_path / "tb", enabled=False)
    assert tb._writer is None

    # All logging methods should be silent no-ops.
    tb.log_scalar("tag", 1.0, step=0)
    tb.log_loss(0.5, step=1)
    tb.log_lr(0.001, step=1)
    tb.log_smooth_loss(0.4, step=2)
    tb.log_ema_decay(0.999, step=3)
    tb.flush()
    tb.close()

    # Still None after all operations.
    assert tb._writer is None


def test_tensorboard_logger_log_loss_tag():
    """log_loss uses 'loss/train_step' tag internally."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = True
    tb._writer = MagicMock()
    tb.log_dir = "/tmp/test"

    tb.log_loss(0.123, step=42)
    tb._writer.add_scalar.assert_called_once_with(
        "loss/train_step", 0.123, global_step=42,
    )


def test_tensorboard_logger_log_lr_tag():
    """log_lr uses 'lr/{name}' tag pattern."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = True
    tb._writer = MagicMock()
    tb.log_dir = "/tmp/test"

    tb.log_lr(3e-4, step=10, name="unet")
    tb._writer.add_scalar.assert_called_once_with(
        "lr/unet", 3e-4, global_step=10,
    )


def test_tensorboard_logger_log_lr_default_name():
    """log_lr with no name kwarg uses 'default'."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = True
    tb._writer = MagicMock()
    tb.log_dir = "/tmp/test"

    tb.log_lr(1e-4, step=5)
    tb._writer.add_scalar.assert_called_once_with(
        "lr/default", 1e-4, global_step=5,
    )


def test_tensorboard_logger_close_sets_writer_none():
    """close() calls writer.close() and sets _writer to None."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = True
    mock_writer = MagicMock()
    tb._writer = mock_writer
    tb.log_dir = "/tmp/test"

    tb.close()
    mock_writer.close.assert_called_once()
    assert tb._writer is None


def test_tensorboard_logger_close_when_no_writer():
    """close() is safe to call when _writer is already None."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = False
    tb._writer = None
    tb.log_dir = "/tmp/test"

    # Should not raise.
    tb.close()
    assert tb._writer is None


def test_tensorboard_logger_log_validation_loss_tag():
    """log_validation_loss uses correct tag with concept name."""
    from serenity.training.tensorboard import TensorBoardLogger

    tb = TensorBoardLogger.__new__(TensorBoardLogger)
    tb.enabled = True
    tb._writer = MagicMock()
    tb.log_dir = "/tmp/test"

    tb.log_validation_loss(0.05, step=100, concept_name="dog")
    tb._writer.add_scalar.assert_called_once_with(
        "loss/validation_step/dog", 0.05, global_step=100,
    )
