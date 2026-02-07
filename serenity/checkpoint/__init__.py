"""Checkpoint save/load/manage system for Serenity.

Provides unified model saving in safetensors and diffusers formats,
training checkpoint persistence (optimizer + EMA + progress), and
rolling checkpoint management with automatic cleanup.

Quick start::

    from serenity.checkpoint import ModelSaver, CheckpointManager

    saver = ModelSaver(model_type="sdxl", training_method="lora")
    saver.save_lora(adapter.state_dict(), "output/my_lora.safetensors")

    mgr = CheckpointManager(output_dir="output/", keep=3)
    mgr.save_at_step(500, model_state=adapter.state_dict(),
                     optimizer_state=optimizer.state_dict())
"""

from __future__ import annotations

from serenity.checkpoint.saver import ModelSaver, create_saver
from serenity.checkpoint.loader import (
    CheckpointData,
    CheckpointFormat,
    detect_format,
    load_checkpoint,
    load_embedding,
    load_lora,
)
from serenity.checkpoint.manager import CheckpointManager, create_manager
from serenity.checkpoint.resume import (
    progress_from_dict,
    progress_to_dict,
    resume_from_checkpoint,
    save_training_checkpoint,
)
from serenity.checkpoint.conversion import (
    ModelFormat,
    detect_model_format,
    convert_diffusers_to_safetensors,
    convert_safetensors_to_diffusers,
    convert_lora_format,
    convert_checkpoint,
)

# Convenience aliases used by the assignment spec
save_model = create_saver


__all__ = [
    # Saver
    "ModelSaver",
    "create_saver",
    "save_model",
    # Loader
    "CheckpointData",
    "CheckpointFormat",
    "detect_format",
    "load_checkpoint",
    "load_embedding",
    "load_lora",
    # Manager
    "CheckpointManager",
    "create_manager",
    # Resume
    "progress_to_dict",
    "progress_from_dict",
    "save_training_checkpoint",
    "resume_from_checkpoint",
    # Conversion
    "ModelFormat",
    "detect_model_format",
    "convert_diffusers_to_safetensors",
    "convert_safetensors_to_diffusers",
    "convert_lora_format",
    "convert_checkpoint",
]
