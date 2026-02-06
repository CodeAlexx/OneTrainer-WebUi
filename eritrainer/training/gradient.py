"""Gradient checkpointing setup."""

from __future__ import annotations

from typing import Any

from eritrainer.core.enums import GradientCheckpointingMethod


def setup_gradient_checkpointing(model: Any, method: GradientCheckpointingMethod) -> None:
    """Enable or disable gradient checkpointing on a model."""

    if method == GradientCheckpointingMethod.OFF:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        else:
            setattr(model, "gradient_checkpointing", False)
        return

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        setattr(model, "gradient_checkpointing", True)
