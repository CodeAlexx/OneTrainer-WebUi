"""EriTrainer package.

This package provides a lightweight training stack with clean boundaries between
models, pipeline/data, memory, and training utilities.
"""

from serenity.core.interfaces import BaseModel, ModelType
from serenity.core.config import TrainConfig, TrainerConfig, TrainingMethod, load_config
from serenity.core.trainer import Trainer
from serenity import models
from serenity import sampling
from serenity import adapters
from serenity import data

__all__ = [
    "BaseModel",
    "ModelType",
    "TrainConfig",
    "TrainerConfig",
    "TrainingMethod",
    "load_config",
    "Trainer",
    "models",
    "sampling",
    "adapters",
    "data",
]
