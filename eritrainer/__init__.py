"""EriTrainer package.

This package provides a lightweight training stack with clean boundaries between
models, pipeline/data, memory, and training utilities.
"""

from eritrainer.core.interfaces import BaseModel, ModelType
from eritrainer.core.config import TrainConfig, TrainerConfig, TrainingMethod, load_config
from eritrainer.core.trainer import Trainer
from eritrainer import models
from eritrainer import sampling
from eritrainer import adapters
from eritrainer import data

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
