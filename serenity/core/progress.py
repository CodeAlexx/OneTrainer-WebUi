"""Training progress tracking."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainProgress:
    epoch: int = 0
    global_step: int = 0
    ema_loss: Optional[float] = None
