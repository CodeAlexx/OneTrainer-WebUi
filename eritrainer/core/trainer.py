"""Trainer orchestration utilities."""

from __future__ import annotations

import gc
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from eritrainer.core.config import TrainConfig
from eritrainer.core.interfaces import BaseModel
from eritrainer.core.progress import TrainProgress
from eritrainer.memory.manager import MemoryManager


class NaNHandler:
    """Detect and respond to NaN/Inf losses."""

    def __init__(
        self,
        max_consecutive: int = 10,
        total_nan_threshold: int = 100,
        emergency_backup_threshold: int = 5,
    ) -> None:
        self.max_consecutive = max_consecutive
        self.total_nan_threshold = total_nan_threshold
        self.emergency_backup_threshold = emergency_backup_threshold
        self.consecutive_nan_count = 0
        self.total_nan_count = 0
        self.last_valid_loss: Optional[float] = None

    def check_loss(self, loss_value: float) -> tuple[bool, str]:
        if loss_value is None or math.isnan(loss_value) or math.isinf(loss_value):
            self.consecutive_nan_count += 1
            self.total_nan_count += 1

            if self.consecutive_nan_count >= self.max_consecutive:
                return False, "abort"
            if self.consecutive_nan_count >= self.emergency_backup_threshold:
                return False, "emergency_backup"
            return False, "skip"

        self.last_valid_loss = float(loss_value)
        self.consecutive_nan_count = 0
        return True, "continue"

    def get_stats(self) -> dict:
        return {
            "total_nan_count": self.total_nan_count,
            "consecutive_nan_count": self.consecutive_nan_count,
            "last_valid_loss": self.last_valid_loss,
        }


@dataclass
class CommandFlags:
    stop: bool = False
    pause: bool = False
    backup: bool = False


class CommandHandler:
    """Simple in-process command synchronization."""

    def __init__(self) -> None:
        self._flags = CommandFlags()

    def check_commands(self) -> CommandFlags:
        return self._flags

    def set_command(self, command: str) -> None:
        if not hasattr(self._flags, command):
            raise ValueError(f"Unknown command: {command}")
        setattr(self._flags, command, True)


class BackupManager:
    """Rolling backup manager for checkpoints and configs."""

    def __init__(self, output_dir: Path, rolling_count: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.rolling_count = rolling_count

    def ensure_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


class GCScheduler:
    """Throttle garbage collection to avoid excessive pauses."""

    def __init__(self, min_interval_seconds: float = 60.0) -> None:
        self._min_interval = float(min_interval_seconds)
        self._last_gc = 0.0

    def request_gc(self) -> bool:
        now = time.monotonic()
        if now - self._last_gc < self._min_interval:
            return False
        self._last_gc = now
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True


class Trainer:
    """High-level training orchestrator."""

    model: Optional[BaseModel]

    def __init__(self, config: TrainConfig, model: Optional[BaseModel] = None) -> None:
        self.config = config
        self.model = model
        self.optimizer = None
        self.progress = TrainProgress()
        self.memory_manager: MemoryManager | None = None

        self.nan_handler = NaNHandler()
        self.command_handler = CommandHandler()
        self.gc_scheduler = GCScheduler()
        self.backup_manager = BackupManager(Path(config.output_dir))

        self._seed_everything(config.seed)
        if self.model is not None:
            self._setup_memory_manager()

    def _seed_everything(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_memory_manager(self) -> None:
        if self.model is None:
            self.memory_manager = None
            return
        self.memory_manager = MemoryManager(self.config, self.model)
        self.memory_manager.setup_optimizations()

    def attach_model(self, model: BaseModel) -> None:
        self.model = model
        self._setup_memory_manager()

    def forward_context(self) -> AbstractContextManager:
        if self.memory_manager is None:
            return nullcontext()
        return self.memory_manager.forward_context()

    def step(self) -> None:
        """Placeholder for a single training step."""
        return


__all__ = [
    "NaNHandler",
    "CommandHandler",
    "BackupManager",
    "GCScheduler",
    "Trainer",
]
