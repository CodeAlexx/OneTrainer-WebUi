"""Trainer orchestration utilities."""

from __future__ import annotations

import gc
import logging
import math
import random
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from serenity.core.config import TrainConfig
from serenity.core.interfaces import BaseModel
from serenity.core.progress import TrainProgress
from serenity.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


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

    def reset_command(self, command: str) -> None:
        """Reset a command flag after handling it."""
        if hasattr(self._flags, command):
            setattr(self._flags, command, False)


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
    """High-level training orchestrator.

    Manages the full training lifecycle: epoch iteration, gradient
    accumulation, optimizer/scheduler stepping, NaN detection, command
    handling, sampling callbacks, and backup scheduling.

    The existing low-level helpers (NaNHandler, CommandHandler,
    BackupManager, GCScheduler) are composed in, not inherited.
    """

    model: Optional[BaseModel]

    def __init__(self, config: TrainConfig, model: Optional[BaseModel] = None) -> None:
        self.config = config
        self.model = model
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: LRScheduler | None = None
        self.progress = TrainProgress()
        self.memory_manager: MemoryManager | None = None

        self.nan_handler = NaNHandler()
        self.command_handler = CommandHandler()
        self.gc_scheduler = GCScheduler()
        self.backup_manager = BackupManager(Path(config.output_dir))

        # Gradient scaler for mixed precision (created lazily in train())
        self._grad_scaler: torch.amp.GradScaler | None = None

        self._seed_everything(config.seed)
        if self.model is not None:
            self._setup_memory_manager()

    # ------------------------------------------------------------------ #
    # Seed management
    # ------------------------------------------------------------------ #

    def _seed_everything(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ #
    # Memory manager
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Mixed precision context
    # ------------------------------------------------------------------ #

    def _autocast_context(self, device_type: str = "cuda") -> AbstractContextManager:
        """Return an autocast context for mixed-precision forward passes.

        Uses the train_dtype hint from config when available.
        Falls back to bfloat16 as a safe default for modern GPUs.
        """
        dtype_str = getattr(self.config, "train_dtype", None)
        if dtype_str is None or device_type == "cpu":
            return nullcontext()

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(str(dtype_str).lower())
        if dtype is None:
            return nullcontext()

        return torch.autocast(device_type=device_type, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Single step (kept for backward compat)
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Placeholder for a single training step."""
        return

    # ------------------------------------------------------------------ #
    # Gradient accumulation check
    # ------------------------------------------------------------------ #

    def _is_accumulation_boundary(self, step: int, accumulation_steps: int) -> bool:
        """Whether this step should trigger optimizer.step()."""
        if accumulation_steps <= 1:
            return True
        return (step + 1) % accumulation_steps == 0

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        dataset: DataLoader | Iterator,
        num_epochs: int,
        *,
        forward_fn: Callable[..., torch.Tensor] | None = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        sample_every_n_steps: int = 0,
        sample_callback: Callable[[TrainProgress], None] | None = None,
        backup_every_n_steps: int = 0,
        backup_callback: Callable[[TrainProgress], None] | None = None,
        on_step_end: Callable[[TrainProgress, float], None] | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> TrainProgress:
        """Run the main training loop.

        This implements the core epoch/step orchestration with:
        - Gradient accumulation over N micro-batches
        - Mixed-precision autocast (optional)
        - NaN detection via NaNHandler
        - Command checking (stop/pause/backup) via CommandHandler
        - Periodic GC via GCScheduler
        - Sampling and backup callbacks at configurable intervals
        - Progress tracking via TrainProgress
        - LR scheduler stepping

        Args:
            dataset: A DataLoader or any iterable of batches.
            num_epochs: Number of training epochs.
            forward_fn: A callable ``(model, batch) -> loss`` that runs the
                forward pass and returns a scalar loss tensor.  If None, the
                trainer expects ``self.model(batch)`` to return the loss.
            gradient_accumulation_steps: Number of micro-batches to accumulate
                before calling ``optimizer.step()``.
            max_grad_norm: If set, clip gradients to this norm before stepping.
            sample_every_n_steps: Fire ``sample_callback`` every N *optimizer*
                steps (0 = disabled).
            sample_callback: Called with current TrainProgress for sampling.
            backup_every_n_steps: Fire ``backup_callback`` every N optimizer
                steps (0 = disabled).
            backup_callback: Called with current TrainProgress for backups.
            on_step_end: Called after every micro-step with (progress, loss).
            use_amp: Enable torch.autocast for mixed precision.
            amp_dtype: Dtype for autocast when use_amp=True.

        Returns:
            The final TrainProgress instance.

        Raises:
            RuntimeError: If model or optimizer is not set, or if NaN loss
                exceeds the abort threshold.
        """
        if self.model is None:
            raise RuntimeError("No model attached. Call attach_model() first.")
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer set. Assign trainer.optimizer before calling train()."
            )

        train_device = torch.device(self.config.train_device)
        progress = self.progress
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        # Gradient scaler for fp16 AMP
        grad_scaler: torch.amp.GradScaler | None = None
        if use_amp and amp_dtype == torch.float16:
            grad_scaler = torch.amp.GradScaler("cuda")

        # Track optimizer steps (as opposed to micro-steps)
        optimizer_steps_this_session = 0

        # Accumulated loss across gradient accumulation micro-batches
        accumulated_loss = 0.0

        logger.info(
            "Starting training: epochs=%d, accum_steps=%d, lr=%.2e",
            num_epochs,
            gradient_accumulation_steps,
            optimizer.param_groups[0]["lr"],
        )

        for epoch_idx in range(progress.epoch, num_epochs):
            progress.epoch = epoch_idx

            # ---- Command check: stop before epoch starts ---- #
            flags = self.command_handler.check_commands()
            if flags.stop:
                logger.info("Stop command received before epoch %d", epoch_idx)
                break

            logger.info("Epoch %d/%d", epoch_idx + 1, num_epochs)

            micro_step_in_epoch = 0

            for batch in dataset:
                # ---- Command check: stop / pause / backup ---- #
                flags = self.command_handler.check_commands()
                if flags.stop:
                    logger.info("Stop command received at step %d", progress.global_step)
                    return progress
                if flags.pause:
                    logger.info("Pause command received, waiting...")
                    while flags.pause:
                        time.sleep(0.5)
                        flags = self.command_handler.check_commands()
                if flags.backup:
                    self.command_handler.reset_command("backup")
                    if backup_callback is not None:
                        backup_callback(progress)

                # ---- Periodic GC ---- #
                self.gc_scheduler.request_gc()

                # ---- Forward pass ---- #
                amp_context = (
                    torch.autocast(device_type=train_device.type, dtype=amp_dtype)
                    if use_amp
                    else nullcontext()
                )

                with amp_context:
                    if forward_fn is not None:
                        loss = forward_fn(self.model, batch)
                    else:
                        loss = self.model(batch)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # ---- Backward pass ---- #
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Track detached loss
                step_loss = loss.detach().item() * gradient_accumulation_steps
                accumulated_loss += loss.detach().item()

                # ---- NaN detection ---- #
                valid, action = self.nan_handler.check_loss(step_loss)
                if not valid:
                    if action == "abort":
                        raise RuntimeError(
                            f"Training loss became NaN for {self.nan_handler.max_consecutive} "
                            f"consecutive steps. Aborting. Stats: {self.nan_handler.get_stats()}"
                        )
                    if action == "emergency_backup" and backup_callback is not None:
                        logger.warning(
                            "Emergency backup triggered after %d consecutive NaN losses",
                            self.nan_handler.consecutive_nan_count,
                        )
                        backup_callback(progress)
                    # For "skip", we continue but still accumulate the step

                micro_step_in_epoch += 1

                # ---- Optimizer step at accumulation boundary ---- #
                if self._is_accumulation_boundary(
                    progress.global_step, gradient_accumulation_steps
                ):
                    # Unscale, clip, step
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )
                        optimizer.step()

                    # LR scheduler step
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    optimizer.zero_grad(set_to_none=True)

                    optimizer_steps_this_session += 1

                    # Update progress with accumulated loss
                    progress.update_loss(accumulated_loss)

                    # Track LR
                    current_lr = optimizer.param_groups[0].get("lr", 0.0)
                    progress.update_lr(current_lr)

                    accumulated_loss = 0.0

                    # ---- Sampling callback ---- #
                    if (
                        sample_every_n_steps > 0
                        and sample_callback is not None
                        and optimizer_steps_this_session % sample_every_n_steps == 0
                    ):
                        sample_callback(progress)

                    # ---- Backup callback ---- #
                    if (
                        backup_every_n_steps > 0
                        and backup_callback is not None
                        and optimizer_steps_this_session % backup_every_n_steps == 0
                    ):
                        backup_callback(progress)

                # ---- Advance step counter ---- #
                progress.next_step(self.config.batch_size)

                # ---- Per-step callback ---- #
                if on_step_end is not None:
                    on_step_end(progress, step_loss)

            # ---- End of epoch ---- #
            progress.next_epoch()
            logger.info(
                "Epoch %d complete. Global step: %d, EMA loss: %s",
                epoch_idx + 1,
                progress.global_step,
                f"{progress.ema_loss:.6f}" if progress.ema_loss is not None else "N/A",
            )

        logger.info("Training complete. Total optimizer steps: %d", optimizer_steps_this_session)
        return progress


__all__ = [
    "NaNHandler",
    "CommandHandler",
    "BackupManager",
    "GCScheduler",
    "Trainer",
]
