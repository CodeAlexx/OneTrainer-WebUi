"""Trainer orchestration utilities."""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import shutil
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Rolling backup manager for checkpoints and configs.

    Creates timestamped backup directories containing model weights,
    optimizer state, progress info, and the training config.  Supports
    rolling backups with auto-pruning of oldest.

    Integrates with ``serenity.checkpoint.CheckpointManager`` when
    available for the actual save mechanics.
    """

    def __init__(
        self,
        output_dir: Path,
        rolling_count: int = 3,
        backup_interval_steps: int = 0,
        backup_before_save: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.rolling_count = max(1, rolling_count)
        self.backup_interval_steps = backup_interval_steps
        self.backup_before_save = backup_before_save
        self._backup_dir = self.output_dir / "backup"
        self._last_backup_step: int = -1

    @property
    def backup_dir(self) -> Path:
        """Root directory containing all backup subdirectories."""
        return self._backup_dir

    def ensure_dir(self) -> None:
        """Create the backup directory tree if it does not exist."""
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def needs_backup(self, current_step: int) -> bool:
        """Check whether a backup should be created at this step."""
        if self.backup_interval_steps <= 0:
            return False
        if self._last_backup_step < 0:
            # No backup yet -- backup on first eligible interval
            return current_step > 0 and current_step % self.backup_interval_steps == 0
        return (current_step - self._last_backup_step) >= self.backup_interval_steps

    def create_backup(
        self,
        progress: TrainProgress,
        *,
        model_state: dict[str, Any] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        ema_state: dict[str, Any] | None = None,
        config_data: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> Path | None:
        """Create a backup snapshot.

        Directory layout::

            <backup_dir>/<timestamp>-backup-<step>/
                model.safetensors    (if model_state provided)
                optimizer/
                    optimizer.pt     (if optimizer_state provided)
                ema/
                    ema.pt           (if ema_state provided)
                meta.json            (progress + config)

        After saving, old backups beyond ``rolling_count`` are pruned.
        Returns the backup path, or None if the save failed.
        """
        self.ensure_dir()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        step_str = f"e{progress.epoch}-s{progress.global_step}"
        if label:
            dirname = f"{timestamp}-{label}-{step_str}"
        else:
            dirname = f"{timestamp}-backup-{step_str}"
        backup_path = self._backup_dir / dirname

        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            # Model weights
            if model_state is not None:
                self._save_model_state(backup_path, model_state)

            # Optimizer state
            if optimizer_state is not None:
                opt_dir = backup_path / "optimizer"
                opt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(optimizer_state, str(opt_dir / "optimizer.pt"))

            # EMA state
            if ema_state is not None:
                ema_dir = backup_path / "ema"
                ema_dir.mkdir(parents=True, exist_ok=True)
                torch.save(ema_state, str(ema_dir / "ema.pt"))

            # Meta / progress
            meta: dict[str, Any] = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "train_progress": {
                    "epoch": progress.epoch,
                    "global_step": progress.global_step,
                    "ema_loss": progress.ema_loss,
                },
            }
            if config_data is not None:
                meta["config"] = config_data

            with open(backup_path / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)

            self._last_backup_step = progress.global_step
            logger.info("Backup created at step %d -> %s", progress.global_step, backup_path)

        except OSError:
            logger.error("Failed to create backup: %s", traceback.format_exc())
            # Clean up partial backup
            try:
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
            except OSError:
                logger.error("Failed to clean up partial backup: %s", traceback.format_exc())
            return None

        # Prune old backups
        self.prune_backups()
        return backup_path

    def _save_model_state(
        self,
        backup_path: Path,
        model_state: dict[str, Any],
    ) -> None:
        """Save model weights, using safetensors if available."""
        try:
            from safetensors.torch import save_file
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in model_state.items()},
                str(backup_path / "model.safetensors"),
            )
        except ImportError:
            torch.save(model_state, str(backup_path / "model.pt"))

    def prune_backups(self, keep: int | None = None) -> list[Path]:
        """Remove old backups, keeping only the *keep* most recent.

        Backups are sorted by directory name (which includes a timestamp
        prefix) so newest are kept.  Returns list of removed paths.
        """
        num_keep = keep if keep is not None else self.rolling_count
        if not self._backup_dir.is_dir():
            return []

        backup_dirs = sorted(
            [d for d in self._backup_dir.iterdir() if d.is_dir()],
            reverse=True,  # newest first
        )

        if len(backup_dirs) <= num_keep:
            return []

        to_remove = backup_dirs[num_keep:]
        removed: list[Path] = []
        for dirpath in to_remove:
            try:
                shutil.rmtree(dirpath)
                removed.append(dirpath)
                logger.info("Removed old backup: %s", dirpath)
            except OSError:
                logger.warning("Could not delete old backup: %s", dirpath)

        return removed

    def list_backups(self) -> list[Path]:
        """Return all backup directories sorted newest-first."""
        if not self._backup_dir.is_dir():
            return []
        return sorted(
            [d for d in self._backup_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

    def get_latest_backup(self) -> Path | None:
        """Return the most recent backup directory, or None."""
        backups = self.list_backups()
        return backups[0] if backups else None


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
        self.backup_manager = BackupManager(
            output_dir=Path(config.output_dir),
            rolling_count=config.rolling_backup_count,
            backup_before_save=config.backup_before_save,
        )

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

                # Track detached loss
                step_loss = loss.detach().item() * gradient_accumulation_steps
                accumulated_loss += loss.detach().item()

                # ---- NaN detection (BEFORE backward to protect optimizer state) ---- #
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
                    # Skip backward pass — zero grads to keep state clean
                    optimizer.zero_grad(set_to_none=True)
                    logger.warning("Skipping backward pass for non-finite loss: %.4f", step_loss)
                else:
                    # ---- Backward pass (only for finite loss) ---- #
                    if grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()

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
