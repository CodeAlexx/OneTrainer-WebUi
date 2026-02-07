"""Checkpoint lifecycle management -- rolling saves, backups, and cleanup.

Mirrors the checkpoint-management logic scattered across OneTrainer's
``GenericTrainer``, ``BackupManager``, and ``InternalModelSaverMixin``,
unified into a single ``CheckpointManager`` class.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from serenity.checkpoint.saver import ModelSaver
from serenity.core.interfaces import ModelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManager:
    """Manage training checkpoint lifecycle inside an output directory.

    Responsibilities:
    * ``save_at_step`` -- create ``checkpoint-{step}/`` directories
    * ``save_backup``  -- create ``backup-{label}/`` snapshots
    * ``get_latest_checkpoint`` -- find newest checkpoint for resume
    * ``cleanup_old_checkpoints`` -- prune to *keep* most recent
    """

    output_dir: Path
    model_type: ModelType | str | None = None
    training_method: str | None = None
    default_dtype: torch.dtype | None = None
    keep: int = 3
    _saver: ModelSaver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self._saver = ModelSaver(
            model_type=self.model_type,
            training_method=self.training_method,
            default_dtype=self.default_dtype,
        )

    # -- Save at training step -------------------------------------------------

    def save_at_step(
        self,
        step: int,
        *,
        model_state: dict[str, Any] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        ema_state: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a checkpoint at training *step*.

        Creates ``<output_dir>/checkpoint-<step>/`` and delegates to
        ``ModelSaver.save_checkpoint``.  After saving, old checkpoints
        beyond ``self.keep`` are automatically pruned.
        """
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        self._saver.save_checkpoint(
            ckpt_dir,
            model_state=model_state,
            optimizer_state=optimizer_state,
            ema_state=ema_state,
            progress=progress,
            extra=extra,
        )
        logger.info("Checkpoint saved at step %d -> %s", step, ckpt_dir)

        self.cleanup_old_checkpoints()
        return ckpt_dir

    # -- Backup snapshots ------------------------------------------------------

    def save_backup(
        self,
        label: str | None = None,
        *,
        model_state: dict[str, Any] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        ema_state: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Create a backup checkpoint with a timestamp-based name.

        Layout: ``<output_dir>/backup-<label>/``

        If *label* is ``None`` the current UTC timestamp is used.
        """
        if label is None:
            label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        backup_dir = self.output_dir / f"backup-{label}"
        self._saver.save_checkpoint(
            backup_dir,
            model_state=model_state,
            optimizer_state=optimizer_state,
            ema_state=ema_state,
            progress=progress,
            extra=extra,
        )
        logger.info("Backup saved -> %s", backup_dir)
        return backup_dir

    # -- Latest checkpoint discovery -------------------------------------------

    @staticmethod
    def get_latest_checkpoint(output_dir: str | Path) -> Path | None:
        """Find the most recent ``checkpoint-<step>/`` directory.

        Searches *output_dir* for directories matching the naming convention
        and returns the one with the highest step number.  Returns ``None``
        if no checkpoints exist.
        """
        root = Path(output_dir).expanduser()
        if not root.is_dir():
            return None

        candidates: list[tuple[int, Path]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("checkpoint-"):
                continue
            suffix = name[len("checkpoint-"):]
            try:
                step = int(suffix)
            except ValueError:
                continue
            candidates.append((step, child))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    # -- Cleanup ---------------------------------------------------------------

    def cleanup_old_checkpoints(
        self,
        output_dir: str | Path | None = None,
        keep: int | None = None,
    ) -> list[Path]:
        """Remove old ``checkpoint-*`` directories, keeping the *keep* newest.

        Returns a list of directories that were removed.
        """
        root = Path(output_dir).expanduser() if output_dir else self.output_dir
        num_keep = keep if keep is not None else self.keep

        candidates: list[tuple[int, Path]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("checkpoint-"):
                continue
            suffix = name[len("checkpoint-"):]
            try:
                step = int(suffix)
            except ValueError:
                continue
            candidates.append((step, child))

        if len(candidates) <= num_keep:
            return []

        candidates.sort(key=lambda t: t[0])
        to_remove = candidates[: len(candidates) - num_keep]

        removed: list[Path] = []
        for _step, ckpt_path in to_remove:
            try:
                shutil.rmtree(ckpt_path)
                removed.append(ckpt_path)
                logger.info("Removed old checkpoint: %s", ckpt_path)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", ckpt_path, exc)

        return removed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_manager(
    output_dir: str | Path,
    model_type: ModelType | str | None = None,
    training_method: str | None = None,
    dtype: torch.dtype | None = None,
    keep: int = 3,
) -> CheckpointManager:
    """Create a ``CheckpointManager`` with common defaults."""
    return CheckpointManager(
        output_dir=Path(output_dir),
        model_type=model_type,
        training_method=training_method,
        default_dtype=dtype,
        keep=keep,
    )


__all__ = [
    "CheckpointManager",
    "create_manager",
]
