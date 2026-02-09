"""Training resume from checkpoint.

Provides ``resume_from_checkpoint()`` which restores optimizer state,
EMA state, and training progress from a Serenity checkpoint directory,
integrating with the ``Trainer`` and ``CheckpointManager`` APIs.

This module completes the checkpoint lifecycle:
- ``saver.py`` -- persists model/optimizer/EMA/progress to disk
- ``loader.py`` -- loads raw checkpoint data into ``CheckpointData``
- ``manager.py`` -- rolling checkpoint management and discovery
- ``resume.py`` (this) -- wires loaded state back into a live Trainer
"""

from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from serenity.checkpoint.loader import CheckpointData, load_checkpoint
from serenity.checkpoint.manager import CheckpointManager
from serenity.core.progress import TrainProgress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress serialization helpers
# ---------------------------------------------------------------------------

def progress_to_dict(progress: TrainProgress) -> dict[str, Any]:
    """Serialize a ``TrainProgress`` to a JSON-safe dict."""
    return asdict(progress)


def progress_from_dict(data: dict[str, Any]) -> TrainProgress:
    """Restore a ``TrainProgress`` from a dict.

    Unknown keys are silently ignored so forward-compatible fields
    added later don't break old checkpoints.
    """
    valid_fields = {f.name for f in fields(TrainProgress)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return TrainProgress(**filtered)


# ---------------------------------------------------------------------------
# Checkpoint save helper (for Trainer integration)
# ---------------------------------------------------------------------------

def save_training_checkpoint(
    trainer: Any,
    checkpoint_path: str | Path,
    *,
    model_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a full training checkpoint from a live Trainer.

    Extracts optimizer state, EMA state, and progress from the trainer
    and delegates to ``ModelSaver.save_checkpoint``.

    Parameters
    ----------
    trainer:
        A ``Trainer`` instance with ``.optimizer``, ``.progress``, and
        optionally ``._ema`` attributes.
    checkpoint_path:
        Destination directory.
    model_state:
        Model/adapter state dict.  If ``None`` and the trainer has a model,
        attempts to extract it.
    extra:
        Additional metadata to include in ``meta.json``.

    Returns
    -------
    Path
        The checkpoint directory path.
    """
    from serenity.checkpoint.saver import ModelSaver

    saver = ModelSaver()

    # Optimizer state
    optimizer_state = None
    if trainer.optimizer is not None:
        optimizer_state = trainer.optimizer.state_dict()

    # EMA state
    ema_state = None
    ema = getattr(trainer, "_ema", None) or getattr(trainer, "ema", None)
    if ema is not None and hasattr(ema, "state_dict"):
        ema_state = ema.state_dict()

    # Progress
    progress_dict = None
    if trainer.progress is not None:
        progress_dict = progress_to_dict(trainer.progress)

    # Model state
    if model_state is None:
        model = getattr(trainer, "model", None)
        if model is not None and hasattr(model, "state_dict"):
            model_state = model.state_dict()

    return saver.save_checkpoint(
        checkpoint_path,
        model_state=model_state,
        optimizer_state=optimizer_state,
        ema_state=ema_state,
        progress=progress_dict,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def resume_from_checkpoint(
    trainer: Any,
    checkpoint_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    load_optimizer: bool = True,
    load_ema: bool = True,
    load_progress: bool = True,
    load_model: bool = False,
    strict_model_load: bool = False,
    map_location: str = "cpu",
) -> CheckpointData | None:
    """Resume training from a checkpoint.

    This function loads checkpoint data and restores it into a live
    ``Trainer`` instance.  It handles:

    1. **Checkpoint discovery** -- if *checkpoint_path* is ``None``,
       automatically finds the latest checkpoint in *output_dir* (or
       ``trainer.config.output_dir``).
    2. **Optimizer state** -- loads optimizer state dict if the trainer
       has an optimizer attached.
    3. **EMA state** -- loads EMA shadow weights if the trainer has an
       EMA module.
    4. **Training progress** -- restores epoch, step, loss history so
       training continues from where it left off.
    5. **Model weights** -- optionally loads model/adapter weights.

    Parameters
    ----------
    trainer:
        A ``Trainer`` instance (from ``serenity.core.trainer``).
    checkpoint_path:
        Explicit path to a checkpoint directory.  If ``None``, the latest
        checkpoint is auto-discovered from *output_dir*.
    output_dir:
        Base output directory to search for checkpoints.  Falls back to
        ``trainer.config.output_dir``.
    load_optimizer:
        Whether to restore optimizer state.
    load_ema:
        Whether to restore EMA state.
    load_progress:
        Whether to restore training progress (epoch, step, loss).
    load_model:
        Whether to load model/adapter weights from the checkpoint.
    strict_model_load:
        Strict mode for ``model.load_state_dict()``.
    map_location:
        Device for torch.load.

    Returns
    -------
    CheckpointData | None
        The loaded checkpoint data, or ``None`` if no checkpoint was found.
    """
    # ---- Discover checkpoint ----
    if checkpoint_path is None:
        search_dir = output_dir or getattr(trainer.config, "output_dir", None)
        if search_dir is None:
            logger.warning("No checkpoint_path or output_dir specified, cannot resume")
            return None

        checkpoint_path = CheckpointManager.get_latest_checkpoint(search_dir)
        if checkpoint_path is None:
            logger.info("No existing checkpoint found in %s, starting from scratch", search_dir)
            return None

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_dir():
        logger.warning("Checkpoint path does not exist: %s", checkpoint_path)
        return None

    logger.info("Resuming from checkpoint: %s", checkpoint_path)

    # ---- Load checkpoint data ----
    data = load_checkpoint(checkpoint_path)

    # ---- Restore optimizer state ----
    if load_optimizer and data.optimizer_state is not None:
        if trainer.optimizer is not None:
            try:
                trainer.optimizer.load_state_dict(data.optimizer_state)
                logger.info("Restored optimizer state from checkpoint")
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load optimizer state: %s", exc)
        else:
            logger.debug("Trainer has no optimizer, skipping optimizer state restore")

    # ---- Restore EMA state ----
    if load_ema and data.ema_state is not None:
        ema = getattr(trainer, "_ema", None) or getattr(trainer, "ema", None)
        if ema is not None and hasattr(ema, "load_state_dict"):
            try:
                ema.load_state_dict(data.ema_state)
                logger.info("Restored EMA state from checkpoint")
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load EMA state: %s", exc)
        else:
            logger.debug("Trainer has no EMA module, skipping EMA state restore")

    # ---- Restore training progress ----
    if load_progress and data.progress is not None:
        try:
            restored = progress_from_dict(data.progress)
            trainer.progress = restored
            logger.info(
                "Restored training progress: epoch=%d, global_step=%d, ema_loss=%s",
                restored.epoch,
                restored.global_step,
                f"{restored.ema_loss:.6f}" if restored.ema_loss is not None else "N/A",
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            logger.warning("Failed to restore training progress: %s", exc)

    # ---- Restore model weights (optional) ----
    if load_model and data.model_state is not None:
        model = getattr(trainer, "model", None)
        if model is not None and hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(data.model_state, strict=strict_model_load)
                logger.info("Restored model weights from checkpoint")
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load model weights: %s", exc)
        else:
            logger.debug("Trainer model does not support load_state_dict")

    return data


__all__ = [
    "progress_to_dict",
    "progress_from_dict",
    "save_training_checkpoint",
    "resume_from_checkpoint",
]
