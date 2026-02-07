"""TensorBoard logging wrapper for Serenity training.

Logs loss/train_step, smooth_loss/train_step, lr/*, ema_decay, and
sample images.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["TensorBoardLogger"]


class TensorBoardLogger:
    """Lightweight TensorBoard logger wrapping ``SummaryWriter``.

    Provides typed helpers for the metrics Serenity's training loop emits:
    scalar values (loss, LR, EMA decay), images (sample outputs), and
    optional histograms (weight distributions).

    Usage::

        tb = TensorBoardLogger(log_dir="workspace/tensorboard")
        tb.log_scalar("loss/train_step", 0.123, step=100)
        tb.log_image("sample0", image_tensor, step=100)
        tb.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        flush_secs: int = 120,
        filename_suffix: str = "",
        enabled: bool = True,
    ) -> None:
        self.log_dir = str(log_dir)
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix
        self.enabled = enabled
        self._writer: Any | None = None

        if self.enabled:
            self._ensure_writer()

    # ------------------------------------------------------------------ #
    # Writer lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_writer(self) -> Any:
        """Lazily create the ``SummaryWriter``."""
        if self._writer is not None:
            return self._writer
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning(
                "tensorboard not installed; TensorBoardLogger disabled"
            )
            self.enabled = False
            return None

        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=self.log_dir,
            flush_secs=self.flush_secs,
            filename_suffix=self.filename_suffix,
        )
        logger.info("TensorBoard logging to %s", self.log_dir)
        return self._writer

    @property
    def writer(self) -> Any | None:
        """Access the underlying ``SummaryWriter`` (or None)."""
        return self._writer

    # ------------------------------------------------------------------ #
    # Scalar logging
    # ------------------------------------------------------------------ #

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Log a scalar value (loss, LR, EMA decay, etc.)."""
        if not self.enabled or self._writer is None:
            return
        self._writer.add_scalar(tag, value, global_step=step)

    def log_loss(self, loss: float, step: int) -> None:
        """Shortcut for ``loss/train_step``."""
        self.log_scalar("loss/train_step", loss, step)

    def log_smooth_loss(self, smooth_loss: float, step: int) -> None:
        """Shortcut for ``smooth_loss/train_step``."""
        self.log_scalar("smooth_loss/train_step", smooth_loss, step)

    def log_lr(self, lr: float, step: int, *, name: str = "default") -> None:
        """Log learning rate for a parameter group."""
        self.log_scalar(f"lr/{name}", lr, step)

    def log_ema_decay(self, decay: float, step: int) -> None:
        """Log the current EMA decay factor."""
        self.log_scalar("ema_decay", decay, step)

    def log_validation_loss(
        self,
        loss: float,
        step: int,
        *,
        concept_name: str = "total_average",
    ) -> None:
        """Log per-concept or total validation loss."""
        self.log_scalar(f"loss/validation_step/{concept_name}", loss, step)

    # ------------------------------------------------------------------ #
    # Image logging
    # ------------------------------------------------------------------ #

    def log_image(
        self,
        tag: str,
        image: torch.Tensor | Any,
        step: int,
        *,
        dataformats: str = "CHW",
    ) -> None:
        """Log a sample image tensor.

        Accepts a ``(C, H, W)`` tensor by default, or a PIL image that
        will be converted via ``torchvision.transforms.functional.pil_to_tensor``.
        """
        if not self.enabled or self._writer is None:
            return

        if not isinstance(image, torch.Tensor):
            try:
                from torchvision.transforms.functional import pil_to_tensor
                image = pil_to_tensor(image)
            except (ImportError, TypeError):
                logger.debug("Could not convert image for tensorboard: %s", tag)
                return

        self._writer.add_image(tag, image, global_step=step, dataformats=dataformats)

    # ------------------------------------------------------------------ #
    # Histogram logging (optional)
    # ------------------------------------------------------------------ #

    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int,
        *,
        bins: str = "tensorflow",
    ) -> None:
        """Log a histogram of tensor values (e.g. weight distributions)."""
        if not self.enabled or self._writer is None:
            return
        self._writer.add_histogram(tag, values, global_step=step, bins=bins)

    # ------------------------------------------------------------------ #
    # Flush / close
    # ------------------------------------------------------------------ #

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer and flush remaining data."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
