"""Training callback system for lifecycle events.

Provides a ``TrainCallbacks`` registry that dispatches training lifecycle
events to registered handlers using a multi-listener pattern for
extensibility.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainCallbacks:
    """Registry for training lifecycle event handlers.

    Multiple callbacks can be registered for each event.  Handlers are
    called in registration order and exceptions are suppressed to prevent
    a misbehaving callback from crashing training.

    Events:
        on_train_start: Training run begins.
        on_train_end: Training run completes (or is interrupted).
        on_epoch_start(epoch): New epoch begins.
        on_epoch_end(epoch): Epoch completes.
        on_step(global_step, loss): Training step completes.
        on_sample(global_step, images): Sampling/validation occurs.
        on_checkpoint(global_step, path): Checkpoint is saved.
        on_log(global_step, metrics): Metrics are logged.
    """

    _on_train_start: list[Callable[[], None]] = field(default_factory=list)
    _on_train_end: list[Callable[[], None]] = field(default_factory=list)
    _on_epoch_start: list[Callable[[int], None]] = field(default_factory=list)
    _on_epoch_end: list[Callable[[int], None]] = field(default_factory=list)
    _on_step: list[Callable[[int, float], None]] = field(default_factory=list)
    _on_sample: list[Callable[[int, Any], None]] = field(default_factory=list)
    _on_checkpoint: list[Callable[[int, str], None]] = field(default_factory=list)
    _on_log: list[Callable[[int, dict[str, Any]], None]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_on_train_start(self, fn: Callable[[], None]) -> None:
        """Register a handler for training start."""
        self._on_train_start.append(fn)

    def register_on_train_end(self, fn: Callable[[], None]) -> None:
        """Register a handler for training end."""
        self._on_train_end.append(fn)

    def register_on_epoch_start(self, fn: Callable[[int], None]) -> None:
        """Register a handler for epoch start."""
        self._on_epoch_start.append(fn)

    def register_on_epoch_end(self, fn: Callable[[int], None]) -> None:
        """Register a handler for epoch end."""
        self._on_epoch_end.append(fn)

    def register_on_step(self, fn: Callable[[int, float], None]) -> None:
        """Register a handler for training step completion."""
        self._on_step.append(fn)

    def register_on_sample(self, fn: Callable[[int, Any], None]) -> None:
        """Register a handler for sample/validation events."""
        self._on_sample.append(fn)

    def register_on_checkpoint(self, fn: Callable[[int, str], None]) -> None:
        """Register a handler for checkpoint save events."""
        self._on_checkpoint.append(fn)

    def register_on_log(self, fn: Callable[[int, dict[str, Any]], None]) -> None:
        """Register a handler for metric logging events."""
        self._on_log.append(fn)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Dispatch training start event."""
        for fn in self._on_train_start:
            with contextlib.suppress(Exception):
                fn()

    def on_train_end(self) -> None:
        """Dispatch training end event."""
        for fn in self._on_train_end:
            with contextlib.suppress(Exception):
                fn()

    def on_epoch_start(self, epoch: int) -> None:
        """Dispatch epoch start event."""
        for fn in self._on_epoch_start:
            with contextlib.suppress(Exception):
                fn(epoch)

    def on_epoch_end(self, epoch: int) -> None:
        """Dispatch epoch end event."""
        for fn in self._on_epoch_end:
            with contextlib.suppress(Exception):
                fn(epoch)

    def on_step(self, global_step: int, loss: float) -> None:
        """Dispatch training step event."""
        for fn in self._on_step:
            with contextlib.suppress(Exception):
                fn(global_step, loss)

    def on_sample(self, global_step: int, images: Any) -> None:
        """Dispatch sample/validation event."""
        for fn in self._on_sample:
            with contextlib.suppress(Exception):
                fn(global_step, images)

    def on_checkpoint(self, global_step: int, path: str) -> None:
        """Dispatch checkpoint save event."""
        for fn in self._on_checkpoint:
            with contextlib.suppress(Exception):
                fn(global_step, path)

    def on_log(self, global_step: int, metrics: dict[str, Any]) -> None:
        """Dispatch metric logging event."""
        for fn in self._on_log:
            with contextlib.suppress(Exception):
                fn(global_step, metrics)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered callbacks."""
        self._on_train_start.clear()
        self._on_train_end.clear()
        self._on_epoch_start.clear()
        self._on_epoch_end.clear()
        self._on_step.clear()
        self._on_sample.clear()
        self._on_checkpoint.clear()
        self._on_log.clear()

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all events."""
        return (
            len(self._on_train_start)
            + len(self._on_train_end)
            + len(self._on_epoch_start)
            + len(self._on_epoch_end)
            + len(self._on_step)
            + len(self._on_sample)
            + len(self._on_checkpoint)
            + len(self._on_log)
        )


__all__ = ["TrainCallbacks"]
