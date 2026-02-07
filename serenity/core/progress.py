"""Training progress tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainProgress:
    """Track epoch, step, and loss metrics throughout training.

    Tracks loss history and LR for richer diagnostics.
    """

    epoch: int = 0
    epoch_step: int = 0
    epoch_sample: int = 0
    epoch_length: int = 0
    global_step: int = 0

    # Loss tracking
    ema_loss: float | None = None
    ema_loss_steps: int = 0
    loss_history: list[float] = field(default_factory=list)

    # LR tracking
    current_lr: float | None = None

    def next_step(self, batch_size: int = 1) -> None:
        """Advance one training step within the current epoch."""
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(self) -> None:
        """Advance to the next epoch, resetting per-epoch counters."""
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    def update_loss(self, loss_value: float, ema_decay_max: float = 0.99) -> None:
        """Update EMA loss and loss history with a new step loss."""
        self.loss_history.append(loss_value)
        self.ema_loss_steps += 1
        decay = min(ema_decay_max, 1.0 - (1.0 / self.ema_loss_steps))

        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = (self.ema_loss * decay) + (loss_value * (1.0 - decay))

    def update_lr(self, lr: float) -> None:
        """Record the current learning rate."""
        self.current_lr = lr

    def filename_string(self) -> str:
        """Generate a filename-safe progress tag."""
        return f"{self.global_step}-{self.epoch}-{self.epoch_step}"


__all__ = ["TrainProgress"]
