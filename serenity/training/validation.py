"""Validation loop for between-epoch or interval-based validation.

Computes per-concept validation losses and logs them to TensorBoard.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from serenity.core.progress import TrainProgress

logger = logging.getLogger(__name__)

__all__ = ["ValidationConfig", "ValidationRunner"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ValidationConfig:
    """Validation-specific settings.

    Attributes:
        enabled: Whether validation is active.
        val_every_n_epochs: Run validation every N epochs (0 = disabled).
        val_every_n_steps: Run validation every N optimizer steps (0 = disabled).
        val_dataset_path: Optional path to a separate validation dataset.
        log_per_concept: If True, log loss per concept/subdirectory.
    """

    enabled: bool = False
    val_every_n_epochs: int = 1
    val_every_n_steps: int = 0
    val_dataset_path: str = ""
    log_per_concept: bool = True


# --------------------------------------------------------------------------- #
# Validation runner
# --------------------------------------------------------------------------- #


class ValidationRunner:
    """Runs a validation loop over a separate dataset.

    Iterates the validation DataLoader with ``torch.no_grad()``, accumulates
    per-concept losses, computes averages, and reports them to a logger
    callback.

    Usage::

        runner = ValidationRunner(config=val_cfg)
        runner.run(
            dataloader=val_loader,
            predict_fn=model_setup.predict,
            loss_fn=model_setup.calculate_loss,
            progress=train_progress,
            log_fn=tensorboard.log_validation_loss,
        )
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()
        self._last_val_epoch: int = -1
        self._last_val_step: int = -1

    # ------------------------------------------------------------------ #
    # Scheduling helpers
    # ------------------------------------------------------------------ #

    def needs_validation(self, progress: TrainProgress) -> bool:
        """Return True if validation should run at the current point."""
        if not self.config.enabled:
            return False

        # Step-based trigger
        if (
            self.config.val_every_n_steps > 0
            and progress.global_step > 0
            and progress.global_step != self._last_val_step
            and progress.global_step % self.config.val_every_n_steps == 0
        ):
            return True

        # Epoch-based trigger
        if (
            self.config.val_every_n_epochs > 0
            and progress.epoch > 0
            and progress.epoch != self._last_val_epoch
            and progress.epoch % self.config.val_every_n_epochs == 0
        ):
            return True

        return False

    # ------------------------------------------------------------------ #
    # Core validation loop
    # ------------------------------------------------------------------ #

    def run(
        self,
        dataloader: DataLoader | Iterable,
        predict_fn: Callable[..., dict[str, Any]],
        loss_fn: Callable[..., torch.Tensor],
        progress: TrainProgress,
        *,
        log_fn: Callable[[float, int, str], None] | None = None,
        model: Any = None,
        config: Any = None,
    ) -> dict[str, float]:
        """Execute one full validation pass.

        Args:
            dataloader: Validation data loader (batch_size=1 recommended).
            predict_fn: ``(model, batch, config, progress) -> model_output_data``.
                Called with ``deterministic=True`` under ``torch.no_grad()``.
            loss_fn: ``(model, batch, model_output_data, config) -> loss_tensor``.
            progress: Current training progress (used for step number).
            log_fn: Optional callback ``(loss, step, concept_name)`` for logging.
            model: The model instance (passed through to predict/loss fns).
            config: The train config (passed through to predict/loss fns).

        Returns:
            Dict mapping concept name to average validation loss.
        """
        logger.info("Running validation at step %d", progress.global_step)

        accumulated_loss: dict[str, float] = defaultdict(float)
        concept_counts: dict[str, int] = defaultdict(int)

        # Label collision handling
        seed_to_label: dict[int, str] = {}
        label_to_seed: dict[str, int] = {}

        with torch.no_grad():
            for batch in dataloader:
                # Predict
                model_output = predict_fn(
                    model, batch, config, progress, deterministic=True
                )

                # Loss
                loss_tensor = loss_fn(model, batch, model_output, config)
                loss_value = loss_tensor.item()

                # Resolve concept label
                concept_name = self._resolve_concept_label(
                    batch, seed_to_label, label_to_seed,
                )

                accumulated_loss[concept_name] += loss_value
                concept_counts[concept_name] += 1

        # Compute averages
        results: dict[str, float] = {}
        for concept_name, total_loss in accumulated_loss.items():
            avg = total_loss / concept_counts[concept_name]
            results[concept_name] = avg

            if log_fn is not None and self.config.log_per_concept:
                log_fn(avg, progress.global_step, concept_name)

        # Log total average when multiple concepts
        if len(results) > 1:
            total_sum = sum(accumulated_loss.values())
            total_count = sum(concept_counts.values())
            total_avg = total_sum / total_count
            results["total_average"] = total_avg

            if log_fn is not None:
                log_fn(total_avg, progress.global_step, "total_average")

        # Update scheduling state
        self._last_val_epoch = progress.epoch
        self._last_val_step = progress.global_step

        logger.info(
            "Validation complete: %s",
            ", ".join(f"{k}={v:.6f}" for k, v in results.items()),
        )
        return results

    # ------------------------------------------------------------------ #
    # Simplified validation (loss-only, no predict/loss split)
    # ------------------------------------------------------------------ #

    def run_simple(
        self,
        dataloader: DataLoader | Iterable,
        forward_fn: Callable[[Any], torch.Tensor],
        progress: TrainProgress,
        *,
        log_fn: Callable[[float, int, str], None] | None = None,
    ) -> float:
        """Run validation with a simple ``forward_fn(batch) -> loss`` interface.

        Returns the average validation loss.
        """
        logger.info("Running simple validation at step %d", progress.global_step)

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                loss = forward_fn(batch)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)

        if log_fn is not None:
            log_fn(avg_loss, progress.global_step, "total_average")

        self._last_val_epoch = progress.epoch
        self._last_val_step = progress.global_step

        logger.info("Simple validation complete: avg_loss=%.6f", avg_loss)
        return avg_loss

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_concept_label(
        batch: dict[str, Any],
        seed_to_label: dict[int, str],
        label_to_seed: dict[str, int],
    ) -> str:
        """Resolve a unique label for each concept, handling collisions.

        If two different concept seeds map to the same label, a numeric
        suffix is added to disambiguate.
        """
        concept_name = ""
        concept_path = ""
        concept_seed = 0

        if "concept_name" in batch:
            val = batch["concept_name"]
            concept_name = val[0] if isinstance(val, (list, tuple)) else str(val)
        if "concept_path" in batch:
            val = batch["concept_path"]
            concept_path = val[0] if isinstance(val, (list, tuple)) else str(val)
        if "concept_seed" in batch:
            val = batch["concept_seed"]
            concept_seed = val.item() if hasattr(val, "item") else int(val)

        label = concept_name if concept_name else os.path.basename(concept_path)
        if not label:
            label = "default"

        # Collision handling
        if label in label_to_seed and label_to_seed[label] != concept_seed:
            suffix = 1
            new_label = f"{label}({suffix})"
            while new_label in label_to_seed and label_to_seed[new_label] != concept_seed:
                suffix += 1
                new_label = f"{label}({suffix})"
            label = new_label

        if concept_seed not in seed_to_label:
            seed_to_label[concept_seed] = label
            label_to_seed[label] = concept_seed

        return seed_to_label.get(concept_seed, label)
