"""
Bridge module to connect OneTrainer's TrainCallbacks with WebSocket event broadcasting.

This module provides callback handlers that can be registered with OneTrainer's
training system to broadcast real-time updates to WebSocket clients.
"""

import asyncio
import logging
import os
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingWebSocketBridge:
    """
    Bridge between OneTrainer's TrainCallbacks and WebSocket EventBroadcaster.

    This class provides callback methods compatible with OneTrainer's callback
    system that broadcast events to connected WebSocket clients.
    """

    def __init__(self, event_broadcaster):
        """
        Initialize the training bridge.

        Args:
            event_broadcaster: EventBroadcaster instance for sending events
        """
        from .events import EventBroadcaster, TrainingStatus, LogLevel

        self.broadcaster: EventBroadcaster = event_broadcaster
        self.TrainingStatus = TrainingStatus
        self.LogLevel = LogLevel
        self._last_progress_time = 0
        self._progress_throttle_interval = 0.1  # Throttle to max 10 updates/sec
        self._training_start_time = None
        self._last_step = 0
        self._event_loop = None

    def _get_or_create_event_loop(self):
        """Get or create an event loop for async operations."""
        if self._event_loop is None or self._event_loop.is_closed():
            try:
                self._event_loop = asyncio.get_event_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def _run_async(self, coro):
        """
        Run an async coroutine from a sync context.

        Args:
            coro: Coroutine to run
        """
        try:
            loop = self._get_or_create_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                asyncio.create_task(coro)
            else:
                # Run the coroutine
                loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"Error running async task: {e}", exc_info=True)

    def on_update_train_progress(self, train_progress, max_step: int, max_epoch: int):
        """
        Callback for training progress updates.

        Args:
            train_progress: TrainProgress object with current progress
            max_step: Maximum steps per epoch
            max_epoch: Maximum number of epochs
        """
        import time

        # Throttle updates to avoid overwhelming clients
        current_time = time.time()
        if current_time - self._last_progress_time < self._progress_throttle_interval:
            return

        self._last_progress_time = current_time

        # Calculate ETA
        eta_seconds = None
        if self._training_start_time is not None and train_progress.global_step > 0:
            elapsed = current_time - self._training_start_time
            total_steps = max_epoch * max_step
            steps_remaining = total_steps - train_progress.global_step
            if steps_remaining > 0:
                steps_per_second = train_progress.global_step / elapsed
                if steps_per_second > 0:
                    eta_seconds = steps_remaining / steps_per_second

        # Broadcast progress event
        self._run_async(
            self.broadcaster.broadcast_training_progress(
                step=train_progress.epoch_step,
                epoch=train_progress.epoch,
                epoch_step=train_progress.epoch_step,
                global_step=train_progress.global_step,
                max_step=max_step,
                max_epoch=max_epoch,
                eta_seconds=eta_seconds,
            )
        )

        self._last_step = train_progress.global_step

    def on_update_status(self, status: str):
        """
        Callback for training status updates.

        Args:
            status: Status message string
        """
        import time

        # Map status strings to TrainingStatus enum
        status_lower = status.lower()
        training_status = self.TrainingStatus.RUNNING  # Default

        # Reset start time if starting
        if "starting" in status_lower or "loading" in status_lower:
            training_status = self.TrainingStatus.STARTING
            if self._training_start_time is None:
                self._training_start_time = time.time()
        elif "training" in status_lower:
            training_status = self.TrainingStatus.RUNNING
            if self._training_start_time is None:
                self._training_start_time = time.time()
        elif "paused" in status_lower:
            training_status = self.TrainingStatus.PAUSED
        elif "stopping" in status_lower or "stopped" in status_lower:
            training_status = self.TrainingStatus.STOPPED
            self._training_start_time = None
        elif "error" in status_lower or "failed" in status_lower:
            training_status = self.TrainingStatus.ERROR
            self._training_start_time = None
        elif "completed" in status_lower or "finished" in status_lower:
            training_status = self.TrainingStatus.COMPLETED
            self._training_start_time = None
        elif "saving" in status_lower or "backup" in status_lower:
            # Keep current status but log the message
            pass

        # Broadcast status event
        self._run_async(
            self.broadcaster.broadcast_training_status(
                status=training_status, message=status
            )
        )

        # Also broadcast as a log event for detailed tracking
        self._run_async(
            self.broadcaster.broadcast_log(
                level=self.LogLevel.INFO, message=status, source="trainer"
            )
        )

    def on_sample_default(self, sampler_output):
        """
        Callback for default sample generation.

        Args:
            sampler_output: ModelSamplerOutput object with generated sample
        """
        self._on_sample(sampler_output, "default")

    def on_sample_custom(self, sampler_output):
        """
        Callback for custom sample generation.

        Args:
            sampler_output: ModelSamplerOutput object with generated sample
        """
        self._on_sample(sampler_output, "custom")

    def _on_sample(self, sampler_output, sample_type: str):
        """
        Internal method to handle sample generation.

        Args:
            sampler_output: ModelSamplerOutput object
            sample_type: "default" or "custom"
        """
        try:
            # Extract sample information
            # Note: Actual structure depends on ModelSamplerOutput implementation
            sample_path = getattr(sampler_output, "path", None)
            if sample_path is None:
                logger.warning("Sample output has no path attribute")
                return

            # Generate sample ID from filename or timestamp
            sample_id = os.path.basename(sample_path) if sample_path else f"sample_{datetime.now().timestamp()}"

            # Get current training state for step/epoch info
            training_state = self.broadcaster.get_training_state()

            # Broadcast sample event
            self._run_async(
                self.broadcaster.broadcast_sample_generated(
                    sample_id=sample_id,
                    path=sample_path,
                    sample_type=sample_type,
                    step=training_state.get("current_step", 0),
                    epoch=training_state.get("current_epoch", 0),
                    prompt=getattr(sampler_output, "prompt", None),
                )
            )

            # Log the sample generation
            self._run_async(
                self.broadcaster.broadcast_log(
                    level=self.LogLevel.INFO,
                    message=f"Generated {sample_type} sample: {sample_id}",
                    source="sampler",
                )
            )

        except Exception as e:
            logger.error(f"Error handling sample output: {e}", exc_info=True)

    def on_update_sample_default_progress(self, step: int, max_step: int):
        """
        Callback for default sample generation progress.

        Args:
            step: Current step
            max_step: Maximum steps
        """
        # Log sampling progress
        self._run_async(
            self.broadcaster.broadcast_log(
                level=self.LogLevel.DEBUG,
                message=f"Sampling progress: {step}/{max_step}",
                source="sampler",
            )
        )

    def on_update_sample_custom_progress(self, step: int, max_step: int):
        """
        Callback for custom sample generation progress.

        Args:
            step: Current step
            max_step: Maximum steps
        """
        # Log sampling progress
        self._run_async(
            self.broadcaster.broadcast_log(
                level=self.LogLevel.DEBUG,
                message=f"Custom sampling progress: {step}/{max_step}",
                source="sampler",
            )
        )

    def create_train_callbacks(self):
        """
        Create a TrainCallbacks object with this bridge's methods.

        Returns:
            TrainCallbacks object configured with bridge callbacks
        """
        from modules.util.callbacks.TrainCallbacks import TrainCallbacks

        return TrainCallbacks(
            on_update_train_progress=self.on_update_train_progress,
            on_update_status=self.on_update_status,
            on_sample_default=self.on_sample_default,
            on_update_sample_default_progress=self.on_update_sample_default_progress,
            on_sample_custom=self.on_sample_custom,
            on_update_sample_custom_progress=self.on_update_sample_custom_progress,
        )

    def update_existing_callbacks(self, callbacks):
        """
        Update existing TrainCallbacks object to include this bridge's methods.

        This allows adding WebSocket broadcasting to existing callback handlers.

        Args:
            callbacks: Existing TrainCallbacks object to update
        """
        callbacks.set_on_update_train_progress(self.on_update_train_progress)
        callbacks.set_on_update_status(self.on_update_status)
        callbacks.set_on_sample_default(self.on_sample_default)
        callbacks.set_on_update_sample_default_progress(
            self.on_update_sample_default_progress
        )
        callbacks.set_on_sample_custom(self.on_sample_custom)
        callbacks.set_on_update_sample_custom_progress(
            self.on_update_sample_custom_progress
        )

    def reset(self):
        """Reset the bridge state (e.g., when starting a new training session)."""
        self._training_start_time = None
        self._last_step = 0
        self._last_progress_time = 0
