"""
Job Queue Service for managing multiple training jobs.

Provides functionality for:
- Adding jobs to queue
- Managing job order
- Auto-starting next job when current completes
- Job history tracking
"""

import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


class JobStatus(str, Enum):
    """Job status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedJob:
    """Represents a job in the queue."""
    id: str
    name: str
    config_path: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "config_path": self.config_path,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "progress": self.progress,
        }


class QueueService:
    """
    Singleton service for managing the training job queue.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._queue: List[QueuedJob] = []
        self._history: List[QueuedJob] = []
        self._current_job: Optional[QueuedJob] = None
        self._queue_lock = threading.Lock()
        self._auto_start = True
        self._trainer_service = None

    def set_trainer_service(self, trainer_service):
        """Set reference to trainer service for auto-starting jobs."""
        self._trainer_service = trainer_service

    def add_job(self, name: str, config_path: str) -> QueuedJob:
        """
        Add a new job to the queue.

        Args:
            name: Job name/description
            config_path: Path to training config file

        Returns:
            The created QueuedJob
        """
        job = QueuedJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            config_path=config_path,
        )

        with self._queue_lock:
            self._queue.append(job)

        return job

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the queue.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was removed, False if not found
        """
        with self._queue_lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    if job.status == JobStatus.RUNNING:
                        return False  # Can't remove running job
                    self._queue.pop(i)
                    return True
        return False

    def move_job(self, job_id: str, new_position: int) -> bool:
        """
        Move a job to a new position in the queue.

        Args:
            job_id: ID of job to move
            new_position: New position (0-indexed)

        Returns:
            True if job was moved, False if not found
        """
        with self._queue_lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    if job.status == JobStatus.RUNNING:
                        return False  # Can't move running job
                    self._queue.pop(i)
                    new_position = max(0, min(new_position, len(self._queue)))
                    self._queue.insert(new_position, job)
                    return True
        return False

    def get_queue(self) -> List[QueuedJob]:
        """Get all jobs in the queue."""
        with self._queue_lock:
            return list(self._queue)

    def get_history(self, limit: int = 50) -> List[QueuedJob]:
        """Get completed/failed job history."""
        with self._queue_lock:
            return list(self._history[-limit:])

    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        """Get a specific job by ID."""
        with self._queue_lock:
            for job in self._queue:
                if job.id == job_id:
                    return job
            for job in self._history:
                if job.id == job_id:
                    return job
        return None

    def get_current_job(self) -> Optional[QueuedJob]:
        """Get the currently running job."""
        return self._current_job

    def start_next(self) -> Optional[QueuedJob]:
        """
        Start the next pending job in the queue.

        Returns:
            The started job, or None if no pending jobs
        """
        if self._current_job and self._current_job.status == JobStatus.RUNNING:
            return None  # Already running a job

        with self._queue_lock:
            for job in self._queue:
                if job.status == JobStatus.PENDING:
                    return self._start_job(job)
        return None

    def _start_job(self, job: QueuedJob) -> Optional[QueuedJob]:
        """Internal method to start a specific job."""
        if not self._trainer_service:
            return None

        try:
            # Load config and start training
            config_path = Path(job.config_path)
            if not config_path.exists():
                job.status = JobStatus.FAILED
                job.error = f"Config file not found: {job.config_path}"
                self._move_to_history(job)
                return None

            # Import here to avoid circular imports
            from modules.util.config.TrainConfig import TrainConfig

            train_config = TrainConfig.default_values()
            with open(config_path, "r") as f:
                train_config.from_dict(json.load(f))

            if not self._trainer_service.initialize_trainer(train_config):
                job.status = JobStatus.FAILED
                job.error = "Failed to initialize trainer"
                self._move_to_history(job)
                return None

            if not self._trainer_service.start_training():
                job.status = JobStatus.FAILED
                job.error = "Failed to start training"
                self._move_to_history(job)
                return None

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._current_job = job

            return job

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self._move_to_history(job)
            return None

    def on_training_complete(self, success: bool, error: Optional[str] = None):
        """
        Called when training completes.

        Args:
            success: Whether training completed successfully
            error: Error message if failed
        """
        if self._current_job:
            with self._queue_lock:
                if success:
                    self._current_job.status = JobStatus.COMPLETED
                else:
                    self._current_job.status = JobStatus.FAILED
                    self._current_job.error = error

                self._current_job.completed_at = datetime.now()
                self._move_to_history(self._current_job)
                self._current_job = None

            # Auto-start next job if enabled
            if self._auto_start:
                self.start_next()

    def _move_to_history(self, job: QueuedJob):
        """Move a job from queue to history."""
        if job in self._queue:
            self._queue.remove(job)
        if job not in self._history:
            self._history.append(job)
            # Keep history limited
            if len(self._history) > 100:
                self._history = self._history[-100:]

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: ID of job to cancel

        Returns:
            True if cancelled, False if not found or already running
        """
        with self._queue_lock:
            for job in self._queue:
                if job.id == job_id:
                    if job.status == JobStatus.RUNNING:
                        # Try to stop training
                        if self._trainer_service:
                            self._trainer_service.stop_training()
                        job.status = JobStatus.CANCELLED
                        job.completed_at = datetime.now()
                        self._move_to_history(job)
                        self._current_job = None
                        return True
                    elif job.status == JobStatus.PENDING:
                        job.status = JobStatus.CANCELLED
                        self._move_to_history(job)
                        return True
        return False

    def clear_history(self):
        """Clear the job history."""
        with self._queue_lock:
            self._history.clear()

    def set_auto_start(self, enabled: bool):
        """Enable or disable auto-starting next job."""
        self._auto_start = enabled


# Singleton accessor
_queue_service: Optional[QueueService] = None


def get_queue_service() -> QueueService:
    """Get the queue service singleton instance."""
    global _queue_service
    if _queue_service is None:
        _queue_service = QueueService()
    return _queue_service
