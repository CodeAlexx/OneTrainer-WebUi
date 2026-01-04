"""Event definitions and broadcaster for OneTrainer WebSocket updates."""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """WebSocket event types."""

    TRAINING_PROGRESS = "training_progress"
    TRAINING_STATUS = "training_status"
    SAMPLE_GENERATED = "sample_generated"
    LOG = "log"
    SYSTEM_STATS = "system_stats"
    VALIDATION_RESULT = "validation_result"


class LogLevel(str, Enum):
    """Log levels for log events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TrainingStatus(str, Enum):
    """Training status states."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class TrainingProgressEvent:
    """Training progress update event."""

    step: int
    epoch: int
    epoch_step: int
    global_step: int
    max_step: int
    max_epoch: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta_seconds: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.TRAINING_PROGRESS
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


@dataclass
class TrainingStatusEvent:
    """Training status change event."""

    status: TrainingStatus
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.TRAINING_STATUS
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


@dataclass
class SampleGeneratedEvent:
    """Sample image generated event."""

    sample_id: str
    path: str
    sample_type: str  # "default" or "custom"
    step: int
    epoch: int
    prompt: Optional[str] = None
    thumbnail_path: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.SAMPLE_GENERATED
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


@dataclass
class LogEvent:
    """Log message event."""

    level: LogLevel
    message: str
    source: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.LOG
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


@dataclass
class SystemStatsEvent:
    """System statistics event."""

    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    cpu_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.SYSTEM_STATS
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


@dataclass
class ValidationResultEvent:
    """Validation result event."""

    step: int
    epoch: int
    validation_loss: float
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["type"] = EventType.VALIDATION_RESULT
        if self.timestamp is None:
            data["timestamp"] = datetime.now().isoformat()
        return data


class EventBroadcaster:
    """
    Event broadcaster that integrates with OneTrainer's callback system
    and broadcasts updates to WebSocket clients.
    """

    def __init__(self, connection_manager):
        """
        Initialize the event broadcaster.

        Args:
            connection_manager: ConnectionManager instance for broadcasting
        """
        from .connection_manager import ConnectionManager

        self.connection_manager: ConnectionManager = connection_manager
        self._training_state = {
            "status": TrainingStatus.IDLE,
            "current_step": 0,
            "current_epoch": 0,
            "last_loss": None,
            "last_lr": None,
        }
        self._background_tasks: List[asyncio.Task] = []
        self._is_monitoring = False
        self._monitor_interval = 2.0  # seconds

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._is_monitoring:
            logger.warning("Monitoring already started")
            return

        self._is_monitoring = True
        logger.info("Starting event broadcaster monitoring")

        # Start system stats monitoring
        task = asyncio.create_task(self._monitor_system_stats())
        self._background_tasks.append(task)

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        logger.info("Stopping event broadcaster monitoring")

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()

    async def broadcast_training_progress(
        self,
        step: int,
        epoch: int,
        epoch_step: int,
        global_step: int,
        max_step: int,
        max_epoch: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        eta_seconds: Optional[float] = None,
    ):
        """Broadcast a training progress event."""
        event = TrainingProgressEvent(
            step=step,
            epoch=epoch,
            epoch_step=epoch_step,
            global_step=global_step,
            max_step=max_step,
            max_epoch=max_epoch,
            loss=loss,
            learning_rate=learning_rate,
            eta_seconds=eta_seconds,
        )

        # Update internal state
        self._training_state["current_step"] = global_step
        self._training_state["current_epoch"] = epoch
        self._training_state["last_loss"] = loss
        self._training_state["last_lr"] = learning_rate

        await self.connection_manager.broadcast(
            event.to_dict(), EventType.TRAINING_PROGRESS
        )

    async def broadcast_training_status(
        self,
        status: TrainingStatus,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Broadcast a training status change event."""
        event = TrainingStatusEvent(status=status, message=message, error=error)

        # Update internal state
        self._training_state["status"] = status

        await self.connection_manager.broadcast(
            event.to_dict(), EventType.TRAINING_STATUS
        )

    async def broadcast_sample_generated(
        self,
        sample_id: str,
        path: str,
        sample_type: str,
        step: int,
        epoch: int,
        prompt: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
    ):
        """Broadcast a sample generated event."""
        event = SampleGeneratedEvent(
            sample_id=sample_id,
            path=path,
            sample_type=sample_type,
            step=step,
            epoch=epoch,
            prompt=prompt,
            thumbnail_path=thumbnail_path,
        )

        await self.connection_manager.broadcast(
            event.to_dict(), EventType.SAMPLE_GENERATED
        )

    async def broadcast_log(
        self,
        level: LogLevel,
        message: str,
        source: Optional[str] = None,
    ):
        """Broadcast a log event."""
        event = LogEvent(level=level, message=message, source=source)

        await self.connection_manager.broadcast(event.to_dict(), EventType.LOG)

    async def broadcast_system_stats(
        self,
        gpu_memory_used_gb: Optional[float] = None,
        gpu_memory_total_gb: Optional[float] = None,
        gpu_utilization_percent: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        ram_used_gb: Optional[float] = None,
        ram_total_gb: Optional[float] = None,
    ):
        """Broadcast system statistics event."""
        event = SystemStatsEvent(
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_utilization_percent=gpu_utilization_percent,
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
        )

        await self.connection_manager.broadcast(
            event.to_dict(), EventType.SYSTEM_STATS
        )

    async def broadcast_validation_result(
        self, step: int, epoch: int, validation_loss: float
    ):
        """Broadcast validation result event."""
        event = ValidationResultEvent(
            step=step, epoch=epoch, validation_loss=validation_loss
        )

        await self.connection_manager.broadcast(
            event.to_dict(), EventType.VALIDATION_RESULT
        )

    async def _monitor_system_stats(self):
        """Background task to monitor and broadcast system statistics."""
        logger.info("System stats monitoring started")

        while self._is_monitoring:
            try:
                stats = self._collect_system_stats()
                if stats:
                    await self.broadcast_system_stats(**stats)
            except Exception as e:
                logger.error(f"Error collecting system stats: {e}", exc_info=True)

            # Wait before next collection
            await asyncio.sleep(self._monitor_interval)

        logger.info("System stats monitoring stopped")

    def _collect_system_stats(self) -> Optional[Dict[str, float]]:
        """
        Collect current system statistics.

        Returns:
            Dictionary with system stats or None if collection fails
        """
        try:
            import torch
            import psutil

            stats = {}

            # GPU stats (if available)
            if torch.cuda.is_available():
                try:
                    # Get stats for first GPU (can be extended for multi-GPU)
                    gpu_mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None

                    stats["gpu_memory_used_gb"] = round(gpu_mem_used, 2)
                    stats["gpu_memory_total_gb"] = round(gpu_mem_total, 2)
                    if gpu_utilization is not None:
                        stats["gpu_utilization_percent"] = round(gpu_utilization, 1)
                except Exception as e:
                    logger.debug(f"Error collecting GPU stats: {e}")

            # CPU and RAM stats
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024**3)
                ram_total_gb = ram.total / (1024**3)

                stats["cpu_percent"] = round(cpu_percent, 1)
                stats["ram_used_gb"] = round(ram_used_gb, 2)
                stats["ram_total_gb"] = round(ram_total_gb, 2)
            except Exception as e:
                logger.debug(f"Error collecting CPU/RAM stats: {e}")

            return stats if stats else None

        except ImportError:
            # psutil not available
            return None
        except Exception as e:
            logger.error(f"Unexpected error collecting system stats: {e}")
            return None

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return self._training_state.copy()


# Singleton instance (to be initialized by the main application)
_broadcaster_instance: Optional[EventBroadcaster] = None


def get_event_broadcaster() -> Optional[EventBroadcaster]:
    """Get the global event broadcaster instance."""
    return _broadcaster_instance


def set_event_broadcaster(broadcaster: EventBroadcaster):
    """Set the global event broadcaster instance."""
    global _broadcaster_instance
    _broadcaster_instance = broadcaster
