"""WebSocket module for real-time updates in OneTrainer web UI."""

from .connection_manager import ConnectionManager
from .handlers import WebSocketHandler
from .events import (
    TrainingProgressEvent,
    TrainingStatusEvent,
    SampleGeneratedEvent,
    LogEvent,
    SystemStatsEvent,
    ValidationResultEvent,
    EventBroadcaster,
    EventType,
    LogLevel,
    TrainingStatus,
    get_event_broadcaster,
    set_event_broadcaster,
)
from .training_bridge import TrainingWebSocketBridge

__all__ = [
    "ConnectionManager",
    "WebSocketHandler",
    "TrainingProgressEvent",
    "TrainingStatusEvent",
    "SampleGeneratedEvent",
    "LogEvent",
    "SystemStatsEvent",
    "ValidationResultEvent",
    "EventBroadcaster",
    "EventType",
    "LogLevel",
    "TrainingStatus",
    "TrainingWebSocketBridge",
    "get_event_broadcaster",
    "set_event_broadcaster",
]
