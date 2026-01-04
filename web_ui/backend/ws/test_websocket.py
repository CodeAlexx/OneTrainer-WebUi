"""
Tests for WebSocket components.

Run with: pytest web_ui/backend/ws/test_websocket.py
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from .connection_manager import ConnectionManager
from .events import (
    TrainingProgressEvent,
    TrainingStatusEvent,
    SampleGeneratedEvent,
    LogEvent,
    SystemStatsEvent,
    EventBroadcaster,
    EventType,
    LogLevel,
    TrainingStatus,
)
from .handlers import WebSocketHandler


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.messages_sent = []
        self.messages_to_receive = []
        self.is_connected = True

    async def accept(self):
        """Mock accept."""
        pass

    async def send_json(self, data):
        """Mock send_json."""
        if not self.is_connected:
            raise Exception("WebSocket disconnected")
        self.messages_sent.append(data)

    async def receive_json(self):
        """Mock receive_json."""
        if not self.messages_to_receive:
            # Simulate waiting
            await asyncio.sleep(0.1)
            if not self.is_connected:
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect()
            return await self.receive_json()
        return self.messages_to_receive.pop(0)

    def disconnect(self):
        """Simulate disconnection."""
        self.is_connected = False


# =============================================================================
# ConnectionManager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_connection_manager_connect():
    """Test connecting a client."""
    manager = ConnectionManager()
    ws = MockWebSocket()

    client_id = await manager.connect(ws)

    assert client_id is not None
    assert client_id.startswith("client_")
    assert manager.get_connection_count() == 1
    assert len(ws.messages_sent) == 1  # Welcome message
    assert ws.messages_sent[0]["type"] == "connection_established"


@pytest.mark.asyncio
async def test_connection_manager_disconnect():
    """Test disconnecting a client."""
    manager = ConnectionManager()
    ws = MockWebSocket()

    client_id = await manager.connect(ws)
    assert manager.get_connection_count() == 1

    await manager.disconnect(client_id)
    assert manager.get_connection_count() == 0


@pytest.mark.asyncio
async def test_connection_manager_subscribe():
    """Test client subscription."""
    manager = ConnectionManager()
    ws = MockWebSocket()

    client_id = await manager.connect(ws)
    await manager.subscribe(client_id, ["training_progress", "training_status"])

    subscriptions = manager.get_client_subscriptions(client_id)
    assert "training_progress" in subscriptions
    assert "training_status" in subscriptions
    assert len(subscriptions) == 2


@pytest.mark.asyncio
async def test_connection_manager_unsubscribe():
    """Test client unsubscription."""
    manager = ConnectionManager()
    ws = MockWebSocket()

    client_id = await manager.connect(ws)
    await manager.subscribe(client_id, ["training_progress", "training_status"])
    await manager.unsubscribe(client_id, ["training_status"])

    subscriptions = manager.get_client_subscriptions(client_id)
    assert "training_progress" in subscriptions
    assert "training_status" not in subscriptions
    assert len(subscriptions) == 1


@pytest.mark.asyncio
async def test_connection_manager_broadcast():
    """Test broadcasting to all clients."""
    manager = ConnectionManager()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    client1 = await manager.connect(ws1)
    client2 = await manager.connect(ws2)

    message = {"type": "test", "data": "hello"}
    await manager.broadcast(message)

    # Both clients should receive the message (plus welcome message)
    assert len(ws1.messages_sent) == 2
    assert len(ws2.messages_sent) == 2
    assert ws1.messages_sent[1] == message
    assert ws2.messages_sent[1] == message


@pytest.mark.asyncio
async def test_connection_manager_broadcast_with_subscription():
    """Test broadcasting to subscribed clients only."""
    manager = ConnectionManager()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    client1 = await manager.connect(ws1)
    client2 = await manager.connect(ws2)

    # Only client1 subscribes
    await manager.subscribe(client1, ["training_progress"])

    message = {"type": "progress", "step": 100}
    await manager.broadcast(message, event_type="training_progress")

    # Only client1 should receive
    assert len(ws1.messages_sent) == 2  # Welcome + broadcast
    assert len(ws2.messages_sent) == 1  # Welcome only


# =============================================================================
# Event Tests
# =============================================================================


def test_training_progress_event():
    """Test TrainingProgressEvent serialization."""
    event = TrainingProgressEvent(
        step=100,
        epoch=1,
        epoch_step=100,
        global_step=100,
        max_step=1000,
        max_epoch=10,
        loss=0.045,
        learning_rate=1e-5,
        eta_seconds=3600.0,
    )

    data = event.to_dict()
    assert data["type"] == EventType.TRAINING_PROGRESS
    assert data["step"] == 100
    assert data["epoch"] == 1
    assert data["loss"] == 0.045
    assert "timestamp" in data


def test_training_status_event():
    """Test TrainingStatusEvent serialization."""
    event = TrainingStatusEvent(
        status=TrainingStatus.RUNNING, message="Training epoch 1/10"
    )

    data = event.to_dict()
    assert data["type"] == EventType.TRAINING_STATUS
    assert data["status"] == TrainingStatus.RUNNING
    assert data["message"] == "Training epoch 1/10"
    assert "timestamp" in data


def test_sample_generated_event():
    """Test SampleGeneratedEvent serialization."""
    event = SampleGeneratedEvent(
        sample_id="sample_001",
        path="/path/to/sample.png",
        sample_type="default",
        step=100,
        epoch=1,
        prompt="a cat",
    )

    data = event.to_dict()
    assert data["type"] == EventType.SAMPLE_GENERATED
    assert data["sample_id"] == "sample_001"
    assert data["path"] == "/path/to/sample.png"
    assert data["prompt"] == "a cat"


def test_log_event():
    """Test LogEvent serialization."""
    event = LogEvent(level=LogLevel.INFO, message="Test log", source="test")

    data = event.to_dict()
    assert data["type"] == EventType.LOG
    assert data["level"] == LogLevel.INFO
    assert data["message"] == "Test log"
    assert data["source"] == "test"


def test_system_stats_event():
    """Test SystemStatsEvent serialization."""
    event = SystemStatsEvent(
        gpu_memory_used_gb=8.5,
        gpu_memory_total_gb=24.0,
        cpu_percent=45.0,
    )

    data = event.to_dict()
    assert data["type"] == EventType.SYSTEM_STATS
    assert data["gpu_memory_used_gb"] == 8.5
    assert data["cpu_percent"] == 45.0


# =============================================================================
# EventBroadcaster Tests
# =============================================================================


@pytest.mark.asyncio
async def test_event_broadcaster_training_progress():
    """Test broadcasting training progress."""
    manager = ConnectionManager()
    broadcaster = EventBroadcaster(manager)
    ws = MockWebSocket()

    client_id = await manager.connect(ws)
    await manager.subscribe(client_id, [EventType.TRAINING_PROGRESS])

    await broadcaster.broadcast_training_progress(
        step=100,
        epoch=1,
        epoch_step=100,
        global_step=100,
        max_step=1000,
        max_epoch=10,
        loss=0.045,
    )

    # Should have welcome message + progress event
    assert len(ws.messages_sent) == 2
    progress_msg = ws.messages_sent[1]
    assert progress_msg["type"] == EventType.TRAINING_PROGRESS
    assert progress_msg["step"] == 100
    assert progress_msg["loss"] == 0.045


@pytest.mark.asyncio
async def test_event_broadcaster_training_status():
    """Test broadcasting training status."""
    manager = ConnectionManager()
    broadcaster = EventBroadcaster(manager)
    ws = MockWebSocket()

    client_id = await manager.connect(ws)
    await manager.subscribe(client_id, [EventType.TRAINING_STATUS])

    await broadcaster.broadcast_training_status(
        status=TrainingStatus.RUNNING, message="Training started"
    )

    assert len(ws.messages_sent) == 2
    status_msg = ws.messages_sent[1]
    assert status_msg["type"] == EventType.TRAINING_STATUS
    assert status_msg["status"] == TrainingStatus.RUNNING


@pytest.mark.asyncio
async def test_event_broadcaster_monitoring():
    """Test starting and stopping monitoring."""
    manager = ConnectionManager()
    broadcaster = EventBroadcaster(manager)

    # Start monitoring
    await broadcaster.start_monitoring()
    assert broadcaster._is_monitoring is True
    assert len(broadcaster._background_tasks) > 0

    # Stop monitoring
    await broadcaster.stop_monitoring()
    assert broadcaster._is_monitoring is False
    assert len(broadcaster._background_tasks) == 0


# =============================================================================
# WebSocketHandler Tests
# =============================================================================


@pytest.mark.asyncio
async def test_websocket_handler_subscribe():
    """Test subscribe message handling."""
    manager = ConnectionManager()
    handler = WebSocketHandler(manager)
    ws = MockWebSocket()

    # Add subscribe message to queue
    ws.messages_to_receive.append(
        {"type": "subscribe", "events": ["training_progress", "training_status"]}
    )

    # Disconnect after one message
    async def delayed_disconnect():
        await asyncio.sleep(0.2)
        ws.disconnect()

    asyncio.create_task(delayed_disconnect())

    # Handle connection
    await handler.handle_connection(ws)

    # Should have: welcome, subscribed confirmation
    assert len(ws.messages_sent) >= 2
    subscribed_msg = ws.messages_sent[1]
    assert subscribed_msg["type"] == "subscribed"
    assert "training_progress" in subscribed_msg["events"]


@pytest.mark.asyncio
async def test_websocket_handler_ping():
    """Test ping/pong handling."""
    manager = ConnectionManager()
    handler = WebSocketHandler(manager)
    ws = MockWebSocket()

    # Add ping message
    ws.messages_to_receive.append(
        {"type": "ping", "timestamp": datetime.now().isoformat()}
    )

    # Disconnect after one message
    async def delayed_disconnect():
        await asyncio.sleep(0.2)
        ws.disconnect()

    asyncio.create_task(delayed_disconnect())

    # Handle connection
    await handler.handle_connection(ws)

    # Should have: welcome, pong
    assert len(ws.messages_sent) >= 2
    pong_msg = ws.messages_sent[1]
    assert pong_msg["type"] == "pong"
    assert "server_timestamp" in pong_msg


@pytest.mark.asyncio
async def test_websocket_handler_invalid_message():
    """Test handling of invalid messages."""
    manager = ConnectionManager()
    handler = WebSocketHandler(manager)
    ws = MockWebSocket()

    # Add invalid message (no type field)
    ws.messages_to_receive.append({"data": "invalid"})

    # Disconnect after one message
    async def delayed_disconnect():
        await asyncio.sleep(0.2)
        ws.disconnect()

    asyncio.create_task(delayed_disconnect())

    # Handle connection
    await handler.handle_connection(ws)

    # Should have: welcome, error
    assert len(ws.messages_sent) >= 2
    error_msg = ws.messages_sent[1]
    assert error_msg["type"] == "error"
    assert "MISSING_TYPE" in error_msg["error_code"]


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_integration():
    """Test full integration of all components."""
    manager = ConnectionManager()
    broadcaster = EventBroadcaster(manager)
    handler = WebSocketHandler(manager)
    ws = MockWebSocket()

    # Subscribe to events
    ws.messages_to_receive.append(
        {
            "type": "subscribe",
            "events": [
                EventType.TRAINING_PROGRESS,
                EventType.TRAINING_STATUS,
            ],
        }
    )

    # Start connection handling in background
    async def handle_connection():
        await handler.handle_connection(ws)

    connection_task = asyncio.create_task(handle_connection())

    # Wait for connection to establish
    await asyncio.sleep(0.1)

    # Broadcast events
    await broadcaster.broadcast_training_status(
        status=TrainingStatus.STARTING, message="Initializing"
    )
    await broadcaster.broadcast_training_progress(
        step=1, epoch=0, epoch_step=1, global_step=1, max_step=100, max_epoch=10
    )

    # Wait for messages to be sent
    await asyncio.sleep(0.1)

    # Disconnect
    ws.disconnect()
    await asyncio.sleep(0.1)

    # Cancel connection task
    connection_task.cancel()
    try:
        await connection_task
    except asyncio.CancelledError:
        pass

    # Verify messages received
    # Welcome + subscribed + status + progress
    assert len(ws.messages_sent) >= 4

    # Find status and progress messages
    status_msgs = [m for m in ws.messages_sent if m.get("type") == EventType.TRAINING_STATUS]
    progress_msgs = [m for m in ws.messages_sent if m.get("type") == EventType.TRAINING_PROGRESS]

    assert len(status_msgs) >= 1
    assert len(progress_msgs) >= 1
    assert status_msgs[0]["status"] == TrainingStatus.STARTING
    assert progress_msgs[0]["step"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
