# OneTrainer WebSocket Real-Time Updates

This module provides WebSocket-based real-time updates for the OneTrainer web UI, enabling live monitoring of training progress, status, samples, logs, and system statistics.

## Architecture

The WebSocket system consists of several components:

### Core Components

1. **ConnectionManager** (`connection_manager.py`)
   - Manages WebSocket connections
   - Handles client subscriptions to event types
   - Broadcasts messages to connected clients
   - Thread-safe operations with async locks

2. **WebSocketHandler** (`handlers.py`)
   - Routes incoming WebSocket messages
   - Handles subscribe/unsubscribe requests
   - Processes commands from clients
   - Error handling and validation

3. **EventBroadcaster** (`events.py`)
   - Defines event types and data structures
   - Broadcasts events to subscribed clients
   - Background monitoring for system stats
   - Singleton pattern for global access

4. **TrainingWebSocketBridge** (`training_bridge.py`)
   - Bridges OneTrainer's TrainCallbacks with WebSocket events
   - Converts training callbacks to WebSocket broadcasts
   - Handles sync-to-async conversion
   - Throttles updates to prevent overwhelming clients

## Event Types

The system supports the following event types:

### 1. Training Progress (`training_progress`)

Sent during training to report progress metrics.

```json
{
  "type": "training_progress",
  "step": 100,
  "epoch": 1,
  "epoch_step": 100,
  "global_step": 100,
  "max_step": 1000,
  "max_epoch": 10,
  "loss": 0.045,
  "learning_rate": 0.00001,
  "eta_seconds": 3600.0,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 2. Training Status (`training_status`)

Sent when training status changes.

```json
{
  "type": "training_status",
  "status": "running",
  "message": "Training epoch 1/10",
  "error": null,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Status values**: `idle`, `starting`, `running`, `paused`, `stopping`, `stopped`, `error`, `completed`

### 3. Sample Generated (`sample_generated`)

Sent when a sample image is generated.

```json
{
  "type": "sample_generated",
  "sample_id": "sample_001",
  "path": "/path/to/sample.png",
  "sample_type": "default",
  "step": 100,
  "epoch": 1,
  "prompt": "a cat sitting on a table",
  "thumbnail_path": "/path/to/thumbnail.png",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 4. Log (`log`)

Sent for logging messages.

```json
{
  "type": "log",
  "level": "info",
  "message": "Checkpoint saved",
  "source": "trainer",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Log levels**: `debug`, `info`, `warning`, `error`, `critical`

### 5. System Stats (`system_stats`)

Sent periodically with system resource usage.

```json
{
  "type": "system_stats",
  "gpu_memory_used_gb": 8.5,
  "gpu_memory_total_gb": 24.0,
  "gpu_utilization_percent": 95.0,
  "cpu_percent": 45.0,
  "ram_used_gb": 16.0,
  "ram_total_gb": 32.0,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 6. Validation Result (`validation_result`)

Sent after validation runs.

```json
{
  "type": "validation_result",
  "step": 1000,
  "epoch": 1,
  "validation_loss": 0.032,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## Client Message Types

Clients can send the following message types:

### Subscribe

Subscribe to event types:

```json
{
  "type": "subscribe",
  "events": ["training_progress", "training_status", "log"]
}
```

Response:

```json
{
  "type": "subscribed",
  "events": ["training_progress", "training_status", "log"],
  "all_subscriptions": ["training_progress", "training_status", "log"]
}
```

### Unsubscribe

Unsubscribe from event types:

```json
{
  "type": "unsubscribe",
  "events": ["log"]
}
```

Response:

```json
{
  "type": "unsubscribed",
  "events": ["log"],
  "remaining_subscriptions": ["training_progress", "training_status"]
}
```

### Ping

Keep-alive ping:

```json
{
  "type": "ping",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

Response:

```json
{
  "type": "pong",
  "client_timestamp": "2024-01-01T12:00:00.000Z",
  "server_timestamp": "2024-01-01T12:00:00.100Z"
}
```

### Command (Future)

Send commands to training:

```json
{
  "type": "command",
  "command": "pause",
  "args": {}
}
```

## Integration with OneTrainer

### Method 1: Create New Callbacks

```python
from web_ui.backend.ws import (
    ConnectionManager,
    EventBroadcaster,
    TrainingWebSocketBridge,
    set_event_broadcaster,
)

# Initialize components
connection_manager = ConnectionManager()
event_broadcaster = EventBroadcaster(connection_manager)
training_bridge = TrainingWebSocketBridge(event_broadcaster)

# Set global broadcaster
set_event_broadcaster(event_broadcaster)

# Start monitoring
await event_broadcaster.start_monitoring()

# Create callbacks for trainer
callbacks = training_bridge.create_train_callbacks()

# Use with trainer
trainer = GenericTrainer(
    config=config,
    callbacks=callbacks,
    commands=commands,
)
```

### Method 2: Update Existing Callbacks

```python
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from web_ui.backend.ws import TrainingWebSocketBridge, EventBroadcaster

# Existing callbacks
callbacks = TrainCallbacks(...)

# Add WebSocket broadcasting
event_broadcaster = EventBroadcaster(connection_manager)
training_bridge = TrainingWebSocketBridge(event_broadcaster)
training_bridge.update_existing_callbacks(callbacks)
```

### FastAPI Integration

```python
from fastapi import FastAPI, WebSocket
from web_ui.backend.ws import ConnectionManager, WebSocketHandler

app = FastAPI()
connection_manager = ConnectionManager()
ws_handler = WebSocketHandler(connection_manager)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_handler.handle_connection(websocket)
```

## Client-Side Usage

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.addEventListener('open', () => {
    // Subscribe to events
    ws.send(JSON.stringify({
        type: 'subscribe',
        events: ['training_progress', 'training_status', 'sample_generated']
    }));
});

ws.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'training_progress':
            updateProgressBar(data.global_step, data.max_step * data.max_epoch);
            break;
        case 'training_status':
            updateStatus(data.status, data.message);
            break;
        case 'sample_generated':
            displaySample(data.path, data.prompt);
            break;
    }
});
```

### Python Client Example

```python
import asyncio
import websockets
import json

async def connect():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps({
            "type": "subscribe",
            "events": ["training_progress", "training_status"]
        }))

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")

            if data['type'] == 'training_progress':
                print(f"Step: {data['global_step']}, Loss: {data['loss']}")

asyncio.run(connect())
```

## Configuration

### Throttling

The bridge throttles progress updates to prevent overwhelming clients:

```python
training_bridge._progress_throttle_interval = 0.1  # Max 10 updates/sec
```

### System Stats Monitoring

Configure monitoring interval:

```python
event_broadcaster._monitor_interval = 2.0  # Check every 2 seconds
```

## Error Handling

All components include comprehensive error handling:

- WebSocket disconnections are handled gracefully
- Failed message sends don't crash the system
- Invalid messages return error responses
- Exceptions are logged with full context

## Performance Considerations

1. **Message Throttling**: Progress updates are throttled to prevent network congestion
2. **Async Operations**: All I/O operations are async for non-blocking execution
3. **Subscription Filtering**: Only subscribed clients receive specific event types
4. **Resource Monitoring**: System stats collection is lightweight and periodic
5. **Connection Cleanup**: Disconnected clients are automatically cleaned up

## Testing

### Manual Testing

1. Start the example server:
```bash
python -m web_ui.backend.ws.example_integration
```

2. Connect with a WebSocket client (e.g., browser console):
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

3. Check status endpoint:
```bash
curl http://localhost:8000/api/ws/status
```

### Unit Testing

```python
import pytest
from web_ui.backend.ws import ConnectionManager, EventBroadcaster

@pytest.mark.asyncio
async def test_connection():
    manager = ConnectionManager()
    # ... test implementation
```

## Troubleshooting

### Common Issues

1. **Connections not receiving events**
   - Verify client is subscribed to event types
   - Check if events are being broadcast
   - Confirm WebSocket connection is established

2. **High memory usage**
   - Adjust monitoring intervals
   - Reduce update frequency
   - Check for connection leaks

3. **Events not firing**
   - Ensure bridge is connected to callbacks
   - Verify event broadcaster is started
   - Check for exceptions in logs

## Future Enhancements

- [ ] Command handling (pause/resume/stop training)
- [ ] Authentication and authorization
- [ ] Multiple room support for concurrent trainings
- [ ] Event replay for reconnecting clients
- [ ] Compression for large payloads
- [ ] Rate limiting per client
- [ ] Metrics and monitoring dashboard

## Dependencies

- `fastapi`: Web framework
- `websockets`: WebSocket support
- `torch`: For GPU stats
- `psutil`: For CPU/RAM stats
- `asyncio`: Async operations

## License

Part of the OneTrainer project.
