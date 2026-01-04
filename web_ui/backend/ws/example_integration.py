"""
Example integration showing how to use the WebSocket system with FastAPI and OneTrainer.

This file demonstrates:
1. Setting up WebSocket endpoint in FastAPI
2. Integrating with OneTrainer's training system
3. Broadcasting events to connected clients
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Import WebSocket components
from .connection_manager import ConnectionManager
from .handlers import WebSocketHandler
from .events import EventBroadcaster, set_event_broadcaster
from .training_bridge import TrainingWebSocketBridge

logger = logging.getLogger(__name__)


# Global instances
connection_manager = ConnectionManager()
event_broadcaster = EventBroadcaster(connection_manager)
ws_handler = WebSocketHandler(connection_manager)
training_bridge = TrainingWebSocketBridge(event_broadcaster)

# Set global broadcaster instance
set_event_broadcaster(event_broadcaster)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events.

    Starts monitoring when app starts, stops when app shuts down.
    """
    # Startup
    logger.info("Starting WebSocket event broadcaster")
    await event_broadcaster.start_monitoring()

    yield

    # Shutdown
    logger.info("Stopping WebSocket event broadcaster")
    await event_broadcaster.stop_monitoring()


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time updates.

    Clients connect to this endpoint to receive training updates.
    """
    await ws_handler.handle_connection(websocket)


@app.get("/api/ws/status")
async def websocket_status():
    """
    Get WebSocket system status.

    Returns information about active connections and training state.
    """
    return {
        "active_connections": connection_manager.get_connection_count(),
        "training_state": event_broadcaster.get_training_state(),
    }


# =============================================================================
# Integration with OneTrainer
# =============================================================================


def get_websocket_callbacks():
    """
    Get TrainCallbacks configured for WebSocket broadcasting.

    Use this when creating a trainer instance:

    Example:
        from web_ui.backend.ws.example_integration import get_websocket_callbacks

        trainer = GenericTrainer(
            config=config,
            callbacks=get_websocket_callbacks(),
            commands=commands,
        )
    """
    return training_bridge.create_train_callbacks()


def add_websocket_to_callbacks(callbacks):
    """
    Add WebSocket broadcasting to existing callbacks.

    Use this to enhance existing callback handlers:

    Example:
        from web_ui.backend.ws.example_integration import add_websocket_to_callbacks

        callbacks = TrainCallbacks(...)
        add_websocket_to_callbacks(callbacks)
    """
    training_bridge.update_existing_callbacks(callbacks)


# =============================================================================
# Example: Manual event broadcasting
# =============================================================================


async def example_manual_events():
    """
    Example showing how to manually broadcast events.

    This can be useful for custom integrations or testing.
    """
    from .events import TrainingStatus, LogLevel

    # Broadcast status change
    await event_broadcaster.broadcast_training_status(
        status=TrainingStatus.STARTING, message="Initializing training..."
    )

    # Broadcast training progress
    await event_broadcaster.broadcast_training_progress(
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

    # Broadcast sample generated
    await event_broadcaster.broadcast_sample_generated(
        sample_id="sample_001",
        path="/path/to/sample.png",
        sample_type="default",
        step=100,
        epoch=1,
        prompt="a cat sitting on a table",
    )

    # Broadcast log message
    await event_broadcaster.broadcast_log(
        level=LogLevel.INFO, message="Training checkpoint saved", source="trainer"
    )

    # Broadcast system stats (collected automatically by default)
    await event_broadcaster.broadcast_system_stats(
        gpu_memory_used_gb=8.5,
        gpu_memory_total_gb=24.0,
        gpu_utilization_percent=95.0,
        cpu_percent=45.0,
        ram_used_gb=16.0,
        ram_total_gb=32.0,
    )


# =============================================================================
# Example: Client-side JavaScript
# =============================================================================

CLIENT_EXAMPLE_JS = """
// Example client-side JavaScript for connecting to WebSocket

const ws = new WebSocket('ws://localhost:8000/ws');

// Connection opened
ws.addEventListener('open', (event) => {
    console.log('Connected to WebSocket');

    // Subscribe to events
    ws.send(JSON.stringify({
        type: 'subscribe',
        events: [
            'training_progress',
            'training_status',
            'sample_generated',
            'log',
            'system_stats'
        ]
    }));
});

// Listen for messages
ws.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);

    switch(data.type) {
        case 'training_progress':
            updateProgressBar(data.global_step, data.max_step * data.max_epoch);
            updateLossChart(data.loss);
            updateETA(data.eta_seconds);
            break;

        case 'training_status':
            updateStatusIndicator(data.status);
            showStatusMessage(data.message);
            break;

        case 'sample_generated':
            addSampleToGallery(data.path, data.prompt);
            break;

        case 'log':
            addLogEntry(data.level, data.message, data.timestamp);
            break;

        case 'system_stats':
            updateGPUChart(data.gpu_memory_used_gb, data.gpu_memory_total_gb);
            updateCPUChart(data.cpu_percent);
            break;

        case 'error':
            console.error('WebSocket error:', data.message);
            break;
    }
});

// Connection closed
ws.addEventListener('close', (event) => {
    console.log('Disconnected from WebSocket');
    // Implement reconnection logic here
});

// Error handler
ws.addEventListener('error', (error) => {
    console.error('WebSocket error:', error);
});

// Helper functions (implement based on your UI)
function updateProgressBar(current, total) {
    const percent = (current / total) * 100;
    document.getElementById('progress-bar').style.width = percent + '%';
    document.getElementById('progress-text').textContent =
        `${current} / ${total} (${percent.toFixed(1)}%)`;
}

function updateLossChart(loss) {
    // Add to your charting library (e.g., Chart.js)
    console.log('Loss:', loss);
}

function updateETA(seconds) {
    if (seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        document.getElementById('eta').textContent =
            `ETA: ${hours}h ${minutes}m`;
    }
}

function updateStatusIndicator(status) {
    const indicator = document.getElementById('status-indicator');
    indicator.className = 'status-' + status;
    indicator.textContent = status.toUpperCase();
}

function showStatusMessage(message) {
    document.getElementById('status-message').textContent = message;
}

function addSampleToGallery(path, prompt) {
    const gallery = document.getElementById('sample-gallery');
    const img = document.createElement('img');
    img.src = path;
    img.alt = prompt;
    img.title = prompt;
    gallery.appendChild(img);
}

function addLogEntry(level, message, timestamp) {
    const logContainer = document.getElementById('logs');
    const entry = document.createElement('div');
    entry.className = 'log-entry log-' + level;
    entry.innerHTML = `
        <span class="timestamp">${new Date(timestamp).toLocaleTimeString()}</span>
        <span class="level">[${level.toUpperCase()}]</span>
        <span class="message">${message}</span>
    `;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateGPUChart(used, total) {
    const percent = (used / total) * 100;
    document.getElementById('gpu-usage').textContent =
        `GPU: ${used.toFixed(1)} / ${total.toFixed(1)} GB (${percent.toFixed(1)}%)`;
}

function updateCPUChart(percent) {
    document.getElementById('cpu-usage').textContent =
        `CPU: ${percent.toFixed(1)}%`;
}
"""


if __name__ == "__main__":
    import uvicorn

    # Run the example server
    uvicorn.run(app, host="0.0.0.0", port=8000)
