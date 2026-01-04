# OneTrainer Web UI

A modern web interface for OneTrainer built with FastAPI and React.

## Architecture

### Backend Structure

```
web_ui/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/                    # REST API endpoints
│   │   ├── training.py         # Training control endpoints
│   │   ├── config.py           # Configuration management
│   │   ├── samples.py          # Sample generation
│   │   └── system.py           # System information
│   ├── services/               # Business logic
│   │   └── trainer_service.py  # Singleton trainer manager
│   └── ws/                     # WebSocket handlers
│       ├── connection_manager.py
│       ├── events.py
│       └── handlers.py
├── frontend/                   # React frontend (to be implemented)
│   └── src/
└── run.py                      # Application entry point
```

## Getting Started

### Prerequisites

- Python 3.10+
- OneTrainer dependencies (see main requirements.txt)
- Additional web UI dependencies:
  - fastapi
  - uvicorn[standard]
  - websockets
  - psutil

### Installation

```bash
# Install web UI dependencies
pip install fastapi uvicorn[standard] websockets psutil pydantic

# Or add to requirements.txt:
# fastapi>=0.104.0
# uvicorn[standard]>=0.24.0
# websockets>=12.0
# psutil>=5.9.0
# pydantic>=2.0.0
```

### Running the Server

From the OneTrainer root directory:

```bash
# Method 1: Using run.py (recommended)
python web_ui/run.py

# Method 2: Direct uvicorn
cd web_ui
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Core Endpoints

#### Training Control
- `GET /api/training/status` - Get current training status
- `POST /api/training/start` - Start training session
- `POST /api/training/stop` - Stop training
- `POST /api/training/backup` - Create backup
- `POST /api/training/save` - Save model
- `GET /api/training/progress` - Get training progress

#### Configuration
- `GET /api/config/current` - Get active config
- `GET /api/config/list` - List available configs
- `GET /api/config/load/{name}` - Load config file
- `POST /api/config/validate` - Validate config
- `GET /api/config/schema` - Get config schema
- `GET /api/config/defaults` - Get default config

#### Samples
- `POST /api/samples/default` - Generate default samples
- `POST /api/samples/custom` - Generate custom sample
- `GET /api/samples/list` - List all samples
- `GET /api/samples/latest` - Get recent samples
- `GET /api/samples/{id}` - Get specific sample

#### System
- `GET /api/system/info` - System information
- `GET /api/system/resources` - Resource usage
- `GET /api/system/processes` - Process information

### WebSocket

Connect to `ws://localhost:8000/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'connected':
      // Initial state received
      break;
    case 'training_state':
      // Training state update
      break;
    case 'sample_default':
      // Default sample generated
      break;
    case 'sample_custom':
      // Custom sample generated
      break;
  }
};

// Keep-alive ping
setInterval(() => ws.send('ping'), 30000);
```

## Trainer Service

The `TrainerService` is a singleton that manages the OneTrainer lifecycle:

### Features

- **Thread-safe state management** - Safe concurrent access
- **WebSocket broadcasting** - Real-time updates to all clients
- **Training control** - Start, stop, pause operations
- **Command routing** - Sample, backup, save commands
- **Progress tracking** - Detailed training metrics

### State Structure

```python
{
    "is_training": bool,
    "status": str,  # idle, initialized, starting, running, stopping, completed, error
    "progress": {
        "epoch": int,
        "epoch_step": int,
        "epoch_sample": int,
        "global_step": int
    },
    "max_step": int,
    "max_epoch": int,
    "error": Optional[str]
}
```

### Usage Example

```python
from web_ui.backend.services.trainer_service import get_trainer_service

# Get singleton instance
trainer = get_trainer_service()

# Initialize with config
from modules.util.config.TrainConfig import TrainConfig
config = TrainConfig(...)
trainer.initialize_trainer(config)

# Start training (runs in background thread)
trainer.start_training()

# Get current state
state = trainer.get_state()

# Send commands
trainer.stop_training()
trainer.sample_default()
trainer.backup()
trainer.save()
```

## Implementation Notes

### Completed

✅ FastAPI application setup with CORS
✅ Lifespan context management
✅ WebSocket endpoint for real-time updates
✅ TrainerService singleton with:
  - GenericTrainer integration
  - TrainCommands wrapper
  - TrainCallbacks integration
  - Thread-safe state management
  - WebSocket broadcasting
✅ REST API structure:
  - Training control endpoints
  - Configuration management endpoints
  - Sample generation endpoints
  - System information endpoints
✅ Entry point script (run.py)
✅ Proper Python package structure

### TODO

The following features have placeholder implementations and need completion:

1. **Config Loading** (`api/config.py`)
   - Load TrainConfig from JSON files
   - Validate config against schema
   - Generate config schema from TrainConfig class
   - Create default config instances

2. **Sample Management** (`api/samples.py`)
   - Convert API requests to SampleConfig objects
   - File system integration for sample discovery
   - Sample metadata tracking

3. **Training Start** (`api/training.py`)
   - Complete config loading in start_training endpoint
   - Proper TrainConfig initialization

4. **Frontend** (`frontend/`)
   - React application
   - Training dashboard
   - Config editor
   - Sample viewer

## Integration with OneTrainer

The web UI integrates with OneTrainer through:

1. **Direct imports** - Uses OneTrainer's modules directly
2. **Path setup** - `run.py` adds parent dir to sys.path
3. **Callbacks** - Hooks into TrainCallbacks for updates
4. **Commands** - Uses TrainCommands for control

### Key Integration Points

```python
# From trainer_service.py
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress
```

## Security Notes

- CORS is configured for local development only
- Production deployment should:
  - Configure proper CORS origins
  - Add authentication/authorization
  - Use HTTPS for WebSocket (wss://)
  - Implement rate limiting
  - Add input validation

## Development

### Running in Development Mode

```bash
# Backend with auto-reload
python web_ui/run.py

# The server will automatically reload on code changes
```

### Testing Imports

```python
# Test that imports work
python -c "from web_ui.backend.services.trainer_service import TrainerService; print('OK')"
```

## Troubleshooting

### Import Errors

If you get import errors for `modules.*`:
- Ensure you're running from the OneTrainer root directory
- Check that `run.py` is adding the correct path to `sys.path`

### WebSocket Connection Issues

- Check firewall settings
- Verify the correct port (8000)
- Check browser console for connection errors

### Training Not Starting

- Verify TrainConfig is valid
- Check trainer initialization in logs
- Review error messages in state.error

## Future Enhancements

- [ ] User authentication
- [ ] Multi-user support
- [ ] Training queue management
- [ ] Advanced sample comparison
- [ ] Model versioning
- [ ] Dataset preview
- [ ] Training metrics visualization
- [ ] Config templates library
- [ ] Remote training support
- [ ] Mobile-responsive UI
