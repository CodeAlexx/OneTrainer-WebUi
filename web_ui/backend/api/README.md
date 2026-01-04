# OneTrainer Web UI API

REST API endpoints for OneTrainer web interface.

## Overview

This API provides comprehensive control over OneTrainer's training operations through HTTP endpoints. All endpoints use JSON for request/response bodies and follow RESTful conventions.

## API Routers

### 1. Training API (`/training`)

Control training operations.

**Endpoints:**

- `POST /training/start` - Start training with config file
  - Request: `{ "config_path": "path/to/config.json", "secrets_path": "path/to/secrets.json" }`
  - Response: `{ "success": true, "message": "Training started successfully" }`

- `POST /training/stop` - Stop current training
  - Response: `{ "success": true, "message": "Stop command sent" }`

- `POST /training/pause` - Pause training (not implemented)
- `POST /training/resume` - Resume training (not implemented)

- `GET /training/status` - Get training status
  - Response: `{ "is_training": true, "status": "training", "error": null }`

- `GET /training/progress` - Get detailed progress
  - Response: `{ "progress": {...}, "max_step": 500, "max_epoch": 10 }`

### 2. Configuration API (`/config`)

Manage training configurations and presets.

**Endpoints:**

- `GET /config/presets` - List all configuration presets
  - Query: `?config_dir=/path/to/configs`
  - Response: `{ "presets": [...], "count": 5 }`

- `GET /config/presets/{name}` - Load specific preset
  - Response: `{ "config": {...} }`

- `POST /config/presets/{name}` - Save configuration as preset
  - Request: `{ "config": {...}, "partial": false }`

- `GET /config/current` - Get current active configuration
- `PUT /config/current` - Update current configuration
  - Request: `{ "config": {...}, "partial": true }`

- `POST /config/validate` - Validate configuration
  - Request: `{ "config": {...} }`
  - Response: `{ "valid": true, "errors": [], "warnings": [] }`

### 3. Samples API (`/samples`)

Generate and retrieve training samples.

**Endpoints:**

- `GET /samples` - List all generated samples
  - Query: `?limit=20&samples_dir=/path/to/samples`
  - Response: `{ "samples": [...], "count": 15 }`

- `POST /samples/generate` - Generate custom sample
  - Request: `{ "prompt": "...", "height": 512, "width": 512, ... }`
  - Response: `{ "success": true, "message": "Sample generation requested" }`

- `POST /samples/generate/default` - Generate with default settings
  - Response: `{ "success": true, "message": "..." }`

- `GET /samples/{sample_id}` - Download sample file
  - Returns: Image/video file (PNG, JPG, MP4, etc.)

### 4. System API (`/system`)

System information and available models.

**Endpoints:**

- `GET /system/info` - Get system information
  - Response: GPU info, CPU count, memory, Python/PyTorch versions, CUDA info

- `GET /system/models` - List available base models
  - Response: All model types from ModelType enum

## Data Models

All request/response models are defined in `/home/alex/OneTrainer/web_ui/backend/models.py` using Pydantic for validation.

### Key Models:

- `TrainingStartRequest` - Config paths for starting training
- `TrainingStatusResponse` - Current training status
- `TrainingProgressResponse` - Detailed progress metrics
- `ConfigPresetInfo` - Configuration preset metadata
- `ConfigValidationResponse` - Validation results
- `SampleInfo` - Sample metadata
- `SampleGenerateRequest` - Custom sample parameters
- `GPUInfo` - GPU device information
- `SystemInfoResponse` - System hardware/software info
- `ModelInfo` - Base model information

## Integration

### Using in FastAPI Application:

```python
from fastapi import FastAPI
from web_ui.backend.api import training, config, samples, system

app = FastAPI(title="OneTrainer API")

# Include all routers
app.include_router(training.router)
app.include_router(config.router)
app.include_router(samples.router)
app.include_router(system.router)
```

### Router Prefixes:

- Training: `/training`
- Config: `/config`
- Samples: `/samples`
- System: `/system`

## Error Handling

All endpoints use standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `409 Conflict` - Operation conflict (e.g., training already running)
- `500 Internal Server Error` - Server error
- `501 Not Implemented` - Feature not yet implemented

Error responses follow this format:
```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": { ... }
}
```

## Service Layer

The API uses `TrainerService` (singleton) for managing training state:

- Located in: `/home/alex/OneTrainer/web_ui/backend/services/trainer_service.py`
- Wraps OneTrainer's `GenericTrainer`, `TrainCommands`, `TrainCallbacks`
- Thread-safe state management
- WebSocket broadcasting for real-time updates

## Dependencies

Required OneTrainer modules:

- `modules.util.config.TrainConfig`
- `modules.util.config.SecretsConfig`
- `modules.util.config.SampleConfig`
- `modules.util.enum.ModelType`
- `modules.trainer.GenericTrainer`
- `modules.util.commands.TrainCommands`
- `modules.util.callbacks.TrainCallbacks`

External dependencies:

- `fastapi` - Web framework
- `pydantic` - Data validation
- `torch` - For GPU information (optional)
- `psutil` - For system info (optional)

## File Structure

```
/home/alex/OneTrainer/web_ui/backend/
├── api/
│   ├── __init__.py       - Router exports
│   ├── training.py       - Training control endpoints
│   ├── config.py         - Configuration management
│   ├── samples.py        - Sample generation/retrieval
│   ├── system.py         - System information
│   └── README.md         - This file
├── models.py             - Pydantic models
├── services/
│   └── trainer_service.py - Training service singleton
└── ws/
    └── __init__.py       - WebSocket handlers
```

## Testing Examples

### Start Training:
```bash
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/config.json"}'
```

### Get Training Status:
```bash
curl http://localhost:8000/training/status
```

### List Samples:
```bash
curl http://localhost:8000/samples?limit=10
```

### Get System Info:
```bash
curl http://localhost:8000/system/info
```

### Validate Config:
```bash
curl -X POST http://localhost:8000/config/validate \
  -H "Content-Type: application/json" \
  -d '{"config": {"model_type": "FLUX_DEV_1", "training_method": "LORA"}}'
```

## Next Steps

To use these APIs in a complete web application:

1. Create FastAPI application that includes these routers
2. Set up WebSocket endpoints for real-time training updates
3. Build frontend that consumes these APIs
4. Add authentication/authorization if needed
5. Configure CORS for web access
6. Add rate limiting for production use

## Notes

- Pause/Resume endpoints return 501 (Not Implemented) as OneTrainer core doesn't support this
- Sample file serving supports images (PNG, JPG, WebP) and videos (MP4, AVI, MOV)
- Configuration presets are discovered from `./configs` directory by default
- Sample discovery parses filenames for metadata (format: `sample_<step>_<epoch>_<epoch_step>.ext`)
