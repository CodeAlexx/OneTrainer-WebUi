"""
API endpoints for OneTrainer Web UI.

This package contains all REST API routers for the web UI backend:
- training: Training control endpoints (start, stop, status, progress)
- config: Configuration management endpoints (presets, validation)
- samples: Sample generation and retrieval endpoints
- system: System information endpoints (GPU info, models list)

Usage:
    from web_ui.backend.api import training, config, samples, system

    app.include_router(training.router)
    app.include_router(config.router)
    app.include_router(samples.router)
    app.include_router(system.router)
"""

from . import training
from . import config
from . import samples
from . import system
from . import settings

__all__ = ["training", "config", "samples", "system", "settings"]
