"""
FastAPI main application for OneTrainer Web UI.

Provides REST API and WebSocket endpoints for managing OneTrainer training sessions.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from web_ui.backend.services.trainer_service import get_trainer_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization and cleanup of the trainer service.
    """
    # Startup
    print("OneTrainer Web UI starting up...")
    trainer_service = get_trainer_service()

    # Set the event loop for async broadcasting from training threads
    try:
        loop = asyncio.get_running_loop()
        trainer_service.set_event_loop(loop)
    except RuntimeError:
        print("Warning: Could not get running event loop for trainer service")

    yield

    # Shutdown
    print("OneTrainer Web UI shutting down...")
    trainer_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="OneTrainer Web UI",
    description="Web interface for OneTrainer - A flexible AI model training tool",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "OneTrainer Web UI",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    trainer_service = get_trainer_service()
    state = trainer_service.get_state()

    return {
        "status": "healthy",
        "training_active": state.get("is_training", False),
        "trainer_status": state.get("status", "unknown")
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training updates.

    Clients connect here to receive live updates about training progress,
    status changes, and sample generation.
    """
    await websocket.accept()

    trainer_service = get_trainer_service()
    trainer_service.register_websocket(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "data": trainer_service.get_state()
        })

        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (for ping/pong, commands, etc.)
            data = await websocket.receive_text()

            # Handle ping
            if data == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        trainer_service.unregister_websocket(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        trainer_service.unregister_websocket(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# Import and include routers
# Note: These will be created separately
try:
    from web_ui.backend.api import training, config, samples, system, filesystem, concepts, queue, inference, tensorboard, caption, plugins, database, tools, settings

    app.include_router(
        training.router,
        prefix="/api/training",
        tags=["training"]
    )
    app.include_router(
        config.router,
        prefix="/api/config",
        tags=["config"]
    )
    app.include_router(
        samples.router,
        prefix="/api/samples",
        tags=["samples"]
    )
    app.include_router(
        system.router,
        prefix="/api/system",
        tags=["system"]
    )
    app.include_router(
        filesystem.router,
        prefix="/api/filesystem",
        tags=["filesystem"]
    )
    app.include_router(
        concepts.router,
        prefix="/api/concepts",
        tags=["concepts"]
    )
    app.include_router(
        queue.router,
        prefix="/api/queue",
        tags=["queue"]
    )
    app.include_router(
        inference.router,
        prefix="/api/inference",
        tags=["inference"]
    )
    app.include_router(
        tensorboard.router,
        prefix="/api/tensorboard",
        tags=["tensorboard"]
    )
    app.include_router(
        caption.router,
        prefix="/api/caption",
        tags=["caption"]
    )
    app.include_router(
        plugins.router,
        prefix="/api/plugins",
        tags=["plugins"]
    )
    app.include_router(
        database.router,
        prefix="/api/db",
        tags=["database"]
    )
    app.include_router(
        tools.router,
        prefix="/api/tools",
        tags=["tools"]
    )
    app.include_router(
        settings.router,
        prefix="/api",
        tags=["settings"]
    )
except ImportError as e:
    print(f"Warning: Could not import API routers: {e}")
    print("API routes will not be available until routers are created.")


# Static file serving for frontend (production)
import os
from pathlib import Path
from fastapi.responses import FileResponse

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    # Serve static assets (js, css, images)
    app.mount(
        "/assets",
        StaticFiles(directory=str(frontend_dist / "assets")),
        name="assets"
    )
    
    # Catchall route for SPA - must be AFTER API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Serve index.html for all non-API routes (SPA routing)
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse({"error": "Frontend not built"}, status_code=404)
else:
    print(f"Warning: Frontend dist not found at {frontend_dist}")
    print("Run 'npm run build' in web_ui/frontend to build the frontend")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    print(f"Error processing request: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_ui.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
