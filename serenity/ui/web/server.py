"""Serenity API server — FastAPI application factory and entry point.

Provides REST and WebSocket endpoints for inference and training.
The browser-based frontend has been removed; use the TUI instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

__all__ = ["create_app", "main"]


def create_app(
    models_dir: str = "/home/alex/EriDiffusion/Models",
    output_dir: str = "/home/alex/serenity/output",
    host: str = "0.0.0.0",
    port: int = 7860,
) -> FastAPI:
    """Build and return the configured FastAPI application.

    Parameters
    ----------
    models_dir:
        Root directory containing model subdirectories.
    output_dir:
        Directory where generated images are saved.
    host:
        Bind address (stored on ``app.state`` for the runner).
    port:
        Bind port (stored on ``app.state`` for the runner).
    """
    app = FastAPI(
        title="Serenity",
        version="0.1.0",
        description="Serenity inference API server",
    )

    # Store run config on app state for later retrieval.
    app.state.host = host
    app.state.port = port

    # ------------------------------------------------------------------
    # CORS — allow external frontends (EriUI, etc.)
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:7860",
            "http://127.0.0.1:7860",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Configure scanner, inference engine, and API module
    # ------------------------------------------------------------------
    from serenity.ui.web.api import configure, router
    from serenity.ui.web.scanner import ModelScanner

    scanner = ModelScanner(root_dirs=[models_dir])

    # Create the inference engine.  Fails gracefully if torch is not
    # available (e.g. lightweight UI-only testing).
    engine = None
    try:
        from serenity.inference import InferenceConfig, InferenceEngine

        engine = InferenceEngine(InferenceConfig())
        logger.info("Inference engine ready")
    except Exception as exc:
        logger.warning("Could not initialise inference engine: %s", exc)

    configure(scanner=scanner, engine=engine, output_dir=output_dir)

    app.include_router(router)

    # ------------------------------------------------------------------
    # Output images served at /output/<filename>
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

    # ------------------------------------------------------------------
    # WebSocket endpoint
    # ------------------------------------------------------------------
    from serenity.ui.web.websocket import manager

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive; client may send pings or commands.
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    # ------------------------------------------------------------------
    # Root — redirect to API docs
    # ------------------------------------------------------------------
    from fastapi.responses import RedirectResponse

    @app.get("/")
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    logger.info("Serenity API created (models=%s, output=%s)", models_dir, output_dir)
    return app


def main() -> None:
    """Run the Serenity API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    uvicorn.run(
        app,
        host=app.state.host,
        port=app.state.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
