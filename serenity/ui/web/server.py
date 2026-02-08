"""Serenity web UI server — FastAPI application factory and entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

__all__ = ["create_app", "main"]

_STATIC_DIR = Path(__file__).parent / "static"


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
        description="Serenity diffusion model inference UI",
    )

    # Store run config on app state for later retrieval.
    app.state.host = host
    app.state.port = port

    # ------------------------------------------------------------------
    # CORS — allow localhost development frontends
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
    # Configure scanner and API module
    # ------------------------------------------------------------------
    from serenity.ui.web.api import configure, router
    from serenity.ui.web.scanner import ModelScanner

    scanner = ModelScanner(root_dirs=[models_dir])
    configure(scanner=scanner, output_dir=output_dir)

    app.include_router(router)

    # ------------------------------------------------------------------
    # Static file mounts
    # ------------------------------------------------------------------
    static_dir = _STATIC_DIR
    static_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Output images served at /output/<filename>
    app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

    # Static assets (HTML/CSS/JS) served at /static/<path>
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
    # Root — serve index.html
    # ------------------------------------------------------------------
    @app.get("/")
    async def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            # Return a minimal placeholder when the frontend hasn't been
            # built yet so the server still starts cleanly.
            from fastapi.responses import HTMLResponse

            return HTMLResponse(
                "<html><body><h1>Serenity</h1>"
                "<p>Frontend not built yet. API available at "
                "<a href='/docs'>/docs</a>.</p></body></html>"
            )
        return FileResponse(str(index_path))

    logger.info("Serenity app created (models=%s, output=%s)", models_dir, output_dir)
    return app


def main() -> None:
    """Run the Serenity web UI server."""
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
