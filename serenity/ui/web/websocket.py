"""WebSocket connection manager for broadcasting live generation progress."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

__all__ = ["ConnectionManager", "manager"]


class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts JSON messages."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket connected (%d active)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass
        logger.debug("WebSocket disconnected (%d active)", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send *message* as JSON to every connected client.

        Silently drops connections that have gone stale.
        """
        stale: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)

        for ws in stale:
            self.disconnect(ws)

    async def send_progress(
        self,
        step: int,
        total: int,
        preview: str | None = None,
    ) -> None:
        """Broadcast a generation progress update."""
        msg: dict[str, Any] = {"type": "progress", "step": step, "total": total}
        if preview is not None:
            msg["preview"] = preview
        await self.broadcast(msg)

    async def send_complete(
        self,
        images: list[str],
        seeds: list[int],
        elapsed: float,
    ) -> None:
        """Broadcast a generation-complete event."""
        await self.broadcast({
            "type": "complete",
            "images": images,
            "seeds": seeds,
            "elapsed": elapsed,
        })

    async def send_error(self, message: str) -> None:
        """Broadcast an error event."""
        await self.broadcast({"type": "error", "message": message})

    async def send_status(
        self,
        model: str,
        vram_used: int,
        vram_total: int,
    ) -> None:
        """Broadcast engine status."""
        await self.broadcast({
            "type": "status",
            "model": model,
            "vram_used": vram_used,
            "vram_total": vram_total,
        })


# Module-level singleton shared by the API routes and server.
manager = ConnectionManager()
