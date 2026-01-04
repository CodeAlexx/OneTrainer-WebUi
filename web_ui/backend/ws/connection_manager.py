"""WebSocket connection manager for OneTrainer web UI."""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        # Active connections mapped by client ID
        self._connections: Dict[str, WebSocket] = {}
        # Client subscriptions (client_id -> set of event types)
        self._subscriptions: Dict[str, Set[str]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        # Connection counter for generating client IDs
        self._connection_counter = 0

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept

        Returns:
            str: Unique client ID for this connection
        """
        await websocket.accept()

        async with self._lock:
            # Generate unique client ID
            self._connection_counter += 1
            client_id = f"client_{self._connection_counter}_{datetime.now().timestamp()}"

            # Store connection and initialize subscriptions
            self._connections[client_id] = websocket
            self._subscriptions[client_id] = set()

            logger.info(f"WebSocket client connected: {client_id}")
            logger.info(f"Total active connections: {len(self._connections)}")

        # Send welcome message
        await self._send_to_client(
            client_id,
            {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return client_id

    async def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection.

        Args:
            client_id: The client ID to disconnect
        """
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
                logger.info(f"WebSocket client disconnected: {client_id}")

            if client_id in self._subscriptions:
                del self._subscriptions[client_id]

            logger.info(f"Total active connections: {len(self._connections)}")

    async def subscribe(self, client_id: str, event_types: list[str]):
        """
        Subscribe a client to specific event types.

        Args:
            client_id: The client ID to subscribe
            event_types: List of event types to subscribe to
        """
        async with self._lock:
            if client_id not in self._subscriptions:
                self._subscriptions[client_id] = set()

            self._subscriptions[client_id].update(event_types)
            logger.debug(
                f"Client {client_id} subscribed to: {event_types}. "
                f"Total subscriptions: {self._subscriptions[client_id]}"
            )

    async def unsubscribe(self, client_id: str, event_types: list[str]):
        """
        Unsubscribe a client from specific event types.

        Args:
            client_id: The client ID to unsubscribe
            event_types: List of event types to unsubscribe from
        """
        async with self._lock:
            if client_id in self._subscriptions:
                self._subscriptions[client_id].difference_update(event_types)
                logger.debug(
                    f"Client {client_id} unsubscribed from: {event_types}. "
                    f"Remaining subscriptions: {self._subscriptions[client_id]}"
                )

    async def broadcast(self, message: Dict[str, Any], event_type: Optional[str] = None):
        """
        Broadcast a message to all connected clients or those subscribed to an event type.

        Args:
            message: The message to broadcast
            event_type: Optional event type to filter subscribers
        """
        if not self._connections:
            return

        # Determine which clients should receive this message
        target_clients = []
        async with self._lock:
            for client_id, websocket in self._connections.items():
                # If event_type is specified, only send to subscribed clients
                if event_type:
                    if event_type in self._subscriptions.get(client_id, set()):
                        target_clients.append((client_id, websocket))
                else:
                    # No event type filter, send to all
                    target_clients.append((client_id, websocket))

        # Send to all target clients
        disconnected_clients = []
        for client_id, websocket in target_clients:
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                logger.warning(f"Client {client_id} disconnected during broadcast")
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific client.

        Args:
            client_id: The target client ID
            message: The message to send
        """
        await self._send_to_client(client_id, message)

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """
        Internal method to send message to a specific client.

        Args:
            client_id: The target client ID
            message: The message to send
        """
        websocket = self._connections.get(client_id)
        if not websocket:
            logger.warning(f"Client {client_id} not found")
            return

        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            logger.warning(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            await self.disconnect(client_id)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)

    def get_client_subscriptions(self, client_id: str) -> Set[str]:
        """Get the subscriptions for a specific client."""
        return self._subscriptions.get(client_id, set()).copy()

    async def broadcast_system_info(self):
        """Broadcast system information to all connected clients."""
        await self.broadcast(
            {
                "type": "system_info",
                "active_connections": self.get_connection_count(),
                "timestamp": datetime.now().isoformat(),
            }
        )
