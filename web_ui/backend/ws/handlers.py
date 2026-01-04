"""WebSocket message handlers for OneTrainer web UI."""

import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles incoming WebSocket messages and routes them to appropriate handlers."""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._handlers = {
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "command": self._handle_command,
            "ping": self._handle_ping,
        }

    async def handle_connection(self, websocket: WebSocket):
        """
        Main handler for a WebSocket connection.
        Manages the connection lifecycle and message routing.

        Args:
            websocket: The WebSocket connection to handle
        """
        # Connect the client
        client_id = await self.connection_manager.connect(websocket)

        try:
            # Message receiving loop
            while True:
                # Receive message from client
                try:
                    data = await websocket.receive_json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {client_id}: {e}")
                    await self._send_error(
                        client_id, "Invalid JSON format", "INVALID_JSON"
                    )
                    continue

                # Route message to appropriate handler
                await self._route_message(client_id, data)

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            # Clean up connection
            await self.connection_manager.disconnect(client_id)

    async def _route_message(self, client_id: str, message: Dict[str, Any]):
        """
        Route an incoming message to the appropriate handler.

        Args:
            client_id: The client ID that sent the message
            message: The message to route
        """
        if not isinstance(message, dict):
            await self._send_error(
                client_id, "Message must be a JSON object", "INVALID_MESSAGE_TYPE"
            )
            return

        message_type = message.get("type")
        if not message_type:
            await self._send_error(
                client_id, "Message missing 'type' field", "MISSING_TYPE"
            )
            return

        handler = self._handlers.get(message_type)
        if not handler:
            await self._send_error(
                client_id,
                f"Unknown message type: {message_type}",
                "UNKNOWN_MESSAGE_TYPE",
            )
            return

        try:
            await handler(client_id, message)
        except Exception as e:
            logger.error(
                f"Error handling message type '{message_type}' from client {client_id}: {e}",
                exc_info=True,
            )
            await self._send_error(
                client_id, f"Error processing message: {str(e)}", "PROCESSING_ERROR"
            )

    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]):
        """
        Handle subscription requests.

        Message format:
        {
            "type": "subscribe",
            "events": ["training_progress", "training_status", ...]
        }
        """
        event_types = message.get("events", [])

        if not isinstance(event_types, list):
            await self._send_error(
                client_id, "'events' must be an array", "INVALID_EVENTS"
            )
            return

        if not event_types:
            await self._send_error(
                client_id, "'events' array cannot be empty", "EMPTY_EVENTS"
            )
            return

        # Subscribe to events
        await self.connection_manager.subscribe(client_id, event_types)

        # Send confirmation
        await self.connection_manager.send_to_client(
            client_id,
            {
                "type": "subscribed",
                "events": event_types,
                "all_subscriptions": list(
                    self.connection_manager.get_client_subscriptions(client_id)
                ),
            },
        )

    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]):
        """
        Handle unsubscription requests.

        Message format:
        {
            "type": "unsubscribe",
            "events": ["training_progress", ...]
        }
        """
        event_types = message.get("events", [])

        if not isinstance(event_types, list):
            await self._send_error(
                client_id, "'events' must be an array", "INVALID_EVENTS"
            )
            return

        # Unsubscribe from events
        await self.connection_manager.unsubscribe(client_id, event_types)

        # Send confirmation
        await self.connection_manager.send_to_client(
            client_id,
            {
                "type": "unsubscribed",
                "events": event_types,
                "remaining_subscriptions": list(
                    self.connection_manager.get_client_subscriptions(client_id)
                ),
            },
        )

    async def _handle_command(self, client_id: str, message: Dict[str, Any]):
        """
        Handle command requests.

        Message format:
        {
            "type": "command",
            "command": "pause|resume|stop|...",
            "args": {...}
        }
        """
        command = message.get("command")

        if not command:
            await self._send_error(
                client_id, "'command' field is required", "MISSING_COMMAND"
            )
            return

        # Log the command
        logger.info(f"Received command '{command}' from client {client_id}")

        # TODO: Implement actual command handling when integrated with training system
        # For now, acknowledge the command
        await self.connection_manager.send_to_client(
            client_id,
            {
                "type": "command_received",
                "command": command,
                "status": "acknowledged",
                "message": f"Command '{command}' received but not yet implemented",
            },
        )

    async def _handle_ping(self, client_id: str, message: Dict[str, Any]):
        """
        Handle ping requests for connection keep-alive.

        Message format:
        {
            "type": "ping",
            "timestamp": "..."
        }
        """
        await self.connection_manager.send_to_client(
            client_id,
            {
                "type": "pong",
                "client_timestamp": message.get("timestamp"),
                "server_timestamp": self._get_timestamp(),
            },
        )

    async def _send_error(self, client_id: str, message: str, error_code: str):
        """
        Send an error message to a client.

        Args:
            client_id: The client ID to send the error to
            message: Error message
            error_code: Error code for client-side handling
        """
        await self.connection_manager.send_to_client(
            client_id,
            {
                "type": "error",
                "error_code": error_code,
                "message": message,
                "timestamp": self._get_timestamp(),
            },
        )

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()
