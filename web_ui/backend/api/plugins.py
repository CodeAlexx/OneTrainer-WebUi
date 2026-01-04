"""
Plugins REST API endpoints.

Provides endpoints for:
- Listing available plugins
- Loading/unloading plugins
- Getting plugin settings
- Managing plugin state
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from web_ui.backend.models import CommandResponse
from web_ui.plugins import get_plugin_manager

router = APIRouter()


# Request/Response Models

class PluginUIElement(BaseModel):
    """UI element configuration."""
    id: str
    label: str
    type: str
    default: Any
    options: Optional[Dict[str, Any]] = None


class PluginInfoResponse(BaseModel):
    """Plugin information."""
    name: str
    display_name: str
    version: str
    description: str
    author: str
    type: str
    enabled: bool
    loaded: bool
    supported_models: List[str]
    ui_elements: List[PluginUIElement]


class PluginsListResponse(BaseModel):
    """List of plugins."""
    plugins: List[PluginInfoResponse]
    available: List[str]  # Discovered but not loaded


class PluginActionRequest(BaseModel):
    """Request to perform action on a plugin."""
    plugin_name: str = Field(..., description="Plugin name")


# Endpoints

@router.get(
    "/",
    response_model=PluginsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List plugins",
    description="Get list of all plugins and their status.",
)
async def list_plugins() -> PluginsListResponse:
    """Get all plugins."""
    manager = get_plugin_manager()
    plugins = manager.get_plugin_infos()
    available = manager.discover_plugins()

    # Filter out already loaded plugins from available
    loaded_names = {p["name"] for p in plugins}
    available = [name for name in available if name not in loaded_names]

    return PluginsListResponse(
        plugins=[PluginInfoResponse(**p) for p in plugins],
        available=available,
    )


@router.post(
    "/load",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Load plugin",
    description="Load a plugin by name.",
)
async def load_plugin(request: PluginActionRequest) -> CommandResponse:
    """Load a plugin."""
    manager = get_plugin_manager()
    plugin = manager.load_plugin(request.plugin_name)

    if plugin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{request.plugin_name}' not found or failed to load"
        )

    return CommandResponse(
        success=True,
        message=f"Plugin '{request.plugin_name}' loaded successfully"
    )


@router.post(
    "/unload",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Unload plugin",
    description="Unload a plugin by name.",
)
async def unload_plugin(request: PluginActionRequest) -> CommandResponse:
    """Unload a plugin."""
    manager = get_plugin_manager()

    if not manager.unload_plugin(request.plugin_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{request.plugin_name}' not found"
        )

    return CommandResponse(
        success=True,
        message=f"Plugin '{request.plugin_name}' unloaded"
    )


@router.post(
    "/enable",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Enable plugin",
    description="Enable a loaded plugin.",
)
async def enable_plugin(request: PluginActionRequest) -> CommandResponse:
    """Enable a plugin."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(request.plugin_name)

    if plugin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{request.plugin_name}' not found"
        )

    plugin.enable()
    return CommandResponse(
        success=True,
        message=f"Plugin '{request.plugin_name}' enabled"
    )


@router.post(
    "/disable",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Disable plugin",
    description="Disable a loaded plugin.",
)
async def disable_plugin(request: PluginActionRequest) -> CommandResponse:
    """Disable a plugin."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(request.plugin_name)

    if plugin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{request.plugin_name}' not found"
        )

    plugin.disable()
    return CommandResponse(
        success=True,
        message=f"Plugin '{request.plugin_name}' disabled"
    )


@router.post(
    "/load-all",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Load all plugins",
    description="Discover and load all available plugins.",
)
async def load_all_plugins() -> CommandResponse:
    """Load all available plugins."""
    manager = get_plugin_manager()
    count = manager.load_all_plugins()

    return CommandResponse(
        success=True,
        message=f"Loaded {count} plugins"
    )


@router.get(
    "/{plugin_name}",
    response_model=PluginInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get plugin info",
    description="Get detailed info for a specific plugin.",
)
async def get_plugin_info(plugin_name: str) -> PluginInfoResponse:
    """Get plugin details."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(plugin_name)

    if plugin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{plugin_name}' not found"
        )

    info = plugin.info
    return PluginInfoResponse(
        name=plugin_name,
        display_name=info.name,
        version=info.version,
        description=info.description,
        author=info.author,
        type=info.plugin_type.value,
        enabled=plugin.is_enabled,
        loaded=plugin.is_loaded,
        supported_models=info.supported_models,
        ui_elements=[
            PluginUIElement(
                id=elem.id,
                label=elem.label,
                type=elem.type,
                default=elem.default,
                options=elem.options,
            )
            for elem in plugin.ui_elements
        ],
    )
