"""
OneTrainer Web UI Plugin System

Plugins extend the inference capabilities of OneTrainer's web interface.
Similar to SD-WebUI Forge extensions, plugins can add:
- Custom preprocessors (ControlNet, IP-Adapter, etc.)
- Custom samplers and schedulers
- Post-processors
- Custom model loaders
"""

from .plugin_base import Plugin, PluginType
from .plugin_manager import PluginManager, get_plugin_manager

__all__ = ['Plugin', 'PluginType', 'PluginManager', 'get_plugin_manager']
