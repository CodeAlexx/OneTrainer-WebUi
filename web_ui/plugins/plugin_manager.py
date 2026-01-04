"""
Plugin Manager for OneTrainer Web UI.

Handles plugin discovery, loading, and lifecycle management.
"""

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

from .plugin_base import (
    Plugin,
    PluginInfo,
    PluginType,
    PreprocessorPlugin,
    PostprocessorPlugin,
    SamplerPlugin,
    ModelLoaderPlugin,
)


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Singleton pattern - use get_plugin_manager() to access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_dirs: List[Path] = []

        # Set default plugin directories
        self._plugin_dirs = [
            Path(__file__).parent / "builtin",  # Built-in plugins
            Path(__file__).parent / "custom",  # User plugins
        ]

        # Create directories if they don't exist
        for plugin_dir in self._plugin_dirs:
            plugin_dir.mkdir(parents=True, exist_ok=True)

    def add_plugin_directory(self, path: Path):
        """Add a directory to search for plugins."""
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
            path.mkdir(parents=True, exist_ok=True)

    def discover_plugins(self) -> List[str]:
        """
        Discover all available plugins in plugin directories.

        Returns:
            List of plugin names found
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for plugin directories with __init__.py
            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    plugin_name = item.name
                    if plugin_name not in self._plugins:
                        discovered.append(plugin_name)

            # Also look for single-file plugins
            for item in plugin_dir.glob("*.py"):
                if item.name.startswith("_"):
                    continue
                plugin_name = item.stem
                if plugin_name not in self._plugins:
                    discovered.append(plugin_name)

        return discovered

    def load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Load a plugin by name.

        Args:
            plugin_name: Name of the plugin to load

        Returns:
            Loaded Plugin instance or None if failed
        """
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]

        # Search for plugin in all directories
        for plugin_dir in self._plugin_dirs:
            plugin_path = plugin_dir / plugin_name

            # Try directory-based plugin
            if plugin_path.is_dir() and (plugin_path / "__init__.py").exists():
                try:
                    plugin = self._load_plugin_module(plugin_path, plugin_name)
                    if plugin:
                        self._plugins[plugin_name] = plugin
                        plugin.load()
                        return plugin
                except Exception as e:
                    print(f"Failed to load plugin {plugin_name}: {e}")
                    continue

            # Try single-file plugin
            plugin_file = plugin_dir / f"{plugin_name}.py"
            if plugin_file.exists():
                try:
                    plugin = self._load_plugin_file(plugin_file, plugin_name)
                    if plugin:
                        self._plugins[plugin_name] = plugin
                        plugin.load()
                        return plugin
                except Exception as e:
                    print(f"Failed to load plugin {plugin_name}: {e}")
                    continue

        return None

    def _load_plugin_module(self, plugin_path: Path, plugin_name: str) -> Optional[Plugin]:
        """Load a directory-based plugin module."""
        spec = importlib.util.spec_from_file_location(
            f"onetrainer_plugins.{plugin_name}",
            plugin_path / "__init__.py"
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Look for Plugin subclass
        return self._find_plugin_class(module, plugin_path)

    def _load_plugin_file(self, plugin_file: Path, plugin_name: str) -> Optional[Plugin]:
        """Load a single-file plugin."""
        spec = importlib.util.spec_from_file_location(
            f"onetrainer_plugins.{plugin_name}",
            plugin_file
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Look for Plugin subclass
        return self._find_plugin_class(module, plugin_file.parent)

    def _find_plugin_class(self, module: Any, plugin_dir: Path) -> Optional[Plugin]:
        """Find and instantiate a Plugin subclass in a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
                and attr not in (PreprocessorPlugin, PostprocessorPlugin, SamplerPlugin, ModelLoaderPlugin)
            ):
                return attr(plugin_dir)
        return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if unloaded successfully
        """
        if plugin_name not in self._plugins:
            return False

        plugin = self._plugins[plugin_name]
        plugin.unload()
        del self._plugins[plugin_name]
        return True

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins."""
        return self._plugins.copy()

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all loaded plugins of a specific type."""
        return [
            p for p in self._plugins.values()
            if p.info.plugin_type == plugin_type and p.is_enabled
        ]

    def get_preprocessors(self) -> List[PreprocessorPlugin]:
        """Get all loaded preprocessor plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, PreprocessorPlugin) and p.is_enabled
        ]

    def get_postprocessors(self) -> List[PostprocessorPlugin]:
        """Get all loaded postprocessor plugins."""
        return [
            p for p in self._plugins.values()
            if isinstance(p, PostprocessorPlugin) and p.is_enabled
        ]

    def load_all_plugins(self) -> int:
        """
        Discover and load all available plugins.

        Returns:
            Number of plugins loaded
        """
        discovered = self.discover_plugins()
        loaded = 0

        for plugin_name in discovered:
            if self.load_plugin(plugin_name):
                loaded += 1

        return loaded

    def get_plugin_infos(self) -> List[Dict[str, Any]]:
        """Get info for all loaded plugins as dicts."""
        infos = []
        for name, plugin in self._plugins.items():
            info = plugin.info
            infos.append({
                "name": name,
                "display_name": info.name,
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "type": info.plugin_type.value,
                "enabled": plugin.is_enabled,
                "loaded": plugin.is_loaded,
                "supported_models": info.supported_models,
                "ui_elements": [
                    {
                        "id": elem.id,
                        "label": elem.label,
                        "type": elem.type,
                        "default": elem.default,
                        "options": elem.options,
                    }
                    for elem in plugin.ui_elements
                ],
            })
        return infos


# Singleton accessor
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the plugin manager singleton instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
