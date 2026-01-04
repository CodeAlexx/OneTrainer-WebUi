"""
Base Plugin class for OneTrainer Web UI extensions.

Plugins can extend inference capabilities similar to SD-WebUI Forge extensions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

import torch
from PIL import Image


class PluginType(Enum):
    """Types of plugins supported."""
    PREPROCESSOR = "preprocessor"  # ControlNet, IP-Adapter, depth, etc.
    SAMPLER = "sampler"  # Custom samplers
    SCHEDULER = "scheduler"  # Custom schedulers
    POSTPROCESSOR = "postprocessor"  # Face restoration, upscaling, etc.
    MODEL_LOADER = "model_loader"  # Custom model loading
    EMBEDDINGS = "embeddings"  # Textual inversion, etc.
    LORA = "lora"  # LoRA/LyCORIS variants
    GENERAL = "general"  # General purpose extensions


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    supported_models: List[str] = field(default_factory=list)  # Empty = all models
    requires_gpu: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PluginUIElement:
    """UI element configuration for plugin settings."""
    id: str
    label: str
    type: str  # 'slider', 'checkbox', 'dropdown', 'text', 'image', 'file'
    default: Any
    options: Optional[Dict[str, Any]] = None  # For dropdowns, sliders (min/max/step)


class Plugin(ABC):
    """
    Base class for OneTrainer Web UI plugins.

    Plugins should inherit from this class and implement the required methods.
    """

    def __init__(self, plugin_dir: Path):
        """
        Initialize the plugin.

        Args:
            plugin_dir: Directory where the plugin is installed
        """
        self.plugin_dir = plugin_dir
        self._enabled = True
        self._loaded = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        pass

    @property
    def ui_elements(self) -> List[PluginUIElement]:
        """Return UI elements for plugin configuration. Override to add custom UI."""
        return []

    def load(self) -> bool:
        """
        Load plugin resources (models, weights, etc.).

        Returns:
            True if loaded successfully
        """
        self._loaded = True
        return True

    def unload(self) -> bool:
        """
        Unload plugin resources to free memory.

        Returns:
            True if unloaded successfully
        """
        self._loaded = False
        return True

    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded."""
        return self._loaded

    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled

    def enable(self):
        """Enable the plugin."""
        self._enabled = True

    def disable(self):
        """Disable the plugin."""
        self._enabled = False


class PreprocessorPlugin(Plugin):
    """
    Base class for preprocessor plugins (ControlNet, IP-Adapter, etc.).

    Preprocessors modify the input to the diffusion model.
    """

    @abstractmethod
    def preprocess(
        self,
        image: Image.Image,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preprocess an image for conditioning.

        Args:
            image: Input image
            settings: Plugin-specific settings from UI

        Returns:
            Dict containing conditioning data to pass to the model
        """
        pass

    def get_conditioning_kwargs(
        self,
        preprocessed: Dict[str, Any],
        model_type: str,
    ) -> Dict[str, Any]:
        """
        Convert preprocessed data to model-specific conditioning kwargs.

        Args:
            preprocessed: Output from preprocess()
            model_type: The model type being used

        Returns:
            kwargs to pass to the pipeline
        """
        return preprocessed


class PostprocessorPlugin(Plugin):
    """
    Base class for postprocessor plugins (upscaling, face restoration, etc.).

    Postprocessors modify the output image.
    """

    @abstractmethod
    def postprocess(
        self,
        image: Image.Image,
        settings: Dict[str, Any],
    ) -> Image.Image:
        """
        Postprocess a generated image.

        Args:
            image: Generated image
            settings: Plugin-specific settings from UI

        Returns:
            Processed image
        """
        pass


class SamplerPlugin(Plugin):
    """
    Base class for custom sampler plugins.
    """

    @abstractmethod
    def get_scheduler(self, **kwargs) -> Any:
        """
        Get the scheduler/sampler instance.

        Returns:
            Scheduler instance compatible with diffusers
        """
        pass


class ModelLoaderPlugin(Plugin):
    """
    Base class for custom model loader plugins.
    """

    @abstractmethod
    def can_load(self, model_path: str, model_type: str) -> bool:
        """
        Check if this plugin can load the given model.

        Returns:
            True if this plugin can handle the model
        """
        pass

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        model_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        """
        Load the model.

        Returns:
            Loaded model/pipeline
        """
        pass
