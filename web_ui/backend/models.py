"""
Pydantic models for OneTrainer Web UI API.

This module defines request/response models for the REST API endpoints,
providing validation, serialization, and documentation for all API interactions.
"""

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# ============================================================================
# Training Models
# ============================================================================

class TrainingStartRequest(BaseModel):
    """Request to start training with a configuration."""
    config_path: str = Field(
        ...,
        description="Path to the training configuration JSON file"
    )
    secrets_path: Optional[str] = Field(
        None,
        description="Optional path to secrets configuration file"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_path": "/path/to/config.json",
                "secrets_path": "/path/to/secrets.json"
            }
        }
    )


class TrainingStatusResponse(BaseModel):
    """Current training status."""
    is_training: bool = Field(..., description="Whether training is currently active")
    status: str = Field(..., description="Current status (idle, starting, training, stopping, completed, error)")
    error: Optional[str] = Field(None, description="Error message if status is error")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_training": True,
                "status": "training",
                "error": None
            }
        }
    )


class TrainingProgress(BaseModel):
    """Training progress information."""
    epoch: int = Field(..., description="Current epoch number")
    epoch_step: int = Field(..., description="Current step within epoch")
    epoch_sample: int = Field(..., description="Current sample within epoch")
    global_step: int = Field(..., description="Global step count across all epochs")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "epoch": 5,
                "epoch_step": 120,
                "epoch_sample": 960,
                "global_step": 1200
            }
        }
    )


class TrainingProgressResponse(BaseModel):
    """Full training progress including limits."""
    progress: Optional[TrainingProgress] = Field(None, description="Current progress metrics")
    max_step: int = Field(0, description="Maximum steps per epoch")
    max_epoch: int = Field(0, description="Maximum number of epochs")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "progress": {
                    "epoch": 5,
                    "epoch_step": 120,
                    "epoch_sample": 960,
                    "global_step": 1200
                },
                "max_step": 500,
                "max_epoch": 10
            }
        }
    )


class CommandResponse(BaseModel):
    """Generic response for command operations."""
    success: bool = Field(..., description="Whether the command was successful")
    message: str = Field(..., description="Human-readable message about the result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Training stopped successfully"
            }
        }
    )


# ============================================================================
# Configuration Models
# ============================================================================

class ConfigPresetInfo(BaseModel):
    """Information about a configuration preset."""
    name: str = Field(..., description="Preset name/identifier")
    path: str = Field(..., description="File path to the preset")
    description: Optional[str] = Field(None, description="Preset description if available")
    last_modified: Optional[datetime] = Field(None, description="Last modification timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "flux_lora_basic",
                "path": "/configs/flux_lora_basic.json",
                "description": "Basic LoRA training for Flux models",
                "last_modified": "2024-12-24T10:30:00"
            }
        }
    )


class ConfigPresetsResponse(BaseModel):
    """List of available configuration presets."""
    presets: List[ConfigPresetInfo] = Field(..., description="Available configuration presets")
    count: int = Field(..., description="Total number of presets")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "presets": [
                    {
                        "name": "flux_lora_basic",
                        "path": "/configs/flux_lora_basic.json",
                        "description": "Basic LoRA training for Flux models",
                        "last_modified": "2024-12-24T10:30:00"
                    }
                ],
                "count": 1
            }
        }
    )


class ConfigResponse(BaseModel):
    """Training configuration data."""
    config: Dict[str, Any] = Field(..., description="Full configuration dictionary")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "model_type": "FLUX_DEV_1",
                    "training_method": "LORA",
                    "learning_rate": 0.0001,
                    "epochs": 10
                }
            }
        }
    )


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    config: Dict[str, Any] = Field(..., description="Configuration dictionary to update")
    partial: bool = Field(
        False,
        description="If true, merge with existing config; if false, replace entirely"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "learning_rate": 0.0002,
                    "epochs": 15
                },
                "partial": True
            }
        }
    )


class ConfigValidationRequest(BaseModel):
    """Request to validate a configuration."""
    config: Dict[str, Any] = Field(..., description="Configuration to validate")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "model_type": "FLUX_DEV_1",
                    "training_method": "LORA"
                }
            }
        }
    )


class ConfigValidationResponse(BaseModel):
    """Configuration validation result."""
    valid: bool = Field(..., description="Whether the configuration is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "valid": False,
                "errors": ["model_type is required", "learning_rate must be positive"],
                "warnings": ["batch_size is quite large, may cause OOM"]
            }
        }
    )


# ============================================================================
# Sample Models
# ============================================================================

class SampleInfo(BaseModel):
    """Information about a generated sample."""
    id: str = Field(..., description="Unique sample identifier")
    path: str = Field(..., description="File path to the sample image/video")
    filename: str = Field(..., description="Sample filename")
    timestamp: datetime = Field(..., description="When the sample was generated")
    epoch: Optional[int] = Field(None, description="Epoch when sample was generated")
    step: Optional[int] = Field(None, description="Step when sample was generated")
    prompt: Optional[str] = Field(None, description="Prompt used for generation")
    seed: Optional[int] = Field(None, description="Seed used for generation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "sample_1200_5_120",
                "path": "/workspace/samples/sample_1200_5_120.png",
                "filename": "sample_1200_5_120.png",
                "timestamp": "2024-12-24T10:30:00",
                "epoch": 5,
                "step": 1200,
                "prompt": "a beautiful landscape",
                "seed": 42
            }
        }
    )


class SamplesListResponse(BaseModel):
    """List of generated samples."""
    samples: List[SampleInfo] = Field(..., description="List of generated samples")
    count: int = Field(..., description="Total number of samples")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "samples": [
                    {
                        "id": "sample_1200_5_120",
                        "path": "/workspace/samples/sample_1200_5_120.png",
                        "filename": "sample_1200_5_120.png",
                        "timestamp": "2024-12-24T10:30:00",
                        "epoch": 5,
                        "step": 1200
                    }
                ],
                "count": 1
            }
        }
    )


class SampleGenerateRequest(BaseModel):
    """Request to generate a sample."""
    prompt: str = Field(..., description="Text prompt for generation")
    negative_prompt: str = Field("", description="Negative prompt")
    height: int = Field(512, description="Image height in pixels")
    width: int = Field(512, description="Image width in pixels")
    seed: int = Field(42, description="Random seed")
    random_seed: bool = Field(False, description="Use random seed instead of fixed")
    diffusion_steps: int = Field(20, description="Number of diffusion steps")
    cfg_scale: float = Field(7.0, description="Classifier-free guidance scale")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "a beautiful landscape with mountains",
                "negative_prompt": "ugly, blurry",
                "height": 512,
                "width": 512,
                "seed": 42,
                "diffusion_steps": 20,
                "cfg_scale": 7.0
            }
        }
    )


# ============================================================================
# System Models
# ============================================================================

class GPUInfo(BaseModel):
    """GPU device information."""
    index: int = Field(..., description="GPU device index")
    name: str = Field(..., description="GPU device name")
    memory_total: int = Field(..., description="Total memory in bytes")
    memory_allocated: int = Field(..., description="Currently allocated memory in bytes")
    memory_reserved: int = Field(..., description="Currently reserved memory in bytes")
    memory_free: int = Field(..., description="Free memory in bytes")
    utilization: Optional[float] = Field(None, description="GPU utilization percentage (0-100)")
    temperature: Optional[int] = Field(None, description="GPU temperature in Celsius")
    fan_speed: Optional[int] = Field(None, description="Fan speed percentage (0-100)")
    power_draw: Optional[float] = Field(None, description="Current power draw in Watts")
    power_limit: Optional[float] = Field(None, description="Power limit in Watts")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 0,
                "name": "NVIDIA RTX 4090",
                "memory_total": 25769803776,
                "memory_allocated": 8589934592,
                "memory_reserved": 10737418240,
                "memory_free": 15032385536,
                "utilization": 75.5,
                "temperature": 65,
                "fan_speed": 45,
                "power_draw": 320.5,
                "power_limit": 450.0
            }
        }
    )


class SystemInfoResponse(BaseModel):
    """System information including GPUs and resources."""
    gpus: List[GPUInfo] = Field(..., description="List of available GPUs")
    cpu_count: int = Field(..., description="Number of CPU cores")
    memory_total: int = Field(..., description="Total system memory in bytes")
    memory_available: int = Field(..., description="Available system memory in bytes")
    python_version: str = Field(..., description="Python version")
    torch_version: str = Field(..., description="PyTorch version")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    cuda_version: Optional[str] = Field(None, description="CUDA version if available")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gpus": [
                    {
                        "index": 0,
                        "name": "NVIDIA RTX 4090",
                        "memory_total": 25769803776,
                        "memory_allocated": 8589934592,
                        "memory_reserved": 10737418240,
                        "memory_free": 15032385536
                    }
                ],
                "cpu_count": 16,
                "memory_total": 68719476736,
                "memory_available": 34359738368,
                "python_version": "3.10.12",
                "torch_version": "2.1.0",
                "cuda_available": True,
                "cuda_version": "12.1"
            }
        }
    )


class ModelInfo(BaseModel):
    """Information about an available base model."""
    name: str = Field(..., description="Model name/identifier")
    type: str = Field(..., description="Model type (e.g., FLUX_DEV_1, STABLE_DIFFUSION_XL_10_BASE)")
    path: Optional[str] = Field(None, description="Path to the model if local")
    source: str = Field(..., description="Model source (local, huggingface, etc)")
    description: Optional[str] = Field(None, description="Model description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "black-forest-labs/FLUX.1-dev",
                "type": "FLUX_DEV_1",
                "path": "/models/flux/FLUX.1-dev",
                "source": "huggingface",
                "description": "FLUX.1 Dev model from Black Forest Labs"
            }
        }
    )


class ModelsListResponse(BaseModel):
    """List of available base models."""
    models: List[ModelInfo] = Field(..., description="Available base models")
    count: int = Field(..., description="Total number of models")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "name": "black-forest-labs/FLUX.1-dev",
                        "type": "FLUX_DEV_1",
                        "source": "huggingface"
                    }
                ],
                "count": 1
            }
        }
    )


# ============================================================================
# Filesystem Models
# ============================================================================

class DirectoryEntry(BaseModel):
    """Information about a file or directory entry."""
    name: str = Field(..., description="File or directory name")
    path: str = Field(..., description="Full path to the entry")
    is_directory: bool = Field(..., description="Whether this entry is a directory")
    size: Optional[int] = Field(None, description="File size in bytes (None for directories)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "training_data",
                "path": "/datasets/training_data",
                "is_directory": True,
                "size": None
            }
        }
    )


class BrowseDirectoryResponse(BaseModel):
    """Response from browsing a directory."""
    path: str = Field(..., description="Absolute path to the browsed directory")
    entries: List[DirectoryEntry] = Field(..., description="List of files and directories")
    count: int = Field(..., description="Total number of entries")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/datasets/training_data",
                "entries": [
                    {
                        "name": "images",
                        "path": "/datasets/training_data/images",
                        "is_directory": True,
                        "size": None
                    },
                    {
                        "name": "example.jpg",
                        "path": "/datasets/training_data/example.jpg",
                        "is_directory": False,
                        "size": 1048576
                    }
                ],
                "count": 2
            }
        }
    )


class ImageFileInfo(BaseModel):
    """Information about an image file."""
    path: str = Field(..., description="Full path to the image file")
    filename: str = Field(..., description="Image filename")
    size: Optional[int] = Field(None, description="File size in bytes")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/datasets/images/photo001.jpg",
                "filename": "photo001.jpg",
                "size": 2097152,
                "width": 1024,
                "height": 768
            }
        }
    )


class ScanDirectoryResponse(BaseModel):
    """Response from scanning a directory for images."""
    path: str = Field(..., description="Absolute path to the scanned directory")
    total_count: int = Field(..., description="Total number of images found")
    files: List[ImageFileInfo] = Field(..., description="List of image files")
    truncated: bool = Field(
        False,
        description="Whether the file list was truncated due to max_files limit"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/datasets/training_data",
                "total_count": 150,
                "files": [
                    {
                        "path": "/datasets/training_data/image001.jpg",
                        "filename": "image001.jpg",
                        "size": 2097152,
                        "width": 1024,
                        "height": 768
                    }
                ],
                "truncated": False
            }
        }
    )


class PathValidationResponse(BaseModel):
    """Response from validating a path."""
    path: str = Field(..., description="Path that was validated (resolved to absolute)")
    exists: bool = Field(..., description="Whether the path exists")
    is_file: bool = Field(..., description="Whether the path is a file")
    is_directory: bool = Field(..., description="Whether the path is a directory")
    readable: bool = Field(..., description="Whether the path is readable")
    writable: bool = Field(..., description="Whether the path is writable")
    message: str = Field(..., description="Human-readable validation message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/datasets/training_data",
                "exists": True,
                "is_file": False,
                "is_directory": True,
                "readable": True,
                "writable": True,
                "message": "Path is a directory and is readable and writable"
            }
        }
    )


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid configuration provided",
                "details": {
                    "field": "learning_rate",
                    "constraint": "must be positive"
                }
            }
        }
    )
