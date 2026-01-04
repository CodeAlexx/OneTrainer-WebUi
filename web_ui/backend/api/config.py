"""
Configuration management REST API endpoints.

Provides endpoints for managing training configurations:
- List and load configuration presets
- Get and update current configuration
- Validate configuration before training
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query

from web_ui.backend.models import (
    ConfigPresetInfo,
    ConfigPresetsResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigValidationRequest,
    ConfigValidationResponse,
    CommandResponse,
)
from web_ui.backend.services.trainer_service import get_trainer_service
from modules.util.config.TrainConfig import TrainConfig

router = APIRouter()


# Default config directory - relative to OneTrainer root
import os
ONETRAINER_ROOT = Path(os.environ.get("ONETRAINER_ROOT", Path(__file__).parent.parent.parent.parent))
# Use training_presets directory where OneTrainer stores its presets
DEFAULT_CONFIG_DIR = ONETRAINER_ROOT / "training_presets"


@router.get(
    "/presets",
    response_model=ConfigPresetsResponse,
    status_code=status.HTTP_200_OK,
    summary="List configuration presets",
    description="Get a list of all available configuration preset files.",
)
async def list_presets(
    config_dir: Optional[str] = Query(None, description="Custom config directory path")
) -> ConfigPresetsResponse:
    """
    List all available configuration presets.

    Args:
        config_dir: Optional custom config directory path

    Returns:
        ConfigPresetsResponse with list of available presets
    """
    # Use provided config_dir or default
    search_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

    if not search_dir.exists():
        return ConfigPresetsResponse(presets=[], count=0)

    presets = []
    for config_file in search_dir.glob("*.json"):
        try:
            stat = config_file.stat()
            preset_info = ConfigPresetInfo(
                name=config_file.stem,
                path=str(config_file.absolute()),
                description=None,  # Could parse from config if available
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            )
            presets.append(preset_info)
        except Exception:
            # Skip files that can't be read
            continue

    # Sort by name
    presets.sort(key=lambda p: p.name)

    return ConfigPresetsResponse(
        presets=presets,
        count=len(presets)
    )


@router.get(
    "/presets/{name}",
    response_model=ConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Load a configuration preset",
    description="Load a specific configuration preset by name.",
)
async def load_preset(
    name: str,
    config_dir: Optional[str] = Query(None, description="Custom config directory path")
) -> ConfigResponse:
    """
    Load a configuration preset by name.

    Args:
        name: Preset name (without .json extension)
        config_dir: Optional custom config directory path

    Returns:
        ConfigResponse with the preset configuration

    Raises:
        HTTPException: If preset not found or invalid JSON
    """
    # Use provided config_dir or default
    search_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    config_path = search_dir / f"{name}.json"

    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration preset '{name}' not found"
        )

    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return ConfigResponse(config=config_dict)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in preset file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load preset: {str(e)}"
        )


@router.post(
    "/presets/{name}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Save a configuration preset",
    description="Save a configuration as a named preset.",
)
async def save_preset(
    name: str,
    request: ConfigUpdateRequest,
    config_dir: Optional[str] = Query(None, description="Custom config directory path")
) -> CommandResponse:
    """
    Save a configuration as a preset.

    Args:
        name: Preset name (without .json extension)
        request: Configuration to save
        config_dir: Optional custom config directory path

    Returns:
        CommandResponse indicating success

    Raises:
        HTTPException: If save fails
    """
    # Use provided config_dir or default
    search_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

    # Create directory if it doesn't exist
    search_dir.mkdir(parents=True, exist_ok=True)

    config_path = search_dir / f"{name}.json"

    try:
        with open(config_path, "w") as f:
            json.dump(request.config, f, indent=2)

        return CommandResponse(
            success=True,
            message=f"Configuration preset '{name}' saved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {str(e)}"
        )


@router.delete(
    "/presets/{name}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a configuration preset",
    description="Delete a preset file.",
)
async def delete_preset(
    name: str,
    config_dir: Optional[str] = Query(None, description="Custom config directory path")
) -> CommandResponse:
    """
    Delete a preset.

    Args:
        name: Preset name (without .json extension)
        config_dir: Optional custom config directory path

    Returns:
        CommandResponse indicating success

    Raises:
        HTTPException: If preset not found or delete fails
    """
    # Use provided config_dir or default
    search_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    config_path = search_dir / f"{name}.json"

    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{name}' not found"
        )

    try:
        config_path.unlink()
        return CommandResponse(
            success=True,
            message=f"Preset '{name}' deleted successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preset: {str(e)}"
        )

@router.post(
    "/save-temp",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Save configuration to temporary file",
    description="Save configuration to a temporary file for training. Returns the path to the saved config.",
)
async def save_temp_config(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Save configuration to a temporary file for training.

    Args:
        request: Configuration to save

    Returns:
        Dictionary with the path to the saved config file

    Raises:
        HTTPException: If save fails
    """
    temp_dir = ONETRAINER_ROOT / "workspace" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = temp_dir / f"config_{timestamp}.json"

    try:
        with open(temp_path, "w") as f:
            json.dump(request.config, f, indent=2)

        return {"path": str(temp_path), "success": True}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save temporary config: {str(e)}"
        )


@router.get(
    "/current",
    response_model=ConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current configuration",
    description="Get the currently active training configuration.",
)
async def get_current_config() -> ConfigResponse:
    """
    Get the current training configuration.

    Returns:
        ConfigResponse with current configuration

    Raises:
        HTTPException: If no configuration is loaded
    """
    trainer_service = get_trainer_service()
    config_dict = trainer_service.get_config()

    if config_dict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No configuration currently loaded"
        )

    return ConfigResponse(config=config_dict)


@router.put(
    "/current",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Update current configuration",
    description="Update the current training configuration (must not be training).",
)
async def update_current_config(request: ConfigUpdateRequest) -> CommandResponse:
    """
    Update the current configuration.

    Args:
        request: Configuration update request

    Returns:
        CommandResponse indicating success

    Raises:
        HTTPException: If training is active or update fails
    """
    trainer_service = get_trainer_service()

    # Check if training is running
    state = trainer_service.get_state()
    if state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot update configuration while training is active"
        )

    try:
        # Create new config from dict
        train_config = TrainConfig.default_values()

        if request.partial:
            # Merge with existing config
            current = trainer_service.get_config()
            if current:
                train_config.from_dict(current)
            train_config.from_dict(request.config)
        else:
            # Replace entirely
            train_config.from_dict(request.config)

        # Initialize trainer with new config
        if not trainer_service.initialize_trainer(train_config):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize trainer with new configuration"
            )

        return CommandResponse(
            success=True,
            message="Configuration updated successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=ConfigValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate configuration",
    description="Validate a configuration without applying it.",
)
async def validate_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """
    Validate a configuration.

    Args:
        request: Configuration to validate

    Returns:
        ConfigValidationResponse with validation results
    """
    errors = []
    warnings = []

    try:
        # Try to create a TrainConfig from the dict
        train_config = TrainConfig.default_values()
        train_config.from_dict(request.config)

        # Basic validation checks
        if not hasattr(train_config, 'model_type') or train_config.model_type is None:
            errors.append("model_type is required")

        if not hasattr(train_config, 'training_method') or train_config.training_method is None:
            errors.append("training_method is required")

        if hasattr(train_config, 'learning_rate') and train_config.learning_rate is not None:
            if train_config.learning_rate <= 0:
                errors.append("learning_rate must be positive")
            elif train_config.learning_rate > 0.01:
                warnings.append("learning_rate is quite high, may cause training instability")

        if hasattr(train_config, 'batch_size') and train_config.batch_size is not None:
            if train_config.batch_size <= 0:
                errors.append("batch_size must be positive")
            elif train_config.batch_size > 32:
                warnings.append("batch_size is quite large, may cause out of memory errors")

        if hasattr(train_config, 'epochs') and train_config.epochs is not None:
            if train_config.epochs <= 0:
                errors.append("epochs must be positive")

        # Add more validation as needed...

    except Exception as e:
        errors.append(f"Configuration parsing error: {str(e)}")

    return ConfigValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


@router.get(
    "/concepts-file",
    status_code=status.HTTP_200_OK,
    summary="Load concepts from file",
    description="Load concepts from a concept file path.",
)
async def load_concepts_file(
    file_path: Optional[str] = Query(None, description="Path to concepts JSON file")
) -> Dict[str, Any]:
    """
    Load concepts from a concepts file.

    Args:
        file_path: Path to the concepts JSON file (relative to ONETRAINER_ROOT or absolute)

    Returns:
        Dict containing concepts list
    """
    if not file_path:
        return {"concepts": [], "error": "No file path provided"}
    
    # Handle relative paths
    concepts_path = Path(file_path)
    if not concepts_path.is_absolute():
        concepts_path = ONETRAINER_ROOT / file_path
    
    if not concepts_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concepts file not found: {file_path}"
        )
    
    try:
        with open(concepts_path, 'r') as f:
            concepts = json.load(f)
        return {"concepts": concepts, "file_path": str(concepts_path)}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in concepts file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load concepts file: {str(e)}"
        )


@router.get(
    "/samples-file",
    status_code=status.HTTP_200_OK,
    summary="Load sample definitions from file",
    description="Load sample definitions from a samples JSON file.",
)
async def load_samples_file(
    file_path: Optional[str] = Query(None, description="Path to samples JSON file")
) -> Dict[str, Any]:
    """
    Load sample definitions from a samples file.
    """
    if not file_path:
        return {"samples": [], "error": "No file path provided"}
    
    samples_path = Path(file_path)
    if not samples_path.is_absolute():
        samples_path = ONETRAINER_ROOT / file_path
    
    if not samples_path.exists():
        return {"samples": [], "file_path": str(samples_path), "error": "File not found"}
    
    try:
        with open(samples_path, 'r') as f:
            samples = json.load(f)
        return {"samples": samples, "file_path": str(samples_path)}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in samples file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load samples file: {str(e)}"
        )


@router.post(
    "/samples-file",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Save sample definitions to file",
    description="Save sample definitions to a samples JSON file.",
)
async def save_samples_file(
    file_path: str = Query(..., description="Path to samples JSON file"),
    request: Dict[str, Any] = None
) -> CommandResponse:
    """
    Save sample definitions to a samples file.
    """
    samples_path = Path(file_path)
    if not samples_path.is_absolute():
        samples_path = ONETRAINER_ROOT / file_path
    
    # Create directory if it doesn't exist
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        samples = request.get("samples", []) if request else []
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=4)
        
        return CommandResponse(
            success=True,
            message=f"Saved {len(samples)} sample definitions to {samples_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save samples file: {str(e)}"
        )
