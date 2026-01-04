"""
Training control REST API endpoints.

Provides endpoints for controlling OneTrainer training operations:
- Start/stop/pause/resume training
- Get training status and progress
- Monitor training metrics
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status

from web_ui.backend.models import (
    TrainingStartRequest,
    TrainingStatusResponse,
    TrainingProgressResponse,
    CommandResponse,
)
from web_ui.backend.services.trainer_service import get_trainer_service
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.SecretsConfig import SecretsConfig

router = APIRouter()


@router.post(
    "/start",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Start training",
    description="Initialize and start training with the provided configuration file.",
)
async def start_training(request: TrainingStartRequest) -> CommandResponse:
    """
    Start training with the specified configuration.

    Args:
        request: Training start request with config path and optional secrets path

    Returns:
        CommandResponse indicating success or failure

    Raises:
        HTTPException: If config is invalid, training already running, or initialization fails
    """
    trainer_service = get_trainer_service()

    # Check if training is already running
    state = trainer_service.get_state()
    if state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress. Stop current training before starting new one."
        )

    # Load configuration
    try:
        config_path = Path(request.config_path)
        if not config_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Configuration file not found: {request.config_path}"
            )

        # Parse training config
        train_config = TrainConfig.default_values()
        with open(config_path, "r") as f:
            train_config.from_dict(json.load(f))

        # Load secrets if provided
        if request.secrets_path:
            secrets_path = Path(request.secrets_path)
            if not secrets_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Secrets file not found: {request.secrets_path}"
                )

            with open(secrets_path, "r") as f:
                secrets_dict = json.load(f)
                train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in configuration file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load configuration: {str(e)}"
        )

    # Initialize trainer
    if not trainer_service.initialize_trainer(train_config):
        state = trainer_service.get_state()
        error_msg = state.get("error", "Unknown error during initialization")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

    # Start training
    if not trainer_service.start_training():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start training thread"
        )

    return CommandResponse(
        success=True,
        message="Training started successfully"
    )


@router.post(
    "/stop",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Stop training",
    description="Request training to stop gracefully at the next checkpoint.",
)
async def stop_training() -> CommandResponse:
    """
    Stop the current training session.

    Returns:
        CommandResponse indicating success or failure

    Raises:
        HTTPException: If training is not currently running
    """
    trainer_service = get_trainer_service()

    # Check if training is running
    state = trainer_service.get_state()
    if not state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is not currently running"
        )

    # Send stop command
    if not trainer_service.stop_training():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send stop command"
        )

    return CommandResponse(
        success=True,
        message="Stop command sent. Training will stop at next checkpoint."
    )


@router.post(
    "/pause",
    response_model=CommandResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Pause training (not implemented)",
    description="Pause training - not currently supported by OneTrainer core.",
)
async def pause_training() -> CommandResponse:
    """
    Pause training (not currently implemented).

    Note: OneTrainer core does not currently support pausing training.
          Use stop and resume with checkpoints instead.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Pause functionality not supported. Use stop and resume from checkpoint instead."
    )


@router.post(
    "/resume",
    response_model=CommandResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Resume training (not implemented)",
    description="Resume paused training - not currently supported by OneTrainer core.",
)
async def resume_training() -> CommandResponse:
    """
    Resume paused training (not currently implemented).

    Note: OneTrainer core does not currently support pause/resume.
          To resume training, start with a config that includes a checkpoint path.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Resume functionality not supported. Start training with continue_last_backup=true instead."
    )


@router.get(
    "/status",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get training status",
    description="Get the current training status (idle, training, stopped, error, etc).",
)
async def get_training_status() -> TrainingStatusResponse:
    """
    Get current training status.

    Returns:
        TrainingStatusResponse with current status information
    """
    trainer_service = get_trainer_service()
    state = trainer_service.get_state()

    return TrainingStatusResponse(
        is_training=state.get("is_training", False),
        status=state.get("status", "idle"),
        error=state.get("error")
    )


@router.get(
    "/progress",
    response_model=TrainingProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Get training progress",
    description="Get detailed training progress including epoch, step, and loss metrics.",
)
async def get_training_progress() -> TrainingProgressResponse:
    """
    Get detailed training progress.

    Returns:
        TrainingProgressResponse with current progress metrics
    """
    trainer_service = get_trainer_service()
    state = trainer_service.get_state()

    return TrainingProgressResponse(
        progress=state.get("progress"),
        max_step=state.get("max_step", 0),
        max_epoch=state.get("max_epoch", 0)
    )


@router.post(
    "/sample",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger sample generation",
    description="Request immediate sample generation during training.",
)
async def trigger_sample() -> CommandResponse:
    """
    Trigger sample generation during training.

    Returns:
        CommandResponse indicating success or failure
    """
    trainer_service = get_trainer_service()

    state = trainer_service.get_state()
    if not state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is not currently running"
        )

    if not trainer_service.sample_default():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send sample command"
        )

    return CommandResponse(
        success=True,
        message="Sample generation requested"
    )


@router.post(
    "/backup",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger backup",
    description="Request immediate backup creation during training.",
)
async def trigger_backup() -> CommandResponse:
    """
    Trigger backup creation during training.

    Returns:
        CommandResponse indicating success or failure
    """
    trainer_service = get_trainer_service()

    state = trainer_service.get_state()
    if not state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is not currently running"
        )

    if not trainer_service.backup():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send backup command"
        )

    return CommandResponse(
        success=True,
        message="Backup requested"
    )


@router.post(
    "/save",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger model save",
    description="Request immediate model save during training.",
)
async def trigger_save() -> CommandResponse:
    """
    Trigger model save during training.

    Returns:
        CommandResponse indicating success or failure
    """
    trainer_service = get_trainer_service()

    state = trainer_service.get_state()
    if not state.get("is_training", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is not currently running"
        )

    if not trainer_service.save():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send save command"
        )

    return CommandResponse(
        success=True,
        message="Model save requested"
    )

