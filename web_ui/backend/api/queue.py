"""
Job Queue REST API endpoints.

Provides endpoints for managing the training job queue:
- List queue and history
- Add/remove/reorder jobs
- Start/cancel jobs
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, status

from web_ui.backend.models import CommandResponse
from web_ui.backend.services.queue_service import get_queue_service, JobStatus

router = APIRouter()


class JobRequest(BaseModel):
    """Request to add a job to the queue."""
    name: str = Field(..., description="Job name/description")
    config_path: str = Field(..., description="Path to training config file")


class MoveRequest(BaseModel):
    """Request to move a job in the queue."""
    position: int = Field(..., description="New position (0-indexed)")


class JobResponse(BaseModel):
    """Response containing job data."""
    id: str
    name: str
    config_path: str
    status: str
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    progress: Optional[dict]


class QueueResponse(BaseModel):
    """Response containing queue data."""
    jobs: List[JobResponse]
    count: int
    current_job: Optional[JobResponse]


class HistoryResponse(BaseModel):
    """Response containing job history."""
    jobs: List[JobResponse]
    count: int


@router.get(
    "",
    response_model=QueueResponse,
    status_code=status.HTTP_200_OK,
    summary="List queue",
    description="Get all jobs in the queue.",
)
async def list_queue() -> QueueResponse:
    """
    List all jobs in the queue.

    Returns:
        QueueResponse with list of jobs
    """
    queue_service = get_queue_service()
    jobs = queue_service.get_queue()
    current = queue_service.get_current_job()

    return QueueResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs],
        count=len(jobs),
        current_job=JobResponse(**current.to_dict()) if current else None
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job history",
    description="Get completed/failed job history.",
)
async def get_history(limit: int = 50) -> HistoryResponse:
    """
    Get job history.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        HistoryResponse with list of completed jobs
    """
    queue_service = get_queue_service()
    jobs = queue_service.get_history(limit)

    return HistoryResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs],
        count=len(jobs)
    )


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add job to queue",
    description="Add a new job to the queue.",
)
async def add_job(request: JobRequest) -> JobResponse:
    """
    Add a new job to the queue.

    Args:
        request: Job details

    Returns:
        The created job
    """
    queue_service = get_queue_service()
    job = queue_service.add_job(request.name, request.config_path)

    return JobResponse(**job.to_dict())


@router.delete(
    "/{job_id}",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Remove job from queue",
    description="Remove a pending job from the queue.",
)
async def remove_job(job_id: str) -> CommandResponse:
    """
    Remove a job from the queue.

    Args:
        job_id: ID of job to remove

    Returns:
        CommandResponse indicating success
    """
    queue_service = get_queue_service()

    if not queue_service.remove_job(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or is currently running"
        )

    return CommandResponse(
        success=True,
        message=f"Job '{job_id}' removed from queue"
    )


@router.post(
    "/{job_id}/move",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Move job in queue",
    description="Reorder a job in the queue.",
)
async def move_job(job_id: str, request: MoveRequest) -> CommandResponse:
    """
    Move a job to a new position.

    Args:
        job_id: ID of job to move
        request: New position

    Returns:
        CommandResponse indicating success
    """
    queue_service = get_queue_service()

    if not queue_service.move_job(job_id, request.position):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or is currently running"
        )

    return CommandResponse(
        success=True,
        message=f"Job '{job_id}' moved to position {request.position}"
    )


@router.post(
    "/{job_id}/cancel",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel job",
    description="Cancel a pending or running job.",
)
async def cancel_job(job_id: str) -> CommandResponse:
    """
    Cancel a job.

    Args:
        job_id: ID of job to cancel

    Returns:
        CommandResponse indicating success
    """
    queue_service = get_queue_service()

    if not queue_service.cancel_job(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found"
        )

    return CommandResponse(
        success=True,
        message=f"Job '{job_id}' cancelled"
    )


@router.post(
    "/start-next",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Start next job",
    description="Manually start the next pending job.",
)
async def start_next() -> CommandResponse:
    """
    Start the next pending job.

    Returns:
        CommandResponse indicating success
    """
    queue_service = get_queue_service()
    job = queue_service.start_next()

    if not job:
        return CommandResponse(
            success=False,
            message="No pending jobs or a job is already running"
        )

    return CommandResponse(
        success=True,
        message=f"Started job '{job.name}'"
    )


@router.delete(
    "/history",
    response_model=CommandResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear history",
    description="Clear the job history.",
)
async def clear_history() -> CommandResponse:
    """
    Clear job history.

    Returns:
        CommandResponse indicating success
    """
    queue_service = get_queue_service()
    queue_service.clear_history()

    return CommandResponse(
        success=True,
        message="History cleared"
    )
