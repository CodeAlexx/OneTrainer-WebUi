"""
System information REST API endpoints.

Provides endpoints for system monitoring and information:
- GPU and hardware information
- Resource usage monitoring
- Available base models
"""

from typing import List, Dict, Any, Optional
import platform
import subprocess
import sys
import os
import signal

from fastapi import APIRouter, status, BackgroundTasks

from web_ui.backend.models import (
    GPUInfo,
    SystemInfoResponse,
    ModelInfo,
    ModelsListResponse,
)

router = APIRouter()


def get_nvidia_smi_metrics() -> Optional[List[Dict[str, Any]]]:
    """
    Get detailed GPU metrics from nvidia-smi.

    Returns:
        List of dictionaries with GPU metrics, or None if unavailable
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        metrics = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                try:
                    metrics.append({
                        'index': int(parts[0]),
                        'temperature': int(parts[1]) if parts[1] != '[N/A]' else None,
                        'fan_speed': int(parts[2]) if parts[2] != '[N/A]' else None,
                        'power_draw': float(parts[3]) if parts[3] != '[N/A]' else None,
                        'power_limit': float(parts[4]) if parts[4] != '[N/A]' else None,
                        'utilization': float(parts[5]) if parts[5] != '[N/A]' else None,
                    })
                except (ValueError, IndexError):
                    continue
        return metrics if metrics else None

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


@router.get(
    "/info",
    response_model=SystemInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get system information",
    description="Get comprehensive system information including CPU, memory, and GPU details.",
)
async def get_system_info() -> SystemInfoResponse:
    """
    Get system information.

    Returns:
        SystemInfoResponse with hardware and software information
    """
    try:
        import psutil
    except ImportError:
        # psutil is optional for basic functionality
        psutil = None

    # Get CPU count
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    # Get memory info
    if psutil:
        memory = psutil.virtual_memory()
        memory_total = memory.total
        memory_available = memory.available
    else:
        memory_total = 0
        memory_available = 0

    # Get GPU info
    gpus: List[GPUInfo] = []
    cuda_available = False
    cuda_version = None

    # Get nvidia-smi metrics first
    nvidia_metrics = get_nvidia_smi_metrics()
    nvidia_metrics_by_index = {}
    if nvidia_metrics:
        for m in nvidia_metrics:
            nvidia_metrics_by_index[m['index']] = m

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_total_gpu = props.total_memory
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_free = memory_total_gpu - memory_reserved

                # Get nvidia-smi metrics for this GPU
                smi_metrics = nvidia_metrics_by_index.get(i, {})

                gpu_info = GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total=memory_total_gpu,
                    memory_allocated=memory_allocated,
                    memory_reserved=memory_reserved,
                    memory_free=memory_free,
                    utilization=smi_metrics.get('utilization'),
                    temperature=smi_metrics.get('temperature'),
                    fan_speed=smi_metrics.get('fan_speed'),
                    power_draw=smi_metrics.get('power_draw'),
                    power_limit=smi_metrics.get('power_limit'),
                )
                gpus.append(gpu_info)

    except ImportError:
        pass

    # Get Python and PyTorch versions
    python_version = sys.version.split()[0]

    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"

    return SystemInfoResponse(
        gpus=gpus,
        cpu_count=cpu_count,
        memory_total=memory_total,
        memory_available=memory_available,
        python_version=python_version,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version
    )


@router.get(
    "/models",
    response_model=ModelsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List available base models",
    description="Get a list of available base models that can be used for training.",
)
async def list_models() -> ModelsListResponse:
    """
    List available base models.

    Returns:
        ModelsListResponse with list of available models

    Note: This is a simplified implementation. In a full implementation,
          this would scan for models in configured directories and/or
          query HuggingFace Hub for available models.
    """
    from modules.util.enum.ModelType import ModelType

    # Get all available model types from the enum
    models: List[ModelInfo] = []

    for model_type in ModelType:
        # Create basic model info from enum
        # In a full implementation, you would:
        # 1. Check if model exists locally
        # 2. Query HuggingFace for model availability
        # 3. Provide download URLs, sizes, etc.

        model_info = ModelInfo(
            name=model_type.value,
            type=model_type.value,
            path=None,  # Would be filled if model is local
            source="configurable",
            description=f"{model_type.value} model type"
        )
        models.append(model_info)

    return ModelsListResponse(
        models=models,
        count=len(models)
    )


def _shutdown_server():
    """Shutdown the server after a brief delay."""
    import time
    time.sleep(0.5)  # Give time for response to be sent
    os.kill(os.getpid(), signal.SIGTERM)


@router.post(
    "/shutdown",
    status_code=status.HTTP_200_OK,
    summary="Shutdown the web UI server",
    description="Gracefully shutdown the backend server.",
)
async def shutdown_server(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Shutdown the server gracefully.
    
    Returns:
        Dict with shutdown confirmation
    """
    background_tasks.add_task(_shutdown_server)
    return {"message": "Server shutting down...", "success": True}
