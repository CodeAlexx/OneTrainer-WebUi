"""TensorBoard API endpoints."""
import os
import subprocess
import signal
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["tensorboard"])

# Track TensorBoard process
_tensorboard_process: Optional[subprocess.Popen] = None
_tensorboard_port: int = 6006
_tensorboard_logdir: Optional[str] = None

ONETRAINER_ROOT = Path(__file__).parent.parent.parent.parent


class TensorBoardStatus(BaseModel):
    running: bool
    port: int
    logdir: Optional[str]
    url: Optional[str]


class TensorBoardStartRequest(BaseModel):
    logdir: Optional[str] = None
    port: int = 6006


@router.get("/status")
async def get_status() -> TensorBoardStatus:
    """Get TensorBoard server status."""
    global _tensorboard_process, _tensorboard_port, _tensorboard_logdir

    running = _tensorboard_process is not None and _tensorboard_process.poll() is None

    return TensorBoardStatus(
        running=running,
        port=_tensorboard_port if running else 6006,
        logdir=_tensorboard_logdir if running else None,
        url=f"http://localhost:{_tensorboard_port}" if running else None
    )


@router.post("/start")
async def start_tensorboard(request: TensorBoardStartRequest) -> TensorBoardStatus:
    """Start TensorBoard server."""
    global _tensorboard_process, _tensorboard_port, _tensorboard_logdir

    # Stop existing if running
    if _tensorboard_process is not None and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()
        _tensorboard_process.wait(timeout=5)

    # Determine log directory
    if request.logdir:
        logdir = request.logdir
    else:
        # Search for tensorboard logs in common locations
        possible_paths = [
            Path.home() / "workspace" / "tensorboard",  # User's home workspace
            ONETRAINER_ROOT / "workspace" / "tensorboard",  # OneTrainer workspace
            ONETRAINER_ROOT / "workspace",
        ]
        
        logdir = None
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                logdir = str(path)
                break
        
        if not logdir:
            raise HTTPException(status_code=404, detail="No TensorBoard logs found. Run a training first.")

    if not os.path.exists(logdir):
        raise HTTPException(status_code=404, detail=f"Log directory not found: {logdir}")

    _tensorboard_port = request.port
    _tensorboard_logdir = logdir

    try:
        # Start TensorBoard
        _tensorboard_process = subprocess.Popen(
            [
                "tensorboard",
                "--logdir", logdir,
                "--port", str(request.port),
                "--bind_all",  # Allow access from other hosts
                "--reload_interval", "5",  # Faster refresh
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        # Give it a moment to start
        import time
        time.sleep(2)

        # Check if still running
        if _tensorboard_process.poll() is not None:
            stderr = _tensorboard_process.stderr.read().decode() if _tensorboard_process.stderr else ""
            raise HTTPException(status_code=500, detail=f"TensorBoard failed to start: {stderr}")

        return TensorBoardStatus(
            running=True,
            port=_tensorboard_port,
            logdir=_tensorboard_logdir,
            url=f"http://localhost:{_tensorboard_port}"
        )

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="TensorBoard not installed. Run: pip install tensorboard")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_tensorboard() -> TensorBoardStatus:
    """Stop TensorBoard server."""
    global _tensorboard_process, _tensorboard_logdir

    if _tensorboard_process is not None:
        if _tensorboard_process.poll() is None:
            # Process is still running, terminate it
            os.killpg(os.getpgid(_tensorboard_process.pid), signal.SIGTERM)
            _tensorboard_process.wait(timeout=5)
        _tensorboard_process = None

    _tensorboard_logdir = None

    return TensorBoardStatus(
        running=False,
        port=6006,
        logdir=None,
        url=None
    )


@router.get("/logs")
async def list_log_directories():
    """List available TensorBoard log directories."""
    # Search multiple possible locations
    possible_workspaces = [
        Path.home() / "workspace" / "tensorboard",
        ONETRAINER_ROOT / "workspace" / "tensorboard",
    ]

    logs = []
    workspace_str = ""
    
    for workspace in possible_workspaces:
        if workspace.exists():
            workspace_str = str(workspace)
            for item in sorted(workspace.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if item.is_dir():
                    # Count event files
                    event_files = list(item.glob("events.out.tfevents.*"))
                    logs.append({
                        "name": item.name,
                        "path": str(item),
                        "event_count": len(event_files),
                        "modified": item.stat().st_mtime
                    })

    return {"logs": logs, "workspace": workspace_str or str(ONETRAINER_ROOT / "workspace" / "tensorboard")}
