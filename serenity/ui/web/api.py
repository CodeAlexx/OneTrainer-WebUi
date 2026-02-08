"""REST API routes for the Serenity web UI."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from serenity.ui.web.scanner import ModelScanner
from serenity.ui.web.websocket import manager

logger = logging.getLogger(__name__)

__all__ = ["router", "configure"]

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Module-level state (configured at startup via ``configure()``)
# ---------------------------------------------------------------------------

_scanner: ModelScanner | None = None
_output_dir: Path = Path("/home/alex/serenity/output")

# Track active generation jobs.
_jobs: dict[str, dict[str, Any]] = {}
_interrupted: set[str] = set()

# Sampler and scheduler names — mirrors the enums in
# ``serenity.inference.sampling`` so the UI can populate dropdowns
# without importing torch.
_SAMPLERS: list[str] = [
    "euler",
    "euler_a",
    "dpm_2m",
    "dpm_2m_sde",
    "dpm_pp_2m",
    "dpm_pp_2m_sde",
    "lcm",
    "heun",
    "deis",
    "unipc",
]

_SCHEDULERS: list[str] = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "ays",
]


def configure(
    scanner: ModelScanner,
    output_dir: str | Path = "/home/alex/serenity/output",
) -> None:
    """Bind a ``ModelScanner`` and output directory to the API module.

    Called once during ``create_app()`` before the server starts.
    """
    global _scanner, _output_dir  # noqa: PLW0603
    _scanner = scanner
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class LoraSpec(BaseModel):
    """A single LoRA with path and weight."""

    path: str
    weight: float = 1.0


class GenerateRequest(BaseModel):
    """Parameters for an image generation request."""

    prompt: str
    negative_prompt: str = ""
    model: str = ""
    seed: int = -1
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    sampler: str = "euler"
    scheduler: str = "normal"
    batch_size: int = Field(default=1, ge=1, le=16)
    loras: list[LoraSpec] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    """Immediate response with a job identifier."""

    job_id: str
    status: str = "queued"


class GenerateResult(BaseModel):
    """Final generation result (returned via ``/api/job/{job_id}`` or WS)."""

    images: list[str]
    seeds: list[int]
    metadata: dict[str, Any]
    elapsed: float


# ---------------------------------------------------------------------------
# Background generation task (mock implementation)
# ---------------------------------------------------------------------------

async def _mock_generate(job_id: str, req: GenerateRequest) -> None:
    """Simulate a generation run, sending progress over WebSocket.

    This placeholder will be replaced with real
    ``InferenceEngine.generate()`` calls once the engine integration
    is complete.
    """
    _jobs[job_id]["status"] = "running"
    t0 = time.monotonic()
    total_steps = req.steps
    seeds: list[int] = []
    image_urls: list[str] = []

    for batch_idx in range(req.batch_size):
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        seeds.append(seed)

    try:
        for step in range(1, total_steps + 1):
            if job_id in _interrupted:
                _interrupted.discard(job_id)
                await manager.send_error("Generation interrupted by user")
                _jobs[job_id]["status"] = "interrupted"
                return

            # Simulate computation time per step.
            await asyncio.sleep(0.05)
            await manager.send_progress(step=step, total=total_steps)

        # Build placeholder image paths.
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        for i, seed in enumerate(seeds):
            filename = f"{ts}_{seed}_{i}.png"
            filepath = _output_dir / filename
            # Write a minimal placeholder file so the path is valid.
            filepath.write_bytes(b"")
            image_urls.append(f"/output/{filename}")

        elapsed = round(time.monotonic() - t0, 3)

        result = {
            "images": image_urls,
            "seeds": seeds,
            "metadata": {
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "model": req.model,
                "steps": req.steps,
                "cfg_scale": req.cfg_scale,
                "width": req.width,
                "height": req.height,
                "sampler": req.sampler,
                "scheduler": req.scheduler,
            },
            "elapsed": elapsed,
        }

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = result
        await manager.send_complete(images=image_urls, seeds=seeds, elapsed=elapsed)

    except Exception as exc:
        logger.exception("Generation failed for job %s", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)
        await manager.send_error(str(exc))


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks) -> GenerateResponse:
    """Queue an image generation job.

    Returns immediately with a ``job_id``.  Progress updates are
    broadcast over the ``/ws`` WebSocket.  Poll ``/api/job/{job_id}``
    to retrieve the final result.
    """
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "request": req.model_dump()}
    background_tasks.add_task(_mock_generate, job_id, req)
    return GenerateResponse(job_id=job_id, status="queued")


@router.get("/job/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Return the current state of a generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/models")
async def list_models() -> dict[str, list[dict[str, Any]]]:
    """Return all scanned models grouped by category."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not configured")
    from dataclasses import asdict
    return {
        cat: [asdict(m) for m in models]
        for cat, models in _scanner.scan_all().items()
    }


@router.get("/models/{category}")
async def list_models_by_category(category: str) -> list[dict[str, Any]]:
    """Return models for a specific category."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not configured")
    from dataclasses import asdict
    all_models = _scanner.scan_all()
    if category not in all_models:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
    return [asdict(m) for m in all_models[category]]


@router.post("/models/refresh")
async def refresh_models() -> dict[str, int]:
    """Force a re-scan of model directories and return counts."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not configured")
    refreshed = _scanner.refresh()
    return {cat: len(models) for cat, models in refreshed.items()}


@router.get("/status")
async def status() -> dict[str, Any]:
    """Return engine status information."""
    # TODO: Wire to real engine state once InferenceEngine is integrated.
    return {
        "engine": "mock",
        "loaded_model": None,
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "running"),
        "total_jobs": len(_jobs),
    }


@router.get("/samplers")
async def list_samplers() -> list[str]:
    """Return available sampler algorithm names."""
    return _SAMPLERS


@router.get("/schedulers")
async def list_schedulers() -> list[str]:
    """Return available noise scheduler names."""
    return _SCHEDULERS


@router.post("/interrupt")
async def interrupt() -> dict[str, str]:
    """Interrupt all currently running generation jobs."""
    interrupted_ids: list[str] = []
    for job_id, job in _jobs.items():
        if job["status"] == "running":
            _interrupted.add(job_id)
            interrupted_ids.append(job_id)
    if not interrupted_ids:
        return {"status": "no_active_jobs"}
    return {"status": "interrupted", "jobs": ",".join(interrupted_ids)}
