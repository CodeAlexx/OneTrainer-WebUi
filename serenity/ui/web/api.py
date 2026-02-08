"""REST API routes for the Serenity web UI."""

from __future__ import annotations

import asyncio
import logging
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
_engine: Any = None  # InferenceEngine — typed as Any to avoid torch import at module level
_current_model: str = ""

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
    engine: Any = None,
    output_dir: str | Path = "/home/alex/serenity/output",
) -> None:
    """Bind a ``ModelScanner``, inference engine, and output directory.

    Called once during ``create_app()`` before the server starts.
    """
    global _scanner, _output_dir, _engine  # noqa: PLW0603
    _scanner = scanner
    _engine = engine
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
# Interruption signal
# ---------------------------------------------------------------------------


class _GenerationInterrupted(Exception):
    """Raised by step callback when user requests interrupt."""


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def _save_image_tensor(tensor: Any, path: Path) -> None:
    """Save a ``[C, H, W]`` tensor in ``[0, 1]`` range as a PNG file."""
    import torch

    t = tensor.detach().cpu().float().clamp(0, 1)
    if t.ndim == 3:
        t = t.permute(1, 2, 0)  # CHW -> HWC
    img_np = (t * 255).to(torch.uint8).numpy()

    try:
        from PIL import Image

        Image.fromarray(img_np).save(str(path))
    except ImportError:
        # Fallback: write as PPM (no external deps needed)
        h, w = img_np.shape[:2]
        header = f"P6\n{w} {h}\n255\n".encode()
        path.with_suffix(".ppm").write_bytes(header + img_np.tobytes())
        logger.warning("PIL not available, saved PPM to %s", path.with_suffix(".ppm"))


# ---------------------------------------------------------------------------
# Background generation task (real engine)
# ---------------------------------------------------------------------------


async def _run_generate(job_id: str, req: GenerateRequest) -> None:
    """Run image generation using the real InferenceEngine.

    Executes the engine in a thread pool so the async event loop stays
    responsive.  Step progress is bridged to WebSocket via
    ``asyncio.run_coroutine_threadsafe``.
    """
    global _current_model  # noqa: PLW0603

    _jobs[job_id]["status"] = "running"
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    try:
        # ---- Model switching ------------------------------------------------
        if req.model and req.model != _current_model:
            _jobs[job_id]["status"] = "loading_model"
            model_name = Path(req.model).stem
            await manager.broadcast({
                "type": "status_update",
                "message": f"Loading model: {model_name}",
            })

            def _load_model() -> None:
                _engine.unload_all()
                _engine._config.model_path = req.model
                _engine.load_model(req.model)

            await asyncio.to_thread(_load_model)
            _current_model = req.model
            _jobs[job_id]["status"] = "running"

        # ---- LoRA config ----------------------------------------------------
        _engine._config.lora_paths = [lr.path for lr in req.loras]
        _engine._config.lora_weights = [lr.weight for lr in req.loras]

        # ---- Progress callback (runs in the inference thread) ---------------
        def _on_step(step: int, total: int, sigma: Any, denoised: Any) -> None:
            if job_id in _interrupted:
                raise _GenerationInterrupted()
            asyncio.run_coroutine_threadsafe(
                manager.send_progress(step=step + 1, total=total),
                loop,
            )

        # ---- Run inference in thread pool -----------------------------------
        result = await asyncio.to_thread(
            _engine.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            width=req.width,
            height=req.height,
            sampler=req.sampler,
            scheduler=req.scheduler,
            batch_size=req.batch_size,
            callback=_on_step,
        )

        # ---- Save images to disk --------------------------------------------
        image_urls: list[str] = []
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        for i, img_tensor in enumerate(result.images):
            seed = result.seeds[i] if i < len(result.seeds) else 0
            filename = f"{ts}_{seed}_{i}.png"
            filepath = _output_dir / filename
            _save_image_tensor(img_tensor, filepath)
            image_urls.append(f"/output/{filename}")

        elapsed = round(time.monotonic() - t0, 3)

        job_result = {
            "images": image_urls,
            "seeds": result.seeds,
            "metadata": result.metadata,
            "elapsed": elapsed,
        }

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = job_result
        await manager.send_complete(
            images=image_urls, seeds=result.seeds, elapsed=elapsed,
        )

    except _GenerationInterrupted:
        _interrupted.discard(job_id)
        _jobs[job_id]["status"] = "interrupted"
        await manager.send_error("Generation interrupted by user")

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
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized. Check server logs.",
        )
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "request": req.model_dump()}
    background_tasks.add_task(_run_generate, job_id, req)
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
    active = sum(1 for j in _jobs.values() if j["status"] in ("running", "loading_model"))
    if _engine is not None:
        es = _engine.get_status()
        return {
            "engine": "active",
            "device": es.get("device", "unknown"),
            "attention_backend": es.get("attention_backend", "unknown"),
            "loaded_model": _current_model or None,
            "model_architecture": es.get("model_architecture"),
            "vram_free_bytes": es.get("vram_free_bytes", 0),
            "loaded_models": es.get("loaded_models", 0),
            "cache_entries": es.get("cache_entries", 0),
            "active_jobs": active,
            "total_jobs": len(_jobs),
        }
    return {
        "engine": "not_initialized",
        "loaded_model": None,
        "active_jobs": active,
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
