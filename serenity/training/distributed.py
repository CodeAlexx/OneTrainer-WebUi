"""Multi-GPU / DDP utilities.

Provides distributed training setup/teardown, rank management, gradient
reduction, synchronization barriers, and parameter broadcasting helpers.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core state queries
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Whether distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """Global rank of the current process (0 if not distributed)."""
    return dist.get_rank() if is_enabled() else 0


def local_rank() -> int:
    """Local rank within the node (from env or fallback to global rank)."""
    if not is_enabled():
        return 0
    return int(os.environ.get("LOCAL_RANK", rank()))


def world_size() -> int:
    """Total number of processes (1 if not distributed)."""
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """Whether this is rank 0 (the main/master process)."""
    return rank() == 0


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_distributed(
    backend: str = "nccl",
    *,
    init_method: str | None = None,
) -> None:
    """Initialize the distributed process group.

    Parameters
    ----------
    backend:
        Communication backend (``"nccl"`` for GPU, ``"gloo"`` for CPU).
    init_method:
        URL for init (defaults to ``env://``).
    """
    if is_enabled():
        logger.warning("Distributed already initialized, skipping setup_distributed()")
        return

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")

    kwargs: dict[str, Any] = {"backend": backend}
    if init_method is not None:
        kwargs["init_method"] = init_method

    dist.init_process_group(**kwargs)

    # Pin each process to its local GPU
    lr = local_rank()
    if torch.cuda.is_available() and lr < torch.cuda.device_count():
        torch.cuda.set_device(lr)

    logger.info(
        "Distributed initialized: rank=%d, world_size=%d, backend=%s, local_rank=%d",
        rank(), world_size(), backend, lr,
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if is_enabled():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

def barrier() -> None:
    """Synchronize all processes (no-op if not distributed)."""
    if is_enabled():
        dist.barrier()


@contextmanager
def sequential_execution(enabled: bool = True) -> Generator[None, None, None]:
    """Execute code sequentially across all ranks.

    Yields once per rank in order, with barriers between each.
    Implemented as a context manager for safer usage.
    """
    if not enabled or not is_enabled():
        yield
        return

    for current in range(world_size()):
        if current == rank():
            yield
        barrier()


@contextmanager
def main_process_first(enabled: bool = True) -> Generator[None, None, None]:
    """Execute code on the main process first, then all others."""
    if not enabled or not is_enabled():
        yield
        return

    if is_main_process():
        yield
    barrier()
    if not is_main_process():
        yield
    barrier()


# ---------------------------------------------------------------------------
# Gradient reduction
# ---------------------------------------------------------------------------

def reduce_tensor_mean(tensor: torch.Tensor) -> None:
    """All-reduce a tensor with mean across ranks (in-place)."""
    if not is_enabled():
        return
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()


def reduce_grads_mean(
    params: list[torch.Tensor],
    *,
    async_op: bool = False,
) -> None:
    """All-reduce gradients across ranks (mean) for a list of parameters.

    Parameters
    ----------
    params:
        Model parameters whose ``.grad`` should be reduced.
    async_op:
        If True, use async all_reduce and store handles for later completion.
        Call ``finish_async_grad_reduce()`` before optimizer.step().
    """
    if not is_enabled():
        return

    for param in params:
        if not param.requires_grad or param.grad is None:
            continue

        if async_op:
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            _async_handles.append((work, param))
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size()


# Async handle tracking
_async_handles: deque[tuple[Any, torch.Tensor]] = deque()


def finish_async_grad_reduce() -> None:
    """Wait for all pending async gradient reductions and apply mean."""
    ws = world_size()
    while _async_handles:
        work, param = _async_handles.popleft()
        work.wait()
        if param.grad is not None:
            param.grad /= ws


# ---------------------------------------------------------------------------
# Parameter broadcasting / divergence
# ---------------------------------------------------------------------------

@torch.no_grad()
def broadcast_parameters(
    params: list[torch.Tensor],
    device: torch.device,
    src: int = 0,
) -> None:
    """Broadcast parameters from *src* rank to all others."""
    if not is_enabled():
        return

    for param in params:
        gpu_param = param.to(device)
        dist.broadcast(gpu_param, src=src)
        if rank() != src and gpu_param is not param:
            param.copy_(gpu_param)


@torch.no_grad()
def parameter_divergence(
    params: list[torch.Tensor],
    device: torch.device,
) -> float | None:
    """Measure absolute divergence of parameters across ranks.

    Returns the sum of absolute differences on rank 0, ``None`` on others.
    Only useful for debugging multi-GPU training.
    """
    if not is_enabled():
        return 0.0

    diff = 0.0
    for param in params:
        param_list = (
            [torch.zeros_like(param, device=device) for _ in range(world_size())]
            if is_main_process()
            else None
        )
        dist.gather(param.to(device), param_list, dst=0)
        if is_main_process() and param_list is not None:
            for r in range(1, world_size()):
                diff += torch.sum(torch.abs(param_list[0] - param_list[r])).item()

    return diff if is_main_process() else None


@torch.no_grad()
def warn_parameter_divergence(
    params: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Log a warning if parameters have diverged across ranks."""
    divergence = parameter_divergence(params, device)
    if divergence is not None and divergence > 0:
        logger.warning("Parameter divergence between GPUs: %f", divergence)


# ---------------------------------------------------------------------------
# Distributed iteration helper
# ---------------------------------------------------------------------------

def distributed_enumerate(
    iterable: Any,
    *,
    distribute: bool = True,
) -> Generator[tuple[int, Any], None, None]:
    """Enumerate an iterable, distributing items across ranks.

    Each rank only yields items where ``index % world_size == rank``.
    If *distribute* is False, only the main process iterates.
    """
    if distribute:
        for i, x in enumerate(iterable):
            if i % world_size() == rank():
                yield i, x
    elif is_main_process():
        yield from enumerate(iterable)


__all__ = [
    # State queries
    "is_enabled",
    "rank",
    "local_rank",
    "world_size",
    "is_main_process",
    # Setup / teardown
    "setup_distributed",
    "cleanup_distributed",
    # Synchronization
    "barrier",
    "sequential_execution",
    "main_process_first",
    # Gradient reduction
    "reduce_tensor_mean",
    "reduce_grads_mean",
    "finish_async_grad_reduce",
    # Parameter utilities
    "broadcast_parameters",
    "parameter_divergence",
    "warn_parameter_divergence",
    # Iteration
    "distributed_enumerate",
]
