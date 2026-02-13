"""Checkpoint save/load/resume utilities for native diffusion training."""

from __future__ import annotations

import random
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from serenity.training.ema import EMAModel

__all__ = [
    "_save_module_state",
    "_save_training_state",
    "_resolve_resume_state_path",
    "_maybe_restore_training_state",
]


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
        return
    if isinstance(value, list | tuple | set):
        for nested in value:
            yield from _iter_tensors(nested)


def _save_module_state(
    module: torch.nn.Module,
    output_path: Path,
    *,
    save_dtype: torch.dtype | None = None,
) -> Path:
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().cpu()
        if save_dtype is not None and value.is_floating_point():
            value = value.to(dtype=save_dtype)
        state_dict[name] = value.contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(output_path))
        return output_path
    except (ImportError, OSError):
        fallback_path = output_path.with_suffix(".pt")
        torch.save(state_dict, fallback_path)
        return fallback_path


def _save_training_state(
    output_path: Path,
    *,
    step: int,
    model_checkpoint: Path | None,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any | None,
    ema_model: EMAModel | None,
) -> Path:
    state: dict[str, Any] = {
        "step": int(step),
        "model_checkpoint": str(model_checkpoint) if model_checkpoint is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        with suppress(Exception):
            state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    return output_path


def _resolve_resume_state_path(config: dict[str, Any], checkpoint_block: dict[str, Any], output_dir: Path) -> Path | None:
    resume_raw = (
        checkpoint_block.get("resume_state")
        or checkpoint_block.get("resume_from")
        or config.get("resume_state")
        or config.get("resume_from")
        or config.get("resume_checkpoint")
    )
    if resume_raw:
        candidate = Path(str(resume_raw)).expanduser()
        if candidate.is_dir():
            states = sorted(candidate.glob("state_step_*.pt"))
            return states[-1] if states else None
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Resume state path not found: {candidate}")

    latest = sorted(output_dir.glob("state_step_*.pt"))
    return latest[-1] if latest else None


def _maybe_restore_training_state(
    *,
    resume_state_path: Path | None,
    train_module: torch.nn.Module,
    adapter: Any | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    ema_model: EMAModel | None,
    optimizer_state_device: torch.device | None = None,
    restore_optimizer_state: bool = True,
) -> int:
    if resume_state_path is None:
        return 1

    state = torch.load(str(resume_state_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid training state file: {resume_state_path}")

    model_checkpoint_raw = state.get("model_checkpoint")
    if model_checkpoint_raw:
        model_checkpoint = Path(str(model_checkpoint_raw)).expanduser()
        if model_checkpoint.exists():
            if adapter is not None:
                with suppress(Exception):
                    adapter.load(str(model_checkpoint))
            else:
                if model_checkpoint.suffix.lower() == ".safetensors":
                    from safetensors.torch import load_file

                    model_state = load_file(str(model_checkpoint))
                else:
                    model_state = torch.load(str(model_checkpoint), map_location="cpu", weights_only=True)
                train_module.load_state_dict(model_state, strict=False)

    optimizer_state = state.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        if not restore_optimizer_state:
            optimizer.state.clear()
            print("[native/diffusion] warning: skipping optimizer state restore for this resume")
        else:
            skip_optimizer_restore = False
            if optimizer_state_device is not None and optimizer_state_device.type == "cuda":
                with suppress(StopIteration):
                    first_param = next(train_module.parameters())
                    if torch.is_tensor(first_param) and first_param.device.type == "cpu":
                        skip_optimizer_restore = True
            if skip_optimizer_restore:
                optimizer.state.clear()
                print(
                    "[native/diffusion] warning: skipping optimizer state restore for offloaded CPU-parameter training; "
                    "optimizer buffers will reinitialize"
                )
            else:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception as exc:
                    optimizer.state.clear()
                    print(f"[native/diffusion] warning: failed to restore optimizer state ({exc}); starting optimizer fresh")
                else:
                    # Torch restores optimizer buffers on CPU when loading from map_location="cpu".
                    # Ensure state tensors are on the expected training device before optimizer.step().
                    fallback_device = optimizer_state_device
                    for param, param_state in optimizer.state.items():
                        if not isinstance(param_state, dict):
                            continue
                        target_device = fallback_device
                        if target_device is None and torch.is_tensor(param):
                            target_device = param.device
                        if target_device is None:
                            continue
                        for key, value in list(param_state.items()):
                            if not torch.is_tensor(value):
                                continue
                            param_state[key] = value.to(device=target_device)

                    # Some historical states can still retain mixed-device internals after load.
                    # If so, drop optimizer buffers and continue from model weights only.
                    if optimizer_state_device is not None:
                        expected = optimizer_state_device.type
                        mixed = False
                        for param_state in optimizer.state.values():
                            for tensor in _iter_tensors(param_state):
                                if tensor.device.type != expected:
                                    mixed = True
                                    break
                            if mixed:
                                break
                        if mixed:
                            optimizer.state.clear()
                            print(
                                "[native/diffusion] warning: optimizer state remained mixed-device after resume; "
                                "resetting optimizer buffers"
                            )

    scheduler_state = state.get("scheduler_state")
    if lr_scheduler is not None and isinstance(scheduler_state, dict):
        with suppress(Exception):
            lr_scheduler.load_state_dict(scheduler_state)

    ema_state = state.get("ema_state")
    if ema_model is not None and isinstance(ema_state, dict):
        with suppress(Exception):
            ema_model.load_state_dict(ema_state)

    py_state = state.get("python_random_state")
    if py_state is not None:
        with suppress(Exception):
            random.setstate(py_state)
    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        with suppress(Exception):
            torch.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda_random_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        with suppress(Exception):
            torch.cuda.set_rng_state_all(cuda_state)

    step = int(state.get("step", 0))
    next_step = max(1, step + 1)
    print(f"[native/diffusion] resumed training from state {resume_state_path} (next step={next_step})")
    return next_step
