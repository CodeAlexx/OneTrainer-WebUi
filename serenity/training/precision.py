"""Mixed precision utilities: autocast, grad scaler, and dtype helpers.

Parity with OneTrainer's dtype_util.py.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import Parameter


# --------------------------------------------------------------------------- #
# Dtype helpers
# --------------------------------------------------------------------------- #


def torch_dtype(dtype_str: str) -> torch.dtype | None:
    """Convert a string dtype name to a torch.dtype.

    Recognizes: float16, fp16, bfloat16, bf16, float32, fp32, tfloat32.
    Returns None for quantized/unknown types.
    """
    _map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "tfloat32": torch.float32,
    }
    return _map.get(dtype_str.lower().replace("_", ""))


def allow_mixed_precision(train_dtype: str, weight_dtypes: list[str]) -> bool:
    """Return True if autocast should be enabled (dtypes are heterogeneous)."""
    all_dtypes = [d for d in [train_dtype] + weight_dtypes if d.upper() != "NONE"]
    return len(set(all_dtypes)) != 1


def enable_grad_scaling(train_dtype: str, parameters: list[Parameter]) -> bool:
    """Whether GradScaler is needed (fp16 training with fp32 parameters)."""
    trainable_dtypes = list({p.dtype for p in parameters})
    return (
        train_dtype.lower() in ("float16", "fp16", "float_16")
        and all(dt == torch.float32 for dt in trainable_dtypes)
    )


# --------------------------------------------------------------------------- #
# Autocast context helpers
# --------------------------------------------------------------------------- #


def create_autocast_context(
    device: torch.device,
    train_dtype: torch.dtype | None,
    weight_dtypes: list[torch.dtype | None] | None = None,
    enable_cache: bool = True,
) -> torch.autocast | nullcontext:
    """Create an autocast context for mixed-precision training.

    If all weight dtypes match train_dtype, autocast is disabled (returns
    a no-op context). Otherwise returns ``torch.autocast`` with the
    train_dtype.
    """
    if weight_dtypes is None:
        weight_dtypes = []
    weight_dtypes = [d for d in weight_dtypes if d is not None]

    if len(weight_dtypes) == 1 and train_dtype == weight_dtypes[0]:
        return torch.autocast(device_type=device.type, enabled=False)

    if train_dtype is None:
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=train_dtype,
        cache_enabled=enable_cache,
    )


def disable_fp16_autocast_context(
    device: torch.device,
    train_dtype: torch.dtype | None,
    fallback_dtype: torch.dtype | None,
    weight_dtypes: list[torch.dtype | None] | None = None,
    enable_cache: bool = True,
) -> tuple[torch.autocast | nullcontext, torch.dtype | None]:
    """Disable fp16 autocast and fall back to another dtype if needed.

    Returns (context, effective_dtype).
    """
    if weight_dtypes is None:
        weight_dtypes = []
    weight_dtypes = [d for d in weight_dtypes if d is not None]

    if train_dtype != torch.float16:
        return nullcontext(), train_dtype

    if len(weight_dtypes) == 1 and fallback_dtype == weight_dtypes[0]:
        return torch.autocast(device_type=device.type, enabled=False), weight_dtypes[0]

    return (
        torch.autocast(device_type=device.type, dtype=fallback_dtype, cache_enabled=enable_cache),
        fallback_dtype,
    )


# --------------------------------------------------------------------------- #
# PrecisionManager
# --------------------------------------------------------------------------- #


@dataclass
class PrecisionManager:
    """Unified autocast + GradScaler wrapper for mixed-precision training.

    Usage::

        pm = PrecisionManager(device=device, dtype=torch.bfloat16)
        with pm.autocast():
            loss = model(x)
        pm.scale(loss).backward()
        pm.step(optimizer)
        pm.update()
    """

    device: torch.device
    dtype: torch.dtype | None = None
    enabled: bool = True
    use_grad_scaler: bool = False
    _scaler: torch.amp.GradScaler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.use_grad_scaler and self.enabled,
        )

    @contextmanager
    def autocast(self):
        """Context manager for autocast."""
        if not self.enabled or self.dtype is None:
            yield
            return
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            yield

    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss for fp16 grad scaler (no-op if not using scaler)."""
        return self._scaler.scale(loss)

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer through the scaler."""
        self._scaler.step(optimizer)

    def update(self) -> None:
        """Update the scaler's scale factor."""
        self._scaler.update()

    @property
    def scaler(self) -> torch.amp.GradScaler:
        """Access the underlying GradScaler."""
        return self._scaler

    @classmethod
    def from_config(
        cls,
        device: torch.device,
        train_dtype_str: str,
        parameters: list[Parameter] | None = None,
    ) -> PrecisionManager:
        """Create from a string dtype and parameter list.

        Automatically enables GradScaler when fp16 training with fp32 params.
        """
        dt = torch_dtype(train_dtype_str)
        use_scaler = False
        if parameters and dt == torch.float16:
            use_scaler = all(p.dtype == torch.float32 for p in parameters)
        return cls(
            device=device,
            dtype=dt,
            enabled=dt is not None,
            use_grad_scaler=use_scaler,
        )


__all__ = [
    "torch_dtype",
    "allow_mixed_precision",
    "enable_grad_scaling",
    "create_autocast_context",
    "disable_fp16_autocast_context",
    "PrecisionManager",
]
