"""VRAM tracking, budget calculation, and platform detection."""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from enum import Enum

import torch

__all__ = [
    "VRAMState",
    "VRAMBudget",
    "get_free_memory",
    "get_total_memory",
    "detect_vram_state",
    "minimum_inference_memory",
    "calculate_budget",
    "is_nvidia",
    "is_windows",
    "is_linux",
    "is_cuda_available",
    "is_xpu_available",
    "get_accelerator_type",
    "device_supports_non_blocking",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_1GB = 1024 * 1024 * 1024
_4GB = 4 * _1GB
_8GB = 8 * _1GB

# Minimum VRAM we keep free during inference (default ~1 GB).
_MIN_INFERENCE_MEMORY = _1GB

# Extra safety margin on Windows because of shared-VRAM accounting.
_EXTRA_RESERVED_WINDOWS = 600 * 1024 * 1024
_EXTRA_RESERVED_LINUX = 400 * 1024 * 1024


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VRAMState(str, Enum):
    """Coarse VRAM availability tier."""

    NO_VRAM = "no_vram"
    LOW_VRAM = "low_vram"
    NORMAL_VRAM = "normal_vram"
    HIGH_VRAM = "high_vram"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Return True if CUDA is usable on this machine."""
    return torch.cuda.is_available()


def is_nvidia() -> bool:
    """Return True when running on an NVIDIA GPU with CUDA."""
    if not is_cuda_available():
        return False
    return torch.version.cuda is not None


def is_windows() -> bool:
    """Return True on Windows."""
    return any(platform.win32_ver())


def is_linux() -> bool:
    """Return True on Linux."""
    return sys.platform.startswith("linux")


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_accelerator_type() -> str:
    """Detect available accelerator: 'cuda', 'xpu', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if is_xpu_available():
        return "xpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Memory queries
# ---------------------------------------------------------------------------

def get_total_memory(device: torch.device | None = None) -> int:
    """Return *total* memory in bytes for *device*.

    For CUDA/XPU devices this queries the driver.  For CPU / unsupported
    devices it falls back to ``0``.
    """
    if device is None:
        device = torch.device(get_accelerator_type())

    if device.type == "cpu":
        return 0

    if device.type == "cuda" and is_cuda_available():
        _free, total = torch.cuda.mem_get_info(device)
        return total

    if device.type == "xpu" and is_xpu_available():
        props = torch.xpu.get_device_properties(device)
        return props.total_memory

    return 0


def get_free_memory(device: torch.device | None = None) -> int:
    """Return *free* memory in bytes for *device*.

    For CUDA devices this queries ``torch.cuda.mem_get_info``.  For XPU
    devices, estimates free memory from total minus allocated.  For CPU or
    when no accelerator is available, returns ``0``.
    """
    if device is None:
        device = torch.device(get_accelerator_type())

    if device.type == "cpu":
        return 0

    if device.type == "cuda" and is_cuda_available():
        torch.cuda.synchronize(device)
        free, _total = torch.cuda.mem_get_info(device)
        return free

    if device.type == "xpu" and is_xpu_available():
        props = torch.xpu.get_device_properties(device)
        used = torch.xpu.memory_allocated(device)
        return props.total_memory - used

    return 0


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

def detect_vram_state(device: torch.device | None = None) -> VRAMState:
    """Classify the device into a coarse VRAM tier.

    * < 4 GB total  -> LOW_VRAM
    * < 8 GB total  -> NORMAL_VRAM
    * >= 8 GB total -> HIGH_VRAM
    * No CUDA       -> NO_VRAM
    """
    total = get_total_memory(device)
    if total == 0:
        return VRAMState.NO_VRAM
    if total < _4GB:
        return VRAMState.LOW_VRAM
    if total < _8GB:
        return VRAMState.NORMAL_VRAM
    return VRAMState.HIGH_VRAM


def minimum_inference_memory() -> int:
    """Return the minimum VRAM (bytes) to keep free during inference.

    Accounts for extra headroom on Windows due to shared-VRAM accounting.
    """
    extra = _EXTRA_RESERVED_WINDOWS if is_windows() else _EXTRA_RESERVED_LINUX
    return _MIN_INFERENCE_MEMORY + extra


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VRAMBudget:
    """Snapshot of a device's VRAM budget."""

    total: int
    """Total VRAM in bytes."""
    reserved: int
    """Bytes reserved for overhead / inference headroom."""
    available: int
    """Bytes available for model weights (total - reserved)."""
    state: VRAMState
    """Coarse VRAM tier."""


# ---------------------------------------------------------------------------
# Non-blocking transfer detection
# ---------------------------------------------------------------------------

_NON_BLOCKING_CACHE: dict[str, bool] = {}


def device_supports_non_blocking(device: torch.device) -> bool:
    """Check if device supports non-blocking memory transfers.

    Some devices or driver versions may not properly support
    ``non_blocking=True`` in ``.to()`` calls.  This function tests
    and caches the result.
    """
    key = str(device)
    if key in _NON_BLOCKING_CACHE:
        return _NON_BLOCKING_CACHE[key]

    if device.type == "cpu":
        _NON_BLOCKING_CACHE[key] = False
        return False

    try:
        t = torch.empty(256, device="cpu", pin_memory=True)
        t.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(device)
        _NON_BLOCKING_CACHE[key] = True
    except RuntimeError:
        # Non-blocking transfer not supported on this device/driver
        _NON_BLOCKING_CACHE[key] = False

    return _NON_BLOCKING_CACHE[key]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def calculate_budget(
    device: torch.device | None = None,
    reserved_fraction: float = 0.1,
) -> VRAMBudget:
    """Calculate a VRAM budget for *device*.

    Parameters
    ----------
    device:
        Target device.  Defaults to the first CUDA device.
    reserved_fraction:
        Fraction of total VRAM to reserve on top of the minimum inference
        headroom.  Clamped to ``[0.0, 0.5]``.
    """
    reserved_fraction = max(0.0, min(reserved_fraction, 0.5))

    total = get_total_memory(device)
    state = detect_vram_state(device)

    min_keep = minimum_inference_memory()
    reserved = max(min_keep, int(total * reserved_fraction))
    available = max(0, total - reserved)

    return VRAMBudget(
        total=total,
        reserved=reserved,
        available=available,
        state=state,
    )
