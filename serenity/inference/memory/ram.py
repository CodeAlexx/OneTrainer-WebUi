"""System RAM pressure monitoring and management."""
from __future__ import annotations

import logging

__all__ = [
    "get_system_ram_info",
    "is_ram_pressure_high",
    "get_ram_usage_ratio",
]

logger = logging.getLogger(__name__)


def _get_psutil():
    """Lazy import psutil."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def get_system_ram_info() -> tuple[int, int, int]:
    """Get system RAM info: (total, available, used) in bytes.

    Returns (0, 0, 0) if psutil is not available.
    """
    psutil = _get_psutil()
    if psutil is None:
        return 0, 0, 0

    vm = psutil.virtual_memory()
    return vm.total, vm.available, vm.used


def get_ram_usage_ratio() -> float:
    """Get current RAM usage as a ratio (0.0 to 1.0).

    Returns 0.0 if psutil is not available.
    """
    total, available, _ = get_system_ram_info()
    if total == 0:
        return 0.0
    return 1.0 - (available / total)


def is_ram_pressure_high(threshold: float = 0.85) -> bool:
    """Check if system RAM pressure exceeds threshold.

    Args:
        threshold: Usage ratio above which pressure is considered high.
            Default 0.85 (85% usage).

    Returns:
        True if RAM usage exceeds threshold, False otherwise.
        Returns False if psutil is not available.
    """
    return get_ram_usage_ratio() > threshold
