"""Generation interrupt mechanism."""

from __future__ import annotations

import threading

__all__ = [
    "GenerationInterrupted",
    "InterruptFlag",
    "check_interrupt",
    "clear_interrupt",
    "set_interrupt",
]


class GenerationInterrupted(Exception):
    """Raised when generation is cancelled."""


class InterruptFlag:
    """Thread-safe interrupt flag for generation cancellation."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def set(self) -> None:
        """Signal that generation should stop."""
        self._flag.set()

    def clear(self) -> None:
        """Reset the interrupt flag."""
        self._flag.clear()

    def is_set(self) -> bool:
        """Return whether the interrupt flag is set."""
        return self._flag.is_set()

    def check(self) -> None:
        """Raise GenerationInterrupted if flag is set."""
        if self._flag.is_set():
            raise GenerationInterrupted("Generation was cancelled")


# Module-level default flag
_default_flag = InterruptFlag()


def check_interrupt() -> None:
    """Check the default interrupt flag."""
    _default_flag.check()


def set_interrupt() -> None:
    """Signal that generation should stop (module-level)."""
    _default_flag.set()


def clear_interrupt() -> None:
    """Reset the module-level interrupt flag."""
    _default_flag.clear()
