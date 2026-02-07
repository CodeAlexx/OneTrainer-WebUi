"""Base adapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AdapterProtocol(Protocol):
    def apply(self, module):  # pragma: no cover - protocol only
        ...
