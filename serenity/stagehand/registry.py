"""Block registry for the stagehand runtime.

Maps model topology to an ordered set of block entries used by the
scheduler, residency map, and transfer engine.  Built once at model load
time by walking ``model.named_modules()`` and matching against a caller-
supplied pattern.  Immutable after construction + validation.

First target model: WAN 2.2 (video diffusion with temporal + spatial
attention blocks).
"""
from __future__ import annotations

import re
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from serenity.stagehand.errors import StagehandOOMError

if TYPE_CHECKING:
    pass

__all__ = ["BlockEntry", "BlockRegistry"]


# ── helpers ──────────────────────────────────────────────────────────────


def _param_size_bytes(module: nn.Module, dtype: torch.dtype) -> int:
    """Sum of all parameter sizes in *module* when stored as *dtype*."""
    total = 0
    for p in module.parameters():
        total += p.numel() * dtype.itemsize
    return total


# ── data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockEntry:
    """Immutable descriptor for a single swappable block.

    Mirrors spec Section 2.2.1.  ``module_ref`` is a weak reference so the
    registry never prevents garbage collection of the underlying module.
    """

    block_id: str
    module_ref: weakref.ref  # weak reference to nn.Module
    size_bytes: int
    dtype: torch.dtype
    dependencies: tuple[str, ...]
    group: str
    exec_order: int
    quant_format: str | None = None
    quant_meta_bytes: int = 0


# ── registry ─────────────────────────────────────────────────────────────


class BlockRegistry:
    """Ordered, immutable registry of swappable blocks.

    Typical usage::

        registry = BlockRegistry()
        registry.build_from_model(wan_model, block_pattern="spatial|temporal", group="wan", dtype=torch.bfloat16)
        registry.validate(pool_capacity_bytes=8 * 1024**3)
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, BlockEntry] = OrderedDict()
        self._frozen: bool = False

    # ── construction ─────────────────────────────────────────────────

    def build_from_model(
        self,
        model: nn.Module,
        block_pattern: str,
        group: str,
        dtype: torch.dtype,
    ) -> None:
        """Walk *model* and register every module whose name matches *block_pattern*.

        Parameters
        ----------
        model:
            The PyTorch model to scan.
        block_pattern:
            Regex pattern matched against the fully-qualified module name
            (as returned by ``named_modules()``).
        group:
            Logical group label (e.g. ``"wan"``, ``"dit"``, ``"te1"``).
        dtype:
            Target dtype used to compute ``size_bytes``.

        Raises
        ------
        RuntimeError
            If the registry has already been frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot build_from_model on a frozen registry")

        compiled = re.compile(block_pattern)
        order = len(self._entries)

        for name, module in model.named_modules():
            if not name:
                continue
            if compiled.search(name):
                block_id = name
                size = _param_size_bytes(module, dtype)
                entry = BlockEntry(
                    block_id=block_id,
                    module_ref=weakref.ref(module),
                    size_bytes=size,
                    dtype=dtype,
                    dependencies=(),
                    group=group,
                    exec_order=order,
                )
                self._entries[block_id] = entry
                order += 1

    def validate(self, pool_capacity_bytes: int) -> None:
        """Check all blocks fit in the pool and freeze the registry.

        Raises
        ------
        StagehandOOMError
            If any single block exceeds *pool_capacity_bytes*.
        """
        for entry in self._entries.values():
            if entry.size_bytes > pool_capacity_bytes:
                raise StagehandOOMError(
                    f"Block {entry.block_id!r} ({entry.size_bytes} bytes) "
                    f"exceeds pool capacity ({pool_capacity_bytes} bytes)"
                )
        self._frozen = True

    # ── queries ──────────────────────────────────────────────────────

    def get(self, block_id: str) -> BlockEntry:
        """Return the entry for *block_id*.

        Raises
        ------
        KeyError
            If *block_id* is not in the registry.
        """
        return self._entries[block_id]

    def blocks_in_order(self) -> list[BlockEntry]:
        """All entries sorted by ``exec_order``."""
        return sorted(self._entries.values(), key=lambda e: e.exec_order)

    def __len__(self) -> int:
        return len(self._entries)

    def groups(self) -> dict[str, list[BlockEntry]]:
        """Entries grouped by their ``group`` field."""
        result: dict[str, list[BlockEntry]] = {}
        for entry in self._entries.values():
            result.setdefault(entry.group, []).append(entry)
        return result

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._entries

    def __repr__(self) -> str:
        return f"BlockRegistry(blocks={len(self)}, frozen={self._frozen})"
