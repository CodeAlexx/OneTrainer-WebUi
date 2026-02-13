"""Stagehand — bounded block-swapping runtime for Serenity.

Top-level package that re-exports all public types from every module.
"""
from __future__ import annotations

from serenity.stagehand.budget import BudgetManager
from serenity.stagehand.config import StagehandConfig
from serenity.stagehand.errors import (
    DtypeMismatchError,
    InvalidStateTransitionError,
    StagehandError,
    StagehandOOMError,
    TransferError,
)
from serenity.stagehand.guards import NumericGuard
from serenity.stagehand.pool import PinnedPool, PinnedSlab
from serenity.stagehand.registry import BlockEntry, BlockRegistry
from serenity.stagehand.residency import BlockState, ResidencyEntry, ResidencyMap
from serenity.stagehand.scheduler import StaticLookaheadPolicy, StagehandScheduler
from serenity.stagehand.telemetry import StagehandTelemetry, StepMetrics
from serenity.stagehand.transfer import AsyncTransferEngine, TransferHandle

__all__ = [
    # config
    "StagehandConfig",
    # errors
    "DtypeMismatchError",
    "InvalidStateTransitionError",
    "StagehandError",
    "StagehandOOMError",
    "TransferError",
    # pool
    "PinnedPool",
    "PinnedSlab",
    # registry
    "BlockEntry",
    "BlockRegistry",
    # residency
    "BlockState",
    "ResidencyEntry",
    "ResidencyMap",
    # transfer
    "AsyncTransferEngine",
    "TransferHandle",
    # scheduler
    "StaticLookaheadPolicy",
    "StagehandScheduler",
    # budget
    "BudgetManager",
    # guards
    "NumericGuard",
    # telemetry
    "StagehandTelemetry",
    "StepMetrics",
]
