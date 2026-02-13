"""Stagehand-based memory strategy for block-swapping training."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
import torch.nn as nn

from serenity.memory.strategy import MemoryStrategy

# Block pattern mapping per model family.
BLOCK_PATTERNS: dict[str, str] = {
    "wan": r"^blocks\.\d+$",
    "flux": r"^transformer_blocks\.\d+$",
    "flux2": r"^transformer_blocks\.\d+$",
    "sd3": r"^transformer_blocks\.\d+$",
    "ltx2": r"^transformer_blocks\.\d+$",
    "qwen": r"^blocks\.\d+$",
    "zimage": r"^blocks\.\d+$",
}

DEFAULT_BLOCK_PATTERN = r"^blocks\.\d+$"


@dataclass
class StagehandStrategyConfig:
    """Configuration subset for StagehandStrategy."""

    family: str = "wan"
    block_pattern: str | None = None
    pinned_pool_mb: int = 8192
    pinned_slab_mb: int = 512
    vram_high_watermark_mb: int = 20000
    vram_low_watermark_mb: int = 16000
    prefetch_window_blocks: int = 2
    max_inflight_transfers: int = 2
    telemetry_enabled: bool = True
    telemetry_file: str = "stagehand_telemetry.jsonl"
    gradient_checkpointing: str = "on"
    dtype: str = "bfloat16"


class StagehandStrategy(MemoryStrategy):
    """Block-swapping memory strategy powered by StagehandRuntime.

    Replaces LayerOffloadStrategy for models too large to fit in VRAM.
    Stagehand streams transformer blocks one at a time through GPU via
    pinned host memory, keeping peak VRAM = small layers + 1 block + activations.
    """

    def __init__(self, config: StagehandStrategyConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            self._config = StagehandStrategyConfig(**{
                k: v for k, v in config.items()
                if k in StagehandStrategyConfig.__dataclass_fields__
            })
        else:
            self._config = config

        self._runtime: Any | None = None
        self._step: int = 0

    def setup(self, model: Any) -> None:
        """Build StagehandRuntime from the model's transformer.

        Parameters
        ----------
        model:
            Object with a ``transformer`` attribute (e.g. the pipeline or a
            SimpleNamespace wrapping the train module).
        """
        from serenity.stagehand import StagehandConfig, StagehandRuntime

        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise RuntimeError("StagehandStrategy.setup() requires model.transformer")

        block_pattern = self._config.block_pattern or BLOCK_PATTERNS.get(
            self._config.family, DEFAULT_BLOCK_PATTERN
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._config.dtype, torch.bfloat16)

        stagehand_config = StagehandConfig(
            stagehand_enabled=True,
            pinned_pool_mb=self._config.pinned_pool_mb,
            pinned_slab_mb=self._config.pinned_slab_mb,
            vram_high_watermark_mb=self._config.vram_high_watermark_mb,
            vram_low_watermark_mb=self._config.vram_low_watermark_mb,
            prefetch_window_blocks=self._config.prefetch_window_blocks,
            max_inflight_transfers=self._config.max_inflight_transfers,
            telemetry_enabled=self._config.telemetry_enabled,
            telemetry_file=self._config.telemetry_file,
        )

        self._runtime = StagehandRuntime(
            model=transformer,
            config=stagehand_config,
            block_pattern=block_pattern,
            group="transformer",
            dtype=dtype,
            inference_mode=False,
        )

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Context manager wrapping an entire forward + backward step.

        The training loop does forward + backward inside this context:

            with strategy.forward_context():
                loss = model(**batch)      # forward hooks fire here
                loss.backward()            # backward hooks fire here

        Stagehand's hooks stream blocks to/from GPU at the right times.
        """
        if self._runtime is None:
            yield
            return

        self._runtime.begin_step(self._step)
        with self._runtime.managed_forward():
            with self._runtime.managed_backward():
                yield
        self._runtime.end_step()
        self._step += 1

    def cleanup(self) -> None:
        """Shutdown the StagehandRuntime and release resources."""
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None

    def estimate_vram_usage(self, batch_size: int, resolution: int) -> int:
        """Estimate peak VRAM in bytes based on watermark config."""
        # With Stagehand, peak VRAM is bounded by the high watermark.
        watermark_bytes = self._config.vram_high_watermark_mb * 1024 * 1024
        # Add batch-dependent activation memory estimate.
        batch_component = max(batch_size, 1) * (resolution * resolution) * 8
        return watermark_bytes + batch_component

    @property
    def conductor(self) -> None:
        """Compatibility shim — StagehandStrategy has no conductor."""
        return None

    @property
    def runtime(self) -> Any | None:
        """Access the underlying StagehandRuntime (for telemetry, stats)."""
        return self._runtime


__all__ = [
    "StagehandStrategy",
    "StagehandStrategyConfig",
    "BLOCK_PATTERNS",
]
