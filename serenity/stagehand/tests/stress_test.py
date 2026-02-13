"""Stagehand long-running stability stress test.

Runnable as::

    python -m serenity.stagehand.tests.stress_test --steps 100 --blocks 20

Builds a synthetic model with N linear blocks, runs M simulated forward
passes through the Stagehand scheduler, and tracks RSS, pool stats, and
telemetry over time.  Exits 0 on pass, 1 on fail.

Tests run WITHOUT a GPU — CUDA is not required.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from serenity.stagehand.budget import BudgetManager
from serenity.stagehand.config import StagehandConfig
from serenity.stagehand.guards import NumericGuard
from serenity.stagehand.pool import PinnedPool
from serenity.stagehand.registry import BlockRegistry
from serenity.stagehand.residency import ResidencyMap
from serenity.stagehand.scheduler import StaticLookaheadPolicy, StagehandScheduler
from serenity.stagehand.telemetry import StagehandTelemetry
from serenity.stagehand.transfer import AsyncTransferEngine

__all__: list[str] = []

# Try to import StagehandRuntime — may not exist yet.
_HAS_RUNTIME = False
try:
    from serenity.stagehand import StagehandRuntime  # type: ignore[attr-defined]
    _HAS_RUNTIME = True
except (ImportError, AttributeError):
    pass


# ── data structures ──────────────────────────────────────────────────────


@dataclass
class StepSnapshot:
    """Per-step telemetry snapshot."""
    step: int
    rss_mb: float
    pool_free: int
    pool_in_use: int
    pool_peak_in_use: int
    prefetch_hits: int
    prefetch_misses: int
    stall_time_ms: float
    evictions: int


@dataclass
class StressResults:
    """Aggregated stress test results."""
    steps: int = 0
    blocks: int = 0
    block_size: int = 0
    rss_start_mb: float = 0.0
    rss_end_mb: float = 0.0
    rss_delta_mb: float = 0.0
    rss_peak_mb: float = 0.0
    pool_peak_in_use: int = 0
    hit_rate_pct: float = 0.0
    mean_stall_ms: float = 0.0
    max_stall_ms: float = 0.0
    stall_trend: str = "stable"
    total_time_s: float = 0.0
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    snapshots: list[StepSnapshot] = field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────────


def _get_rss_mb() -> float:
    """Current RSS in MB via getrusage (platform-independent)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux, ru_maxrss is in KB.  On macOS, it's in bytes.
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _get_rss_current_mb() -> float:
    """Try to read current RSS from /proc/self/status (Linux) or fall back."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except (OSError, ValueError):
        pass
    return _get_rss_mb()


class _NeverAboveBudget:
    """Budget mock that never triggers eviction."""
    def above_high_watermark(self) -> bool:
        return False

    def below_low_watermark(self) -> bool:
        return True


class _BlockContainer(nn.Module):
    """Wrapper that holds blocks in a ModuleList named ``block``.

    ``named_modules()`` yields ``block.0``, ``block.1``, etc.
    """

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.block = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block:
            x = b(x)
        return x


def _make_model(num_blocks: int, hidden: int) -> _BlockContainer:
    """Build a synthetic model with N linear blocks."""
    blocks = [nn.Linear(hidden, hidden, bias=False) for _ in range(num_blocks)]
    return _BlockContainer(blocks)


def _build_components(
    model: nn.Module,
    num_blocks: int,
    pool_mb: int,
    slab_mb: int,
    prefetch_window: int,
) -> dict:
    """Build the full Stagehand component stack."""
    registry = BlockRegistry()
    registry.build_from_model(
        model,
        block_pattern=r"block\.\d+",
        group="stress",
        dtype=torch.bfloat16,
    )
    registry.validate(pool_capacity_bytes=pool_mb * 1024 * 1024)

    residency = ResidencyMap(registry)
    pool = PinnedPool(total_mb=pool_mb, slab_mb=slab_mb)
    engine = AsyncTransferEngine(pool=pool, max_inflight=4)

    budget = _NeverAboveBudget()
    policy = StaticLookaheadPolicy(prefetch_window=prefetch_window)

    guards = NumericGuard(
        strict_bf16=False,
        fail_on_dtype_promotion=False,
        nan_inf_check=False,  # Skip for speed in stress test.
    )

    config = StagehandConfig(
        pinned_pool_mb=pool_mb,
        pinned_slab_mb=slab_mb,
        nan_inf_check=False,
    )
    telemetry = StagehandTelemetry(enabled=True, interval_steps=max(num_blocks, 10))

    scheduler = StagehandScheduler(
        registry=registry,
        residency=residency,
        transfer_engine=engine,
        budget=budget,
        policy=policy,
        guards=guards,
        telemetry=telemetry,
        config=config,
    )

    return {
        "model": model,
        "registry": registry,
        "residency": residency,
        "pool": pool,
        "engine": engine,
        "budget": budget,
        "policy": policy,
        "guards": guards,
        "config": config,
        "telemetry": telemetry,
        "scheduler": scheduler,
    }


def _run_step(stack: dict, step: int) -> None:
    """Simulate one forward pass through all blocks."""
    scheduler = stack["scheduler"]
    registry = stack["registry"]
    ordered = registry.blocks_in_order()

    scheduler.begin_step(step)
    for entry in ordered:
        scheduler.before_block(entry.block_id)
        # Simulate compute output (small tensor for speed).
        output = torch.randn(64, dtype=torch.float32)
        scheduler.after_block(entry.block_id, output)
    scheduler.end_step()


# ── main test loop ───────────────────────────────────────────────────────


def run_stress_test(args: argparse.Namespace) -> StressResults:
    """Execute the stress test and return results."""
    results = StressResults(
        steps=args.steps,
        blocks=args.blocks,
        block_size=args.block_size,
    )

    print(f"Stagehand Stress Test")
    print(f"  Steps:      {args.steps}")
    print(f"  Blocks:     {args.blocks}")
    print(f"  Block size: {args.block_size}")
    print(f"  Pool:       {args.pool_mb} MB ({args.slab_mb} MB slabs)")
    print(f"  Runtime:    {'StagehandRuntime' if _HAS_RUNTIME else 'direct components'}")
    print()

    # Build model and components.
    model = _make_model(args.blocks, args.block_size)
    stack = _build_components(
        model=model,
        num_blocks=args.blocks,
        pool_mb=args.pool_mb,
        slab_mb=args.slab_mb,
        prefetch_window=3,
    )

    pool = stack["pool"]
    telemetry = stack["telemetry"]

    # Record initial RSS.
    results.rss_start_mb = _get_rss_current_mb()
    rss_peak = results.rss_start_mb

    t0 = time.monotonic()

    # Main loop.
    for step in range(args.steps):
        _run_step(stack, step)

        # Snapshot.
        current_rss = _get_rss_current_mb()
        rss_peak = max(rss_peak, current_rss)
        pool_stats = pool.stats()

        snap = StepSnapshot(
            step=step,
            rss_mb=current_rss,
            pool_free=pool_stats["free"],
            pool_in_use=pool_stats["in_use"],
            pool_peak_in_use=pool_stats["peak_in_use"],
            prefetch_hits=telemetry._history[-1].prefetch_hits if telemetry._history else 0,
            prefetch_misses=telemetry._history[-1].prefetch_misses if telemetry._history else 0,
            stall_time_ms=telemetry._history[-1].stall_time_ms if telemetry._history else 0.0,
            evictions=telemetry._history[-1].evictions if telemetry._history else 0,
        )
        results.snapshots.append(snap)

        # Progress indicator every 10% of steps.
        if args.steps >= 10 and (step + 1) % max(1, args.steps // 10) == 0:
            pct = (step + 1) / args.steps * 100
            print(
                f"  [{pct:5.1f}%] step={step + 1:>5} "
                f"rss={current_rss:.0f}MB "
                f"pool={pool_stats['in_use']}/{pool_stats['total']} "
                f"hit_rate={telemetry.hit_rate() * 100:.1f}%"
            )

    elapsed = time.monotonic() - t0

    # Compute final metrics.
    results.rss_end_mb = _get_rss_current_mb()
    results.rss_delta_mb = results.rss_end_mb - results.rss_start_mb
    results.rss_peak_mb = rss_peak
    results.pool_peak_in_use = pool.stats()["peak_in_use"]
    results.hit_rate_pct = telemetry.hit_rate() * 100.0
    results.mean_stall_ms = telemetry.mean_stall_ms()
    results.max_stall_ms = telemetry.max_stall_ms()
    results.stall_trend = _compute_stall_trend(results.snapshots)
    results.total_time_s = elapsed

    # Acceptance checks.
    rss_threshold_mb = 100.0

    if results.rss_delta_mb > rss_threshold_mb:
        results.failure_reasons.append(
            f"RSS delta {results.rss_delta_mb:.1f} MB exceeds {rss_threshold_mb:.0f} MB threshold"
        )

    results.passed = len(results.failure_reasons) == 0

    # Cleanup.
    telemetry.close()
    pool.shutdown()

    return results


def _compute_stall_trend(snapshots: list[StepSnapshot]) -> str:
    """Determine if stall times are growing, stable, or shrinking."""
    if len(snapshots) < 6:
        return "stable"

    n = len(snapshots)
    mid = n // 2
    first_half = [s.stall_time_ms for s in snapshots[:mid]]
    second_half = [s.stall_time_ms for s in snapshots[mid:]]

    first_avg = sum(first_half) / len(first_half) if first_half else 0.0
    second_avg = sum(second_half) / len(second_half) if second_half else 0.0

    delta = second_avg - first_avg
    if delta > 1.0:
        return "growing"
    elif delta < -1.0:
        return "shrinking"
    return "stable"


def _print_results(results: StressResults) -> None:
    """Print the final summary."""
    print()
    print("=" * 60)
    print("Stagehand Stress Test Results:")
    print(f"  Steps:             {results.steps}")
    print(f"  Blocks:            {results.blocks}")
    print(f"  RSS start:         {results.rss_start_mb:.0f} MB")
    print(f"  RSS end:           {results.rss_end_mb:.0f} MB")
    rss_sign = "+" if results.rss_delta_mb >= 0 else ""
    rss_status = "PASS" if results.rss_delta_mb < 100 else "FAIL"
    print(f"  RSS delta:         {rss_sign}{results.rss_delta_mb:.0f} MB ({rss_status}: < 100MB threshold)")
    print(f"  RSS peak:          {results.rss_peak_mb:.0f} MB")
    print(f"  Peak pool in-use:  {results.pool_peak_in_use} slabs")
    print(f"  Prefetch hit rate: {results.hit_rate_pct:.1f}%")
    print(f"  Mean stall:        {results.mean_stall_ms:.1f}ms")
    print(f"  Max stall:         {results.max_stall_ms:.1f}ms")
    print(f"  Stall trend:       {results.stall_trend}")
    print(f"  Total time:        {results.total_time_s:.1f}s")
    print()
    if results.passed:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")
        for reason in results.failure_reasons:
            print(f"    - {reason}")
    print("=" * 60)


def _write_json_output(results: StressResults, path: str) -> None:
    """Write results to a JSON file."""
    data = {
        "steps": results.steps,
        "blocks": results.blocks,
        "block_size": results.block_size,
        "rss_start_mb": results.rss_start_mb,
        "rss_end_mb": results.rss_end_mb,
        "rss_delta_mb": results.rss_delta_mb,
        "rss_peak_mb": results.rss_peak_mb,
        "pool_peak_in_use": results.pool_peak_in_use,
        "hit_rate_pct": results.hit_rate_pct,
        "mean_stall_ms": results.mean_stall_ms,
        "max_stall_ms": results.max_stall_ms,
        "stall_trend": results.stall_trend,
        "total_time_s": results.total_time_s,
        "passed": results.passed,
        "failure_reasons": results.failure_reasons,
        "snapshots": [
            {
                "step": s.step,
                "rss_mb": s.rss_mb,
                "pool_free": s.pool_free,
                "pool_in_use": s.pool_in_use,
                "pool_peak_in_use": s.pool_peak_in_use,
                "prefetch_hits": s.prefetch_hits,
                "prefetch_misses": s.prefetch_misses,
                "stall_time_ms": s.stall_time_ms,
                "evictions": s.evictions,
            }
            for s in results.snapshots
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults written to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stagehand long-running stability stress test.",
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of simulated forward steps (default: 100).",
    )
    parser.add_argument(
        "--blocks", type=int, default=20,
        help="Number of linear blocks in the synthetic model (default: 20).",
    )
    parser.add_argument(
        "--block-size", type=int, default=256,
        help="Hidden dimension of each linear block (default: 256).",
    )
    parser.add_argument(
        "--pool-mb", type=int, default=256,
        help="Total pinned pool size in MB (default: 256).",
    )
    parser.add_argument(
        "--slab-mb", type=int, default=64,
        help="Individual slab size in MB (default: 64).",
    )
    parser.add_argument(
        "--monitor_rss", action="store_true", default=True,
        help="Monitor RSS memory usage (default: True).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path for JSON output file.",
    )

    args = parser.parse_args()

    # Validate pool configuration.
    if args.pool_mb % args.slab_mb != 0:
        print(
            f"ERROR: pool-mb ({args.pool_mb}) must be evenly divisible by "
            f"slab-mb ({args.slab_mb})",
            file=sys.stderr,
        )
        sys.exit(1)

    results = run_stress_test(args)
    _print_results(results)

    if args.output:
        _write_json_output(results, args.output)

    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
