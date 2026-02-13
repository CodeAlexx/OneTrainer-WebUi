"""Phase 5 acceptance tests — StagehandTelemetry."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from serenity.stagehand.telemetry import StagehandTelemetry, StepMetrics


# ── helpers ──────────────────────────────────────────────────────────────


def _run_steps(
    telemetry: StagehandTelemetry,
    num_steps: int,
    *,
    stall_ms: float = 0.0,
    hits: int = 5,
    misses: int = 1,
    evictions: int = 0,
    vram_used_mb: float = 20000.0,
    nan_count: int = 0,
    inf_count: int = 0,
) -> None:
    """Record *num_steps* of telemetry with fixed metrics per step."""
    for i in range(num_steps):
        telemetry.begin_step(i)
        telemetry.record_h2d(256 * 1024 * 1024)
        telemetry.record_d2h(64 * 1024 * 1024)
        if stall_ms > 0:
            telemetry.record_stall(stall_ms)
        for _ in range(hits):
            telemetry.record_prefetch_hit()
        for _ in range(misses):
            telemetry.record_prefetch_miss()
        for _ in range(evictions):
            telemetry.record_eviction()
        telemetry.record_vram(vram_used_mb, vram_used_mb + 500.0)
        telemetry.record_pool_stats(free=6, in_use=2)
        if nan_count > 0 or inf_count > 0:
            telemetry.record_nan_inf(nan_count, inf_count)
        telemetry.end_step()


# ── tests ────────────────────────────────────────────────────────────────


class TestStepMetricsDefaults:
    def test_defaults(self) -> None:
        m = StepMetrics()
        assert m.step == 0
        assert m.h2d_bytes == 0
        assert m.stall_time_ms == 0.0
        assert m.nan_count == 0
        assert m.inf_count == 0

    def test_custom_values(self) -> None:
        m = StepMetrics(step=42, h2d_bytes=1024, stall_time_ms=5.0)
        assert m.step == 42
        assert m.h2d_bytes == 1024
        assert m.stall_time_ms == 5.0


class TestTelemetryJSONL:
    def test_jsonl_output_10_steps(self) -> None:
        """Record 10 steps and verify JSONL file has 10 valid lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            telemetry = StagehandTelemetry(
                enabled=True, interval_steps=5, output_file=path
            )
            _run_steps(telemetry, 10)
            telemetry.close()

            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]

            assert len(lines) == 10

            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["step"] == i
                assert "h2d_bytes" in data
                assert "d2h_bytes" in data
                assert "stall_time_ms" in data
                assert "stall_count" in data
                assert "evictions" in data
                assert "prefetch_hits" in data
                assert "prefetch_misses" in data
                assert "vram_used_mb" in data
                assert "vram_reserved_mb" in data
                assert "host_pool_free" in data
                assert "host_pool_in_use" in data
                assert "dtype_promotions" in data
                assert "nan_count" in data
                assert "inf_count" in data
        finally:
            os.unlink(path)

    def test_jsonl_values_correct(self) -> None:
        """Verify JSONL values match what was recorded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            telemetry = StagehandTelemetry(
                enabled=True, interval_steps=100, output_file=path
            )
            _run_steps(telemetry, 1, hits=8, misses=2, evictions=3, stall_ms=2.5)
            telemetry.close()

            with open(path) as f:
                data = json.loads(f.readline())

            assert data["step"] == 0
            assert data["h2d_bytes"] == 256 * 1024 * 1024
            assert data["d2h_bytes"] == 64 * 1024 * 1024
            assert data["prefetch_hits"] == 8
            assert data["prefetch_misses"] == 2
            assert data["evictions"] == 3
            assert abs(data["stall_time_ms"] - 2.5) < 1e-6
            assert data["stall_count"] == 1
            assert data["host_pool_free"] == 6
            assert data["host_pool_in_use"] == 2
        finally:
            os.unlink(path)


class TestHitRate:
    def test_hit_rate_calculation(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 5, hits=8, misses=2)
        # hit_rate = total_hits / (total_hits + total_misses)
        # 5 steps * 8 hits = 40, 5 steps * 2 misses = 10
        expected = 40.0 / 50.0
        assert abs(telemetry.hit_rate() - expected) < 1e-6

    def test_hit_rate_zero_events(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        assert telemetry.hit_rate() == 0.0

    def test_hit_rate_all_hits(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 3, hits=10, misses=0)
        assert telemetry.hit_rate() == 1.0

    def test_hit_rate_all_misses(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 3, hits=0, misses=10)
        assert telemetry.hit_rate() == 0.0


class TestStallMetrics:
    def test_mean_stall_ms(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 4, stall_ms=10.0)
        assert abs(telemetry.mean_stall_ms() - 10.0) < 1e-6

    def test_max_stall_ms(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        # Record varying stalls manually.
        for i, stall in enumerate([1.0, 5.0, 3.0, 10.0, 2.0]):
            telemetry.begin_step(i)
            telemetry.record_stall(stall)
            telemetry.end_step()
        assert telemetry.max_stall_ms() == 10.0

    def test_mean_stall_no_history(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        assert telemetry.mean_stall_ms() == 0.0

    def test_max_stall_no_history(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        assert telemetry.max_stall_ms() == 0.0


class TestLogLineFormat:
    def test_format_matches_spec(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 1, hits=5, misses=1, vram_used_mb=21913.6)

        metrics = telemetry._history[0]
        line = telemetry._format_log_line(metrics)

        assert line.startswith("[STAGEHAND step=0]")
        assert "hit=" in line
        assert "stall=" in line
        assert "evict=" in line
        assert "vram=" in line
        assert "pool=" in line

    def test_format_specific_values(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)

        telemetry.begin_step(42)
        for _ in range(95):
            telemetry.record_prefetch_hit()
        for _ in range(5):
            telemetry.record_prefetch_miss()
        telemetry.record_stall(1.3)
        telemetry.record_eviction()
        telemetry.record_eviction()
        telemetry.record_vram(21913.6, 22500.0)
        telemetry.record_pool_stats(free=6, in_use=2)
        telemetry.end_step()

        metrics = telemetry._history[0]
        line = telemetry._format_log_line(metrics)

        # Expected format: [STAGEHAND step=42] hit=95.0% stall=1.3ms evict=2 vram=21.4G pool=6/8
        assert "step=42" in line
        assert "hit=95.0%" in line
        assert "stall=1.3ms" in line
        assert "evict=2" in line
        assert "pool=6/8" in line


class TestNanInfRecording:
    def test_nan_inf_recorded(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 3, nan_count=2, inf_count=1)

        for m in telemetry._history:
            assert m.nan_count == 2
            assert m.inf_count == 1

    def test_nan_inf_in_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            telemetry = StagehandTelemetry(
                enabled=True, interval_steps=100, output_file=path
            )
            _run_steps(telemetry, 1, nan_count=5, inf_count=3)
            telemetry.close()

            with open(path) as f:
                data = json.loads(f.readline())
            assert data["nan_count"] == 5
            assert data["inf_count"] == 3
        finally:
            os.unlink(path)


class TestVRAMTrend:
    def test_stable(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 10, vram_used_mb=20000.0)
        assert telemetry.vram_trend() == "stable"

    def test_growing(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        for i in range(20):
            telemetry.begin_step(i)
            telemetry.record_vram(20000.0 + i * 100.0, 22000.0)
            telemetry.end_step()
        assert telemetry.vram_trend() == "growing"

    def test_shrinking(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        for i in range(20):
            telemetry.begin_step(i)
            telemetry.record_vram(22000.0 - i * 100.0, 22000.0)
            telemetry.end_step()
        assert telemetry.vram_trend() == "shrinking"

    def test_not_enough_data(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 2)
        assert telemetry.vram_trend() == "stable"


class TestDisabledTelemetry:
    def test_disabled_no_ops(self) -> None:
        telemetry = StagehandTelemetry(enabled=False)
        telemetry.begin_step(0)
        telemetry.record_h2d(1024)
        telemetry.record_stall(5.0)
        telemetry.end_step()
        assert len(telemetry._history) == 0

    def test_disabled_no_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            telemetry = StagehandTelemetry(
                enabled=False, output_file=path
            )
            telemetry.begin_step(0)
            telemetry.end_step()
            telemetry.close()

            # File should exist but be empty (disabled telemetry never writes).
            with open(path) as f:
                content = f.read()
            assert content == ""
        finally:
            os.unlink(path)


class TestRollingWindow:
    def test_maxlen_100(self) -> None:
        telemetry = StagehandTelemetry(enabled=True)
        _run_steps(telemetry, 150)
        # Rolling window maxlen=100.
        assert len(telemetry._history) == 100
        # Oldest step should be 50 (150 - 100).
        assert telemetry._history[0].step == 50
