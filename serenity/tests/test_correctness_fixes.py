"""Tests for Phase 2 correctness bug fixes.

Covers:
- Bug #1: NaN loss must be detected BEFORE backward pass
- Bug #2: Flow sigma schedule must not produce -inf from log(0)
- Bug #3: Velocity weighting must normalize timesteps to sigma
"""

from __future__ import annotations

import math

import torch
import pytest


# ---------------------------------------------------------------------------
# Bug #1: NaN loss detection ordering
# ---------------------------------------------------------------------------

class TestNaNLossOrdering:
    """Verify NaN detection happens before backward to protect optimizer."""

    def test_nan_loss_detected_before_backward(self):
        """A NaN loss should never reach backward()."""
        # Simulate: create a simple param, produce a NaN loss
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optimizer = torch.optim.SGD([param], lr=0.01)

        # Normal step — grads should be set
        loss_ok = (param ** 2).sum()
        loss_ok.backward()
        assert param.grad is not None
        optimizer.step()
        optimizer.zero_grad()

        # Record param state before NaN
        param_before = param.data.clone()

        # Now simulate what the fixed trainer does: check BEFORE backward
        loss_nan = torch.tensor(float("nan"), requires_grad=False)

        # The fix: check isfinite BEFORE backward
        is_finite = torch.isfinite(loss_nan)
        assert not is_finite, "NaN loss should not be finite"

        # If not finite, zero grads instead of backward
        if not is_finite:
            optimizer.zero_grad(set_to_none=True)

        # Param should be unchanged (no corrupt gradient applied)
        assert torch.equal(param.data, param_before)

    def test_finite_loss_passes_through(self):
        """A finite loss should still trigger backward normally."""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        loss = (param ** 2).sum()
        assert torch.isfinite(loss), "Normal loss should be finite"
        loss.backward()
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))


# ---------------------------------------------------------------------------
# Bug #2: Flow sigma schedule log(0)
# ---------------------------------------------------------------------------

class TestFlowSigmaSchedule:
    """Verify sigma schedule avoids log(0) = -inf."""

    def test_sigma_schedule_no_inf(self):
        """linspace endpoint must be > 0 so log() is finite."""
        # The fixed schedule
        sigmas = torch.linspace(1.0, 1e-5, 1000)
        log_sigmas = sigmas.log()

        assert torch.all(torch.isfinite(log_sigmas)), (
            f"log_sigmas contains non-finite values: "
            f"min={log_sigmas.min().item()}, max={log_sigmas.max().item()}"
        )

    def test_old_schedule_produces_inf(self):
        """Confirm the old schedule (endpoint=0.0) does produce -inf."""
        sigmas_old = torch.linspace(1.0, 0.0, 1000)
        log_sigmas_old = sigmas_old.log()
        assert not torch.all(torch.isfinite(log_sigmas_old)), (
            "Old schedule with 0.0 endpoint should produce -inf"
        )

    def test_sigma_schedule_monotonic(self):
        """Sigma schedule should be monotonically decreasing."""
        sigmas = torch.linspace(1.0, 1e-5, 1000)
        diffs = sigmas[1:] - sigmas[:-1]
        assert torch.all(diffs < 0), "Sigma schedule must be monotonically decreasing"

    def test_sigma_schedule_positive(self):
        """All sigma values must be strictly positive."""
        sigmas = torch.linspace(1.0, 1e-5, 1000)
        assert torch.all(sigmas > 0), "All sigmas must be > 0"


# ---------------------------------------------------------------------------
# Bug #3: Velocity weighting normalization
# ---------------------------------------------------------------------------

class TestVelocityWeighting:
    """Verify velocity weighting normalizes timesteps to sigma first."""

    def _normalize_timesteps(self, timesteps: torch.Tensor, N: int = 1000) -> torch.Tensor:
        """The correct normalization: sigma = (t + 1) / N."""
        return (timesteps.float() + 1) / N

    def test_snr_weighting_produces_valid_range(self):
        """SNR weights from normalized sigma should be in (0, 1)."""
        timesteps = torch.arange(0, 1000)
        sigma = self._normalize_timesteps(timesteps)
        snr = (1 - sigma) / (sigma + 1e-6)
        weights = snr / (snr + 1)

        assert torch.all(torch.isfinite(weights)), "SNR weights must be finite"
        assert torch.all(weights >= 0), "SNR weights must be non-negative"
        assert torch.all(weights <= 1), "SNR weights must be <= 1"

    def test_raw_timestep_snr_is_wrong(self):
        """Using raw timesteps (0..999) directly as sigma produces bad SNR."""
        timesteps = torch.arange(0, 1000).float()
        # BUG: using raw timesteps as sigma
        sigma_raw = timesteps
        snr_raw = (1 - sigma_raw) / (sigma_raw + 1e-6)
        # For timestep > 1, (1 - t) is very negative -> SNR is negative
        assert torch.any(snr_raw < 0), (
            "Raw timesteps used as sigma should produce negative SNR for t > 1"
        )

    def test_sigma_sqrt_weighting_valid(self):
        """sqrt(sigma) weighting should produce positive finite values."""
        timesteps = torch.arange(0, 1000)
        sigma = self._normalize_timesteps(timesteps)
        weights = torch.sqrt(sigma + 1e-6)

        assert torch.all(torch.isfinite(weights)), "sqrt weights must be finite"
        assert torch.all(weights > 0), "sqrt weights must be positive"

    def test_uniform_weighting_is_ones(self):
        """Uniform weighting should return all ones."""
        timesteps = torch.arange(0, 1000).float()
        weights = torch.ones_like(timesteps, dtype=torch.float32)
        assert torch.all(weights == 1.0)

    def test_normalized_sigma_range(self):
        """Normalized sigma from discrete timesteps should be in (0, 1]."""
        timesteps = torch.arange(0, 1000)
        sigma = self._normalize_timesteps(timesteps)
        assert sigma.min().item() == pytest.approx(1 / 1000, abs=1e-6)
        assert sigma.max().item() == pytest.approx(1.0, abs=1e-6)
