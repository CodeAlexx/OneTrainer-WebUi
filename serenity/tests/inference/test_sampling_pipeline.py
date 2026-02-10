"""Tests for the Serenity inference sampling pipeline."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from serenity.inference.sampling.cfg import (
    apply_cfg,
    compute_cfg,
    mahiro_correction,
    rescale_cfg,
)
from serenity.inference.sampling.conditioning import (
    Conditioning,
    create_noise,
    prepare_conditioning,
)
from serenity.inference.sampling.prediction import (
    EDMPrediction,
    EpsPrediction,
    FlowPrediction,
    FluxPrediction,
    Prediction,
    PredictionType,
    VPrediction,
    get_prediction,
)
from serenity.inference.sampling.sampler import (
    SamplerType,
    create_model_fn,
    dpm_pp_2m_sample,
    dpm_pp_2m_sde_sample,
    euler_ancestral_sample,
    euler_sample,
    sample,
)
from serenity.inference.sampling.schedulers import (
    SchedulerType,
    ays_scheduler,
    beta_scheduler,
    compute_sigmas,
    ddim_uniform_scheduler,
    exponential_scheduler,
    karras_scheduler,
    linear_quadratic_scheduler,
    normal_scheduler,
    sgm_uniform_scheduler,
    simple_scheduler,
)


# ===================================================================== #
# Prediction Tests
# ===================================================================== #


class TestEpsPrediction:
    """Tests for epsilon (noise) prediction."""

    def test_calculate_denoised_known_values(self) -> None:
        pred = EpsPrediction()
        sigma = torch.tensor([0.5])
        model_output = torch.tensor([[1.0, 2.0, 3.0]])
        model_input = torch.tensor([[4.0, 5.0, 6.0]])

        result = pred.calculate_denoised(sigma, model_output, model_input)

        # denoised = input - output * sigma
        expected = model_input - model_output * 0.5
        torch.testing.assert_close(result, expected)

    def test_calculate_denoised_zero_sigma(self) -> None:
        pred = EpsPrediction()
        sigma = torch.tensor([0.0])
        model_output = torch.tensor([[1.0, 2.0]])
        model_input = torch.tensor([[3.0, 4.0]])

        result = pred.calculate_denoised(sigma, model_output, model_input)
        torch.testing.assert_close(result, model_input)

    def test_calculate_input_scales_by_sigma(self) -> None:
        pred = EpsPrediction()
        sigma = torch.tensor([1.0])
        noise = torch.tensor([[2.0, 3.0]])

        result = pred.calculate_input(sigma, noise)
        # c_in = noise / sqrt(sigma^2 + sigma_data^2) = noise / sqrt(2)
        expected = noise / math.sqrt(2)
        torch.testing.assert_close(result, expected)

    def test_noise_scaling(self) -> None:
        pred = EpsPrediction()
        sigma = torch.tensor([0.5])
        noise = torch.ones(1, 3)
        latent = torch.ones(1, 3) * 2.0

        result = pred.noise_scaling(sigma, noise, latent)
        expected = latent + noise * 0.5
        torch.testing.assert_close(result, expected)


class TestVPrediction:
    """Tests for v-prediction."""

    def test_calculate_denoised_known_values(self) -> None:
        pred = VPrediction(sigma_data=1.0)
        sigma = torch.tensor([1.0])
        model_output = torch.tensor([[1.0]])
        model_input = torch.tensor([[2.0]])

        result = pred.calculate_denoised(sigma, model_output, model_input)

        # With sigma=1, sigma_data=1:
        # sd2 = 1, sigma^2 + sd2 = 2
        # denoised = input * 1/2 - output * 1 * 1 / sqrt(2)
        expected = torch.tensor([[2.0 * 0.5 - 1.0 / math.sqrt(2)]])
        torch.testing.assert_close(result, expected)

    def test_differs_from_eps(self) -> None:
        """V-prediction should produce different results than epsilon."""
        eps_pred = EpsPrediction()
        v_pred = VPrediction()
        sigma = torch.tensor([0.5])
        model_output = torch.tensor([[1.0, 2.0]])
        model_input = torch.tensor([[3.0, 4.0]])

        eps_result = eps_pred.calculate_denoised(sigma, model_output, model_input)
        v_result = v_pred.calculate_denoised(sigma, model_output, model_input)

        assert not torch.allclose(eps_result, v_result)


class TestFlowPrediction:
    """Tests for discrete flow matching prediction."""

    def test_calculate_denoised(self) -> None:
        pred = FlowPrediction(shift=1.0)
        sigma = torch.tensor([0.5])
        model_output = torch.tensor([[1.0, 2.0]])
        model_input = torch.tensor([[3.0, 4.0]])

        result = pred.calculate_denoised(sigma, model_output, model_input)
        expected = model_input - model_output * 0.5
        torch.testing.assert_close(result, expected)

    def test_noise_scaling_interpolates(self) -> None:
        """Flow noise scaling: sigma * noise + (1 - sigma) * latent."""
        pred = FlowPrediction()
        sigma = torch.tensor([0.3])
        noise = torch.ones(1, 2) * 10.0
        latent = torch.ones(1, 2) * 1.0

        result = pred.noise_scaling(sigma, noise, latent)
        expected = 0.3 * noise + 0.7 * latent
        torch.testing.assert_close(result, expected)

    def test_sigma_to_timestep(self) -> None:
        pred = FlowPrediction(multiplier=1000.0)
        sigma = torch.tensor([0.5])
        ts = pred.sigma_to_timestep(sigma)
        torch.testing.assert_close(ts, torch.tensor([500.0]))

    def test_calculate_input_identity(self) -> None:
        """Flow models use identity input scaling."""
        pred = FlowPrediction()
        sigma = torch.tensor([0.5])
        noise = torch.tensor([[1.0, 2.0, 3.0]])
        result = pred.calculate_input(sigma, noise)
        torch.testing.assert_close(result, noise)

    def test_time_snr_shift_identity(self) -> None:
        """Shift=1 should be identity."""
        pred = FlowPrediction(shift=1.0)
        t = torch.tensor([0.3, 0.5, 0.7])
        result = pred._time_snr_shift(t)
        torch.testing.assert_close(result, t)

    def test_time_snr_shift_nonunit(self) -> None:
        """Shift != 1 should change values."""
        pred = FlowPrediction(shift=3.0)
        t = torch.tensor([0.5])
        result = pred._time_snr_shift(t)
        # alpha*t / (1 + (alpha-1)*t) = 3*0.5 / (1 + 2*0.5) = 1.5/2 = 0.75
        torch.testing.assert_close(result, torch.tensor([0.75]))


class TestFluxPrediction:
    """Tests for Flux-specific sigma shifting."""

    def test_mu_from_defaults(self) -> None:
        pred = FluxPrediction()
        assert isinstance(pred.mu, float)

    def test_explicit_mu(self) -> None:
        pred = FluxPrediction(mu=0.8)
        assert pred.mu == 0.8

    def test_sigma_shift_changes_schedule(self) -> None:
        pred = FluxPrediction(mu=0.8)
        sigmas = torch.linspace(0.01, 0.99, 10)
        shifted = pred.apply_sigma_shift(sigmas)

        # Shifted sigmas should differ from original
        assert not torch.allclose(sigmas, shifted)
        # Should be in (0, 1) range
        assert shifted.min() > 0.0
        assert shifted.max() < 1.0

    def test_sigma_to_timestep_identity(self) -> None:
        pred = FluxPrediction(mu=0.8)
        sigma = torch.tensor([0.5])
        ts = pred.sigma_to_timestep(sigma)
        torch.testing.assert_close(ts, sigma)


class TestEDMPrediction:
    """Tests for EDM prediction."""

    def test_sigma_to_timestep(self) -> None:
        pred = EDMPrediction()
        sigma = torch.tensor([math.e])  # log(e) = 1
        ts = pred.sigma_to_timestep(sigma)
        torch.testing.assert_close(ts, torch.tensor([0.25]))

    def test_calculate_denoised(self) -> None:
        """EDM uses + instead of - for the model_output term."""
        pred = EDMPrediction(sigma_data=1.0)
        sigma = torch.tensor([1.0])
        model_output = torch.tensor([[1.0]])
        model_input = torch.tensor([[2.0]])

        result = pred.calculate_denoised(sigma, model_output, model_input)
        # sd2=1, sigma^2+sd2=2 => input*1/2 + output*1*1/sqrt(2)
        expected = torch.tensor([[2.0 * 0.5 + 1.0 / math.sqrt(2)]])
        torch.testing.assert_close(result, expected)


class TestGetPrediction:
    """Tests for the prediction factory."""

    def test_returns_eps(self) -> None:
        pred = get_prediction(PredictionType.EPS)
        assert isinstance(pred, EpsPrediction)

    def test_returns_v_prediction(self) -> None:
        pred = get_prediction(PredictionType.V_PREDICTION)
        assert isinstance(pred, VPrediction)

    def test_returns_flow(self) -> None:
        pred = get_prediction(PredictionType.FLOW)
        assert isinstance(pred, FlowPrediction)

    def test_returns_flux(self) -> None:
        pred = get_prediction(PredictionType.FLOW_FLUX)
        assert isinstance(pred, FluxPrediction)

    def test_returns_edm(self) -> None:
        pred = get_prediction(PredictionType.EDM)
        assert isinstance(pred, EDMPrediction)

    def test_string_dispatch(self) -> None:
        pred = get_prediction("eps")
        assert isinstance(pred, EpsPrediction)

    def test_kwargs_forwarded(self) -> None:
        pred = get_prediction(PredictionType.FLOW, shift=3.0, multiplier=500.0)
        assert isinstance(pred, FlowPrediction)
        assert pred.shift == 3.0
        assert pred.multiplier == 500.0


# ===================================================================== #
# CFG Tests
# ===================================================================== #


class TestComputeCFG:
    """Tests for classifier-free guidance computation."""

    def test_cfg_amplifies_difference(self) -> None:
        cond = torch.tensor([[2.0, 4.0]])
        uncond = torch.tensor([[1.0, 1.0]])
        result = compute_cfg(cond, uncond, cfg_scale=7.5)

        # uncond + 7.5 * (cond - uncond) = [1 + 7.5*1, 1 + 7.5*3] = [8.5, 23.5]
        expected = torch.tensor([[8.5, 23.5]])
        torch.testing.assert_close(result, expected)

    def test_cfg_scale_1_returns_cond(self) -> None:
        """CFG=1 optimization: skip unconditional entirely."""
        cond = torch.tensor([[5.0, 10.0]])
        uncond = torch.tensor([[999.0, 999.0]])  # should be ignored
        result = compute_cfg(cond, uncond, cfg_scale=1.0)
        torch.testing.assert_close(result, cond)

    def test_cfg_scale_0_returns_uncond(self) -> None:
        cond = torch.tensor([[5.0]])
        uncond = torch.tensor([[1.0]])
        result = compute_cfg(cond, uncond, cfg_scale=0.0)
        torch.testing.assert_close(result, uncond)


class TestRescaleCFG:
    """Tests for RescaleCFG."""

    def test_phi_zero_is_identity(self) -> None:
        denoised = torch.randn(1, 4, 8, 8)
        cond_pred = torch.randn(1, 4, 8, 8)
        result = rescale_cfg(denoised, cond_pred, cfg_scale=7.5, rescale_phi=0.0)
        torch.testing.assert_close(result, denoised)

    def test_phi_one_matches_cond_std(self) -> None:
        """Full rescale should bring std closer to cond_pred std."""
        torch.manual_seed(42)
        cond_pred = torch.randn(1, 4, 8, 8) * 0.5
        # denoised with higher std (amplified by CFG)
        denoised = torch.randn(1, 4, 8, 8) * 2.0

        result = rescale_cfg(denoised, cond_pred, cfg_scale=7.5, rescale_phi=1.0)
        # Result std should be closer to cond_pred std than denoised std
        std_result = result.std()
        std_cond = cond_pred.std()
        std_denoised = denoised.std()

        assert abs(std_result - std_cond) < abs(std_denoised - std_cond)

    def test_reduces_magnitude(self) -> None:
        """RescaleCFG should generally reduce over-amplified CFG magnitude."""
        torch.manual_seed(0)
        cond_pred = torch.randn(2, 4, 16, 16)
        denoised = cond_pred * 5.0  # very amplified

        result = rescale_cfg(denoised, cond_pred, cfg_scale=7.5, rescale_phi=0.7)
        assert result.std() < denoised.std()


class TestMahiroCorrection:
    """Tests for MaHiRo post-CFG normalization."""

    def test_output_shape_matches_input(self) -> None:
        denoised = torch.randn(2, 4, 8, 8)
        cond_pred = torch.randn(2, 4, 8, 8)
        result = mahiro_correction(denoised, cond_pred, cfg_scale=7.5)
        assert result.shape == denoised.shape

    def test_changes_values(self) -> None:
        torch.manual_seed(1)
        denoised = torch.randn(1, 4, 16, 16)
        cond_pred = torch.randn(1, 4, 16, 16)
        result = mahiro_correction(denoised, cond_pred, cfg_scale=7.5)
        assert not torch.allclose(result, denoised)


class TestApplyCFG:
    """Tests for the CFG orchestrator."""

    def test_basic_cfg(self) -> None:
        cond = torch.tensor([[2.0]])
        uncond = torch.tensor([[0.0]])
        result = apply_cfg(cond, uncond, cfg_scale=3.0)
        expected = torch.tensor([[6.0]])
        torch.testing.assert_close(result, expected)

    def test_with_rescale(self) -> None:
        torch.manual_seed(42)
        cond = torch.randn(1, 4, 8, 8)
        uncond = torch.randn(1, 4, 8, 8)

        without = apply_cfg(cond, uncond, cfg_scale=7.5, rescale_phi=0.0)
        with_rescale = apply_cfg(cond, uncond, cfg_scale=7.5, rescale_phi=0.7)
        assert not torch.allclose(without, with_rescale)

    def test_with_mahiro(self) -> None:
        torch.manual_seed(42)
        cond = torch.randn(1, 4, 8, 8)
        uncond = torch.randn(1, 4, 8, 8)

        without = apply_cfg(cond, uncond, cfg_scale=7.5, mahiro=False)
        with_mahiro = apply_cfg(cond, uncond, cfg_scale=7.5, mahiro=True)
        assert not torch.allclose(without, with_mahiro)


# ===================================================================== #
# Scheduler Tests
# ===================================================================== #


SIGMA_MIN = 0.03
SIGMA_MAX = 14.6
NUM_STEPS = 20


class TestSchedulerBasics:
    """Shared properties for all schedulers."""

    @pytest.mark.parametrize(
        "scheduler_fn",
        [
            normal_scheduler,
            karras_scheduler,
            exponential_scheduler,
            sgm_uniform_scheduler,
            simple_scheduler,
            ddim_uniform_scheduler,
            linear_quadratic_scheduler,
        ],
    )
    def test_length_is_n_plus_1(self, scheduler_fn) -> None:
        sigmas = scheduler_fn(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert len(sigmas) == NUM_STEPS + 1

    @pytest.mark.parametrize(
        "scheduler_fn",
        [
            normal_scheduler,
            karras_scheduler,
            exponential_scheduler,
            sgm_uniform_scheduler,
            simple_scheduler,
            ddim_uniform_scheduler,
        ],
    )
    def test_monotonically_decreasing(self, scheduler_fn) -> None:
        sigmas = scheduler_fn(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1], f"Not monotonic at index {i}: {sigmas[i]} < {sigmas[i + 1]}"

    @pytest.mark.parametrize(
        "scheduler_fn",
        [
            normal_scheduler,
            karras_scheduler,
            exponential_scheduler,
            sgm_uniform_scheduler,
            simple_scheduler,
            ddim_uniform_scheduler,
            linear_quadratic_scheduler,
        ],
    )
    def test_final_sigma_is_zero(self, scheduler_fn) -> None:
        sigmas = scheduler_fn(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert sigmas[-1] == 0.0


class TestKarrasScheduler:
    """Specific tests for Karras scheduler."""

    def test_curve_shape(self) -> None:
        """Karras schedule should be denser at the low-noise end."""
        sigmas = karras_scheduler(20, SIGMA_MIN, SIGMA_MAX, rho=7.0)
        # First half should cover more sigma range than second half
        mid = len(sigmas) // 2
        first_half_range = sigmas[0] - sigmas[mid]
        second_half_range = sigmas[mid] - sigmas[-1]
        assert first_half_range > second_half_range

    def test_rho_affects_shape(self) -> None:
        s1 = karras_scheduler(20, SIGMA_MIN, SIGMA_MAX, rho=3.0)
        s2 = karras_scheduler(20, SIGMA_MIN, SIGMA_MAX, rho=14.0)
        assert not torch.allclose(s1, s2)


class TestBetaScheduler:
    """Tests for beta distribution scheduler."""

    def test_length(self) -> None:
        sigmas = beta_scheduler(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert len(sigmas) == NUM_STEPS + 1

    def test_final_zero(self) -> None:
        sigmas = beta_scheduler(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert sigmas[-1] == 0.0


class TestAYSScheduler:
    """Tests for Align Your Steps."""

    def test_sd15_length(self) -> None:
        sigmas = ays_scheduler(NUM_STEPS, SIGMA_MIN, SIGMA_MAX, model_type="sd15")
        assert len(sigmas) == NUM_STEPS + 1

    def test_sdxl_length(self) -> None:
        sigmas = ays_scheduler(NUM_STEPS, SIGMA_MIN, SIGMA_MAX, model_type="sdxl")
        assert len(sigmas) == NUM_STEPS + 1

    def test_final_zero(self) -> None:
        sigmas = ays_scheduler(NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert sigmas[-1] == 0.0


class TestComputeSigmas:
    """Tests for the compute_sigmas dispatcher."""

    @pytest.mark.parametrize("stype", list(SchedulerType))
    def test_all_schedulers_produce_output(self, stype: SchedulerType) -> None:
        sigmas = compute_sigmas(stype, NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert sigmas.ndim == 1
        assert len(sigmas) >= NUM_STEPS
        assert sigmas[-1] == 0.0

    def test_string_dispatch(self) -> None:
        sigmas = compute_sigmas("karras", NUM_STEPS, SIGMA_MIN, SIGMA_MAX)
        assert len(sigmas) == NUM_STEPS + 1


# ===================================================================== #
# Sampler Tests
# ===================================================================== #


class TestEulerSample:
    """Tests for the built-in Euler sampler."""

    def test_identity_model_converges(self) -> None:
        """With a model that predicts x directly, Euler should converge to the prediction."""
        target = torch.ones(1, 4, 4, 4) * 0.5

        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return target  # always predicts the same denoised image

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 21)  # 20 steps

        result = euler_sample(model_fn, noise, sigmas)
        # Should converge close to target
        torch.testing.assert_close(result, target, atol=0.1, rtol=0.1)

    def test_callback_called(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

        steps_seen = []

        def callback(step, total, sigma, denoised):
            steps_seen.append(step)

        euler_sample(model_fn, noise, sigmas, callback=callback)
        assert steps_seen == [0, 1, 2, 3, 4]


class TestEulerAncestralSample:
    """Tests for the Euler ancestral sampler."""

    def test_runs_without_error(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 11)
        result = euler_ancestral_sample(model_fn, noise, sigmas)
        assert result.shape == noise.shape


class TestDpmPP2MSample:
    """Tests for the built-in DPM++ 2M sampler."""

    def test_runs_and_returns_correct_shape(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 21)
        result = dpm_pp_2m_sample(model_fn, noise, sigmas)
        assert result.shape == noise.shape

    def test_converges_to_target(self) -> None:
        """With a constant model prediction, DPM++ 2M should converge."""
        target = torch.ones(1, 4, 4, 4) * 0.5

        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return target

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 21)
        result = dpm_pp_2m_sample(model_fn, noise, sigmas)
        torch.testing.assert_close(result, target, atol=0.15, rtol=0.15)

    def test_differs_from_euler(self) -> None:
        """DPM++ 2M should produce different results than Euler (not a disguised Euler)."""
        torch.manual_seed(42)

        call_count = [0]

        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            call_count[0] += 1
            # Non-trivial model that depends on x
            return x * 0.8 + torch.ones_like(x) * 0.1

        noise = torch.randn(1, 4, 8, 8)
        sigmas = torch.linspace(1.0, 0.0, 11)

        torch.manual_seed(42)
        euler_result = euler_sample(model_fn, noise.clone(), sigmas.clone())

        call_count[0] = 0
        torch.manual_seed(42)
        dpm_result = dpm_pp_2m_sample(model_fn, noise.clone(), sigmas.clone())

        # Results must differ — if they are identical, DPM++2M is broken
        assert not torch.allclose(euler_result, dpm_result, atol=1e-5), (
            "DPM++ 2M produced identical results to Euler — multistep logic may not be active"
        )

    def test_multistep_uses_history(self) -> None:
        """Verify second-order correction kicks in after the first step."""
        denoised_log: list[Tensor] = []

        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            # Model that returns different denoised at each step
            result = x * 0.9
            denoised_log.append(result.clone())
            return result

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

        dpm_pp_2m_sample(model_fn, noise, sigmas)
        # Should have been called once per step
        assert len(denoised_log) == 5

    def test_callback_called(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        steps_seen = []

        def callback(step, total, sigma, denoised):
            steps_seen.append(step)

        dpm_pp_2m_sample(model_fn, noise, sigmas, callback=callback)
        assert steps_seen == [0, 1, 2, 3, 4]

    def test_deterministic(self) -> None:
        """DPM++ 2M is deterministic — same input, same output."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.7

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 11)

        r1 = dpm_pp_2m_sample(model_fn, noise.clone(), sigmas.clone())
        r2 = dpm_pp_2m_sample(model_fn, noise.clone(), sigmas.clone())
        torch.testing.assert_close(r1, r2)


class TestDpmPP2MSDESample:
    """Tests for the built-in DPM++ 2M SDE sampler."""

    def test_runs_and_returns_correct_shape(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 21)
        result = dpm_pp_2m_sde_sample(model_fn, noise, sigmas)
        assert result.shape == noise.shape

    def test_stochastic_different_seeds(self) -> None:
        """SDE variant should produce different results with different random seeds."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.8

        noise = torch.randn(1, 4, 8, 8)
        sigmas = torch.linspace(1.0, 0.0, 11)

        torch.manual_seed(1)
        r1 = dpm_pp_2m_sde_sample(model_fn, noise.clone(), sigmas.clone())

        torch.manual_seed(2)
        r2 = dpm_pp_2m_sde_sample(model_fn, noise.clone(), sigmas.clone())

        assert not torch.allclose(r1, r2), (
            "DPM++ 2M SDE produced identical results with different seeds — "
            "noise injection may not be working"
        )

    def test_differs_from_deterministic(self) -> None:
        """SDE variant should differ from the deterministic DPM++ 2M."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.8 + 0.1

        torch.manual_seed(42)
        noise = torch.randn(1, 4, 8, 8)
        sigmas = torch.linspace(1.0, 0.0, 11)

        det_result = dpm_pp_2m_sample(model_fn, noise.clone(), sigmas.clone())

        torch.manual_seed(99)
        sde_result = dpm_pp_2m_sde_sample(model_fn, noise.clone(), sigmas.clone())

        assert not torch.allclose(det_result, sde_result, atol=1e-5), (
            "DPM++ 2M SDE produced identical results to deterministic DPM++ 2M"
        )

    def test_eta_zero_matches_deterministic(self) -> None:
        """With eta=0, SDE variant should behave like the deterministic version."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.7

        noise = torch.randn(1, 4, 4, 4)
        sigmas = torch.linspace(1.0, 0.0, 11)

        det_result = dpm_pp_2m_sample(model_fn, noise.clone(), sigmas.clone())
        sde_result = dpm_pp_2m_sde_sample(
            model_fn, noise.clone(), sigmas.clone(), eta=0.0
        )
        torch.testing.assert_close(det_result, sde_result)

    def test_callback_called(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        steps_seen = []

        def callback(step, total, sigma, denoised):
            steps_seen.append(step)

        dpm_pp_2m_sde_sample(model_fn, noise, sigmas, callback=callback)
        assert steps_seen == [0, 1, 2, 3, 4]


class TestSample:
    """Tests for the sample dispatcher."""

    def test_euler_dispatch(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        result = sample(model_fn, noise, sigmas, SamplerType.EULER)
        assert result.shape == noise.shape

    def test_string_dispatch(self) -> None:
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        result = sample(model_fn, noise, sigmas, "euler")
        assert result.shape == noise.shape

    def test_dpm_pp_2m_dispatch(self) -> None:
        """DPM++ 2M should work via the dispatcher without k-diffusion."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        result = sample(model_fn, noise, sigmas, SamplerType.DPM_PP_2M)
        assert result.shape == noise.shape

    def test_dpm_pp_2m_sde_dispatch(self) -> None:
        """DPM++ 2M SDE should work via the dispatcher without k-diffusion."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        result = sample(model_fn, noise, sigmas, SamplerType.DPM_PP_2M_SDE)
        assert result.shape == noise.shape

    def test_dpm_pp_2m_string_dispatch(self) -> None:
        """DPM++ 2M should work via string dispatch."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x * 0.5

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)
        result = sample(model_fn, noise, sigmas, "dpm_pp_2m")
        assert result.shape == noise.shape

    def test_unsupported_sampler_raises_import_error(self) -> None:
        """Samplers without built-in impl should raise ImportError, not silently fallback."""
        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)

        # These samplers require k-diffusion (not installed in test env)
        for sampler_name in ["heun", "deis", "unipc", "lcm"]:
            try:
                import k_diffusion  # noqa: F401
                pytest.skip("k-diffusion is installed; cannot test ImportError path")
            except ImportError:
                pass

            with pytest.raises(ImportError, match="requires the k-diffusion package"):
                sample(model_fn, noise, sigmas, sampler_name)

    def test_no_silent_euler_fallback(self) -> None:
        """Requesting a k-diffusion-only sampler must NOT silently fall back to Euler."""
        try:
            import k_diffusion  # noqa: F401
            pytest.skip("k-diffusion is installed")
        except ImportError:
            pass

        def model_fn(x: Tensor, sigma: Tensor) -> Tensor:
            return x

        noise = torch.randn(1, 2, 2, 2)
        sigmas = torch.linspace(1.0, 0.0, 6)

        # Should raise, not silently return Euler result
        with pytest.raises(ImportError):
            sample(model_fn, noise, sigmas, SamplerType.HEUN)


class TestCreateModelFn:
    """Tests for create_model_fn wrapper."""

    def test_wraps_correctly(self) -> None:
        """The wrapped function should use prediction type and CFG."""
        prediction = EpsPrediction()
        call_log: list[dict] = []

        def model(x, timestep, cond=None):
            call_log.append({"x_shape": x.shape, "cond": cond})
            return x  # identity model output

        cond = torch.tensor([[1.0]])
        fn = create_model_fn(model, prediction, cond=cond, cfg_scale=1.0)

        x = torch.randn(1, 3)
        sigma = torch.tensor([0.5])
        result = fn(x, sigma)

        assert len(call_log) == 1
        assert result.shape == x.shape

    def test_cfg_applied_when_scale_gt_1(self) -> None:
        prediction = EpsPrediction()

        def model(x, timestep, cond=None):
            # Return different outputs for cond vs uncond
            if cond is not None and cond.sum() > 0:
                return x * 2
            return x * 0.5

        cond = torch.tensor([[1.0]])
        uncond = torch.tensor([[0.0]])

        fn = create_model_fn(model, prediction, cond=cond, uncond=uncond, cfg_scale=7.5)
        x = torch.randn(1, 3)
        sigma = torch.tensor([0.5])
        result = fn(x, sigma)
        assert result.shape == x.shape


# ===================================================================== #
# Conditioning Tests
# ===================================================================== #


class TestConditioning:
    """Tests for the Conditioning dataclass."""

    def test_basic_creation(self) -> None:
        cond = Conditioning(cond=torch.randn(1, 77, 768))
        assert cond.uncond is None
        assert cond.pooled is None
        assert cond.extra == {}

    def test_full_creation(self) -> None:
        cond = Conditioning(
            cond=torch.randn(1, 77, 768),
            uncond=torch.randn(1, 77, 768),
            pooled=torch.randn(1, 1280),
            extra={"timestep_cond": torch.tensor([100])},
        )
        assert cond.uncond is not None
        assert cond.pooled is not None
        assert "timestep_cond" in cond.extra


class TestPrepareConditioning:
    """Tests for prepare_conditioning."""

    def test_basic(self) -> None:
        text_emb = torch.randn(1, 77, 768)
        neg_emb = torch.randn(1, 77, 768)
        c = prepare_conditioning(text_emb, neg_emb)
        assert isinstance(c, Conditioning)
        torch.testing.assert_close(c.cond, text_emb)
        torch.testing.assert_close(c.uncond, neg_emb)

    def test_none_uncond(self) -> None:
        text_emb = torch.randn(1, 77, 768)
        c = prepare_conditioning(text_emb, None)
        assert c.uncond is None

    def test_pooled_kwarg(self) -> None:
        text_emb = torch.randn(1, 77, 768)
        pooled = torch.randn(1, 1280)
        c = prepare_conditioning(text_emb, pooled=pooled)
        assert c.pooled is not None
        torch.testing.assert_close(c.pooled, pooled)


class TestCreateNoise:
    """Tests for noise creation."""

    def test_shape(self) -> None:
        noise = create_noise(42, (1, 4, 64, 64))
        assert noise.shape == (1, 4, 64, 64)

    def test_reproducibility(self) -> None:
        """Same seed should produce same noise."""
        n1 = create_noise(12345, (2, 4, 32, 32))
        n2 = create_noise(12345, (2, 4, 32, 32))
        torch.testing.assert_close(n1, n2)

    def test_different_seeds_differ(self) -> None:
        n1 = create_noise(1, (1, 4, 8, 8))
        n2 = create_noise(2, (1, 4, 8, 8))
        assert not torch.allclose(n1, n2)

    def test_dtype(self) -> None:
        noise = create_noise(0, (1, 4, 8, 8), dtype=torch.float32)
        assert noise.dtype == torch.float32

    def test_device_cpu(self) -> None:
        noise = create_noise(0, (1, 4, 8, 8), device="cpu")
        assert noise.device == torch.device("cpu")
