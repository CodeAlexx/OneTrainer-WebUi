"""Custom gradient scaler with fused backward pass support.

Extends ``torch.amp.GradScaler`` with per-parameter unscale/step methods
for fused backward passes that combine loss scaling with backward to
reduce peak memory.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler, OptState
from torch.nn import Parameter

__all__ = ["CustomGradScaler", "DummyOptimizer"]


class DummyOptimizer:
    """Minimal optimizer facade for per-parameter unscaling.

    ``GradScaler._unscale_grads_`` expects an optimizer with a
    ``param_groups`` attribute.  This wraps a single parameter so
    we can unscale one tensor at a time during fused backward passes.
    """

    def __init__(self, parameter: Parameter | Tensor) -> None:
        self.param_groups = [{"params": [parameter]}]


class CustomGradScaler(GradScaler):
    """Extends ``GradScaler`` with per-parameter fused backward support.

    In a fused backward pass the optimizer step is executed inside the
    backward hook for each parameter, immediately after its gradient is
    computed.  This avoids accumulating all gradients in memory at once.

    The three key methods are:

    1. :meth:`unscale_parameter_` -- unscale a single parameter's gradient.
    2. :meth:`maybe_opt_step_parameter` -- step only if no inf/nan found.
    3. :meth:`step_after_unscale_parameter_` -- finalize state after all
       per-parameter steps.

    Usage in a fused backward hook::

        def grad_hook(tensor):
            scaler.unscale_parameter_(tensor, optimizer)
            nn.utils.clip_grad_norm_(tensor, max_norm)
            scaler.maybe_opt_step_parameter(tensor, param_group, i, optimizer)
            tensor.grad = None
    """

    def __init__(self) -> None:
        super().__init__()

    def unscale_parameter_(
        self,
        parameter: Parameter | Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Unscale the gradient of a single parameter.

        This is the per-parameter equivalent of ``unscale_(optimizer)``
        used during fused backward passes.

        Raises:
            RuntimeError: If unscale has already been called on this
                optimizer since the last ``update()``, or if ``step()``
                was already called.
        """
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer "
                "since the last update()."
            )
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options,
        # so compute the reciprocal in FP64.
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full(
            (), 0.0, dtype=torch.float32, device=self._scale.device,
        )

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            DummyOptimizer(parameter), inv_scale, found_inf, False,
        )

    def maybe_opt_step_parameter(
        self,
        parameter: Parameter | Tensor,
        param_group: dict,
        i: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Step the optimizer for a single parameter if no inf/nan detected.

        The optimizer must implement ``step_parameter(parameter, param_group, i)``.
        """
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if not sum(
            v.item() for v in optimizer_state["found_inf_per_device"].values()
        ):
            optimizer.step_parameter(parameter, param_group, i)

    def step_after_unscale_parameter_(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Finalize scaler state after all per-parameter steps.

        Call this once after the backward pass completes and all
        per-parameter steps have been executed.

        Raises:
            RuntimeError: If ``step()`` was already called, or if no
                inf checks were recorded.
        """
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        optimizer_state["stage"] = OptState.UNSCALED

        self._check_scale_growth_tracker("step")

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert (
            len(optimizer_state["found_inf_per_device"]) > 0
        ), "No inf checks were recorded for this optimizer."

        optimizer_state["stage"] = OptState.STEPPED
