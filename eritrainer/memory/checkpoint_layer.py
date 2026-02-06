"""Checkpoint wrappers that integrate with native layer offloading."""

from __future__ import annotations

from itertools import count
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint

from eritrainer.memory.conductor import LayerOffloadConductor

_CALL_ID = count(1)


def generate_call_id() -> int:
    return next(_CALL_ID)


class OffloadCheckpointLayer(nn.Module):
    """Wrap a single layer with optional checkpointing and offload callbacks."""

    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        conductor: LayerOffloadConductor | None = None,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = int(layer_idx)
        self.conductor = conductor
        self.use_checkpointing = bool(use_checkpointing)

    def _forward_fn(self, *args, **kwargs) -> Any:
        return self.layer(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        call_id = generate_call_id()
        effective_args = args
        if self.conductor is not None:
            maybe_args = self.conductor.before_layer(self.layer_idx, call_id, args)
            if isinstance(maybe_args, tuple):
                effective_args = maybe_args
            elif maybe_args is not None:
                effective_args = (maybe_args,)

        run_with_checkpoint = (
            self.use_checkpointing
            and self.training
            and len(kwargs) == 0
            and any(torch.is_tensor(arg) and arg.requires_grad for arg in effective_args)
        )

        if run_with_checkpoint:
            output = torch.utils.checkpoint.checkpoint(
                self._forward_fn,
                *effective_args,
                use_reentrant=True,
            )
        else:
            output = self._forward_fn(*effective_args, **kwargs)

        if self.conductor is not None:
            output = self.conductor.after_layer(self.layer_idx, call_id, output)

        return output

    def __repr__(self) -> str:
        return (
            "OffloadCheckpointLayer("
            f"layer_idx={self.layer_idx}, "
            f"use_checkpointing={self.use_checkpointing}, "
            f"wrapped={self.layer.__class__.__name__})"
        )
