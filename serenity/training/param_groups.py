"""Named parameter groups for per-component learning rates.

Ported from OneTrainer's ``NamedParameterGroup`` / ``NamedParameterGroupCollection``
(``modules/util/NamedParameterGroup.py``).  Serenity simplifies the API while
keeping full PyTorch optimizer compatibility.

Each group is a dict following the PyTorch convention::

    {'params': [...], 'lr': float, 'name': str}

This module also provides ``create_param_groups`` which inspects a model and
config to build per-component groups with independent learning rates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named group primitives
# ---------------------------------------------------------------------------

@dataclass
class NamedParameterGroup:
    """A single named optimizer parameter group."""

    unique_name: str
    parameters: list[Parameter] = field(default_factory=list)
    learning_rate: float | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is None:
            self.display_name = self.unique_name
        # Materialise iterators so they can be reused
        if not isinstance(self.parameters, list):
            self.parameters = list(self.parameters)


class NamedParameterGroupCollection:
    """Collect multiple ``NamedParameterGroup`` entries and convert to
    PyTorch optimizer ``param_groups`` format.

    Mirrors OneTrainer's ``NamedParameterGroupCollection`` but avoids the
    coupling to ``TrainConfig`` for the LR scaler.
    """

    def __init__(self) -> None:
        self._groups: list[NamedParameterGroup] = []

    # -- mutators --

    def add_group(self, group: NamedParameterGroup) -> None:
        self._groups.append(group)

    def add(
        self,
        name: str,
        parameters: Iterable[Parameter],
        lr: float | None = None,
        *,
        display_name: str | None = None,
    ) -> None:
        """Convenience shorthand to create and add a group in one call."""
        self.add_group(NamedParameterGroup(
            unique_name=name,
            parameters=list(parameters),
            learning_rate=lr,
            display_name=display_name,
        ))

    # -- accessors --

    def all_parameters(self) -> list[Parameter]:
        """Flat list of all parameters across groups."""
        return [p for g in self._groups for p in g.parameters]

    @property
    def unique_names(self) -> list[str]:
        return [g.unique_name for g in self._groups]

    @property
    def display_names(self) -> list[str]:
        return [g.display_name or g.unique_name for g in self._groups]

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups)

    # -- optimizer integration --

    def for_optimizer(
        self,
        base_lr: float,
        *,
        lr_scale: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Convert to a list of dicts suitable for a PyTorch optimizer.

        Parameters
        ----------
        base_lr:
            Default learning rate used when a group has no explicit LR.
        lr_scale:
            Multiplicative scaling applied to every group's LR (e.g. for
            batch-size-aware scaling).

        Returns
        -------
        list[dict]
            ``[{'params': [...], 'lr': float, 'initial_lr': float, 'name': str}, ...]``
        """
        result: list[dict[str, Any]] = []
        for group in self._groups:
            lr = group.learning_rate if group.learning_rate is not None else base_lr
            lr = lr * lr_scale
            result.append({
                "name": group.display_name or group.unique_name,
                "params": list(group.parameters),
                "lr": lr,
                "initial_lr": lr,
            })
        return result


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def _collect_trainable(module: torch.nn.Module) -> list[Parameter]:
    """Return parameters that have ``requires_grad=True``."""
    return [p for p in module.parameters() if p.requires_grad]


def create_param_groups(
    model: Any,
    config: Any,
    *,
    lr_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Build per-component parameter groups from a Serenity model + config.

    Inspects well-known model attributes (``text_encoder``, ``text_encoder_2``,
    ``unet``, ``transformer``, plus LoRA variants) and the corresponding
    ``TrainModelPartConfig`` entries on *config* to decide which components
    are trainable and at what learning rate.

    Parameters
    ----------
    model:
        A Serenity model instance (or any object with standard component attrs).
    config:
        A ``TrainConfig`` instance with per-component ``TrainModelPartConfig``
        sub-configs and a ``learning_rate`` base value.
    lr_scale:
        Extra multiplier applied to all LRs (e.g. sqrt-scaling by batch size).

    Returns
    -------
    list[dict]
        Ready for ``torch.optim.Optimizer(param_groups, ...)``.
    """
    collection = NamedParameterGroupCollection()
    base_lr: float = getattr(config, "learning_rate", 1e-4)

    # Component discovery table: (attr_name, config_attr, group_name)
    _COMPONENT_TABLE: list[tuple[str, str, str]] = [
        # Text encoders
        ("text_encoder_1_lora", "text_encoder", "text_encoder_1_lora"),
        ("text_encoder_2_lora", "text_encoder_2", "text_encoder_2_lora"),
        ("text_encoder_1", "text_encoder", "text_encoder_1"),
        ("text_encoder_2", "text_encoder_2", "text_encoder_2"),
        ("text_encoder", "text_encoder", "text_encoder"),
        # Backbone
        ("transformer_lora", "transformer", "transformer_lora"),
        ("unet_lora", "unet", "unet_lora"),
        ("transformer", "transformer", "transformer"),
        ("unet", "unet", "unet"),
    ]

    seen_attrs: set[str] = set()

    for model_attr, cfg_attr, group_name in _COMPONENT_TABLE:
        component = getattr(model, model_attr, None)
        if component is None:
            continue
        if model_attr in seen_attrs:
            continue

        part_cfg = getattr(config, cfg_attr, None)
        if part_cfg is None:
            continue

        # Check if this component should be trained
        train_flag = getattr(part_cfg, "train", True)
        if not train_flag:
            continue

        # Get component-specific LR (falls back to base)
        component_lr = getattr(part_cfg, "learning_rate", None)
        if component_lr is None or component_lr <= 0:
            component_lr = base_lr

        # Collect trainable params
        if hasattr(component, "parameters"):
            params = _collect_trainable(component)
        else:
            continue

        if not params:
            continue

        collection.add(group_name, params, lr=component_lr)
        seen_attrs.add(model_attr)

        # LoRA attr implies we skip the non-LoRA version
        if model_attr.endswith("_lora"):
            base_attr = model_attr.replace("_lora", "")
            seen_attrs.add(base_attr)

    # Check for standalone LoRA layers (e.g. from LyCORIS)
    lora_layers = getattr(model, "lora_layers", None)
    if lora_layers is not None and "lora_layers" not in seen_attrs:
        params = _collect_trainable(lora_layers) if hasattr(lora_layers, "parameters") else []
        if params:
            lora_lr = base_lr
            collection.add("lora_layers", params, lr=lora_lr)

    if len(collection) == 0:
        logger.warning(
            "create_param_groups: no trainable components found on model %s",
            type(model).__name__,
        )

    return collection.for_optimizer(base_lr, lr_scale=lr_scale)


__all__ = [
    "NamedParameterGroup",
    "NamedParameterGroupCollection",
    "create_param_groups",
]
