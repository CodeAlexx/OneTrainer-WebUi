"""Per-model training setup classes.

Provides a single ``ModelSetup`` base with thin per-model subclasses.

Each setup class handles:
- ``setup_model()``:  Prepare model for training (freeze layers, inject LoRA, etc.)
- ``setup_train_device()``: Move components to train/temp devices
- ``create_parameters()``: Build named parameter groups for the optimizer
- ``setup_optimizable_parameters()``: Convenience that combines the above
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from serenity.core.interfaces import ModelType
from serenity.training.param_groups import (
    NamedParameterGroup,
    NamedParameterGroupCollection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ModelSetup(ABC):
    """Base training setup for all model families.

    Subclasses implement the abstract hooks for each model family.
    """

    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        *,
        debug_mode: bool = False,
    ) -> None:
        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode
        self.frozen_parameters: dict[str, list[torch.nn.Parameter]] = {}

    # -- abstract hooks --

    @abstractmethod
    def create_parameters(
        self,
        model: Any,
        config: Any,
    ) -> NamedParameterGroupCollection:
        """Build named parameter groups for the optimizer."""

    @abstractmethod
    def setup_model(self, model: Any, config: Any) -> None:
        """Prepare model for training (freeze layers, inject adapters, etc.)."""

    @abstractmethod
    def setup_train_device(self, model: Any, config: Any) -> None:
        """Place model components on train/temp devices."""

    # -- convenience --

    def setup_optimizable_parameters(
        self,
        model: Any,
        config: Any,
    ) -> list[dict[str, Any]]:
        """Full setup pipeline: model prep + device placement + param groups.

        Returns optimizer-ready parameter group list.
        """
        self.setup_model(model, config)
        self.setup_train_device(model, config)
        collection = self.create_parameters(model, config)
        base_lr: float = getattr(config, "learning_rate", 1e-4)
        return collection.for_optimizer(base_lr)

    # -- shared helpers --

    def _add_model_part_parameters(
        self,
        collection: NamedParameterGroupCollection,
        name: str,
        module: torch.nn.Module | None,
        part_config: Any,
        *,
        freeze_filters: list[str] | None = None,
    ) -> None:
        """Add a model component's parameters to the collection.

        Parameters
        ----------
        collection:
            Target collection.
        name:
            Unique group name (e.g. "text_encoder_1", "unet_lora").
        module:
            The ``nn.Module`` to extract parameters from.
        part_config:
            A ``TrainModelPartConfig`` (or compatible) with ``.train`` and
            ``.learning_rate`` attributes.
        freeze_filters:
            If provided, only parameters whose names match one of these
            substring filters will be included; the rest are frozen.
        """
        if module is None:
            return

        train = getattr(part_config, "train", True)
        if not train:
            return

        lr = getattr(part_config, "learning_rate", None)

        if freeze_filters:
            selected: list[torch.nn.Parameter] = []
            frozen: list[torch.nn.Parameter] = []
            for pname, param in module.named_parameters():
                if any(f in pname for f in freeze_filters):
                    selected.append(param)
                else:
                    frozen.append(param)
            self.frozen_parameters[name] = frozen
            params = selected
            logger.debug(
                "%s: %d selected, %d frozen by filter",
                name, len(selected), len(frozen),
            )
        else:
            params = list(module.parameters())

        if params:
            collection.add_group(NamedParameterGroup(
                unique_name=name,
                parameters=params,
                learning_rate=lr,
            ))

    def _set_requires_grad(
        self,
        name: str,
        module: torch.nn.Module | None,
        part_config: Any,
    ) -> None:
        """Enable or disable gradients for a model component.

        """
        if module is None:
            return

        train = getattr(part_config, "train", True)
        module.requires_grad_(train)

        # Frozen parameters from selective filters must stay frozen
        if name in self.frozen_parameters:
            for param in self.frozen_parameters[name]:
                param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Per-model setup: Stable Diffusion XL
# ---------------------------------------------------------------------------

class SDXLSetup(ModelSetup):
    """Training setup for SDXL models (fine-tune and LoRA)."""

    def create_parameters(
        self,
        model: Any,
        config: Any,
    ) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()

        # Text encoders
        te1 = getattr(model, "text_encoder_1_lora", None) or getattr(model, "text_encoder_1", None)
        te2 = getattr(model, "text_encoder_2_lora", None) or getattr(model, "text_encoder_2", None)
        te1_name = "text_encoder_1_lora" if hasattr(model, "text_encoder_1_lora") and model.text_encoder_1_lora else "text_encoder_1"
        te2_name = "text_encoder_2_lora" if hasattr(model, "text_encoder_2_lora") and model.text_encoder_2_lora else "text_encoder_2"

        self._add_model_part_parameters(collection, te1_name, te1, config.text_encoder)
        self._add_model_part_parameters(collection, te2_name, te2, config.text_encoder_2)

        # UNet / LoRA
        unet = getattr(model, "unet_lora", None) or getattr(model, "unet", None)
        unet_name = "unet_lora" if hasattr(model, "unet_lora") and model.unet_lora else "unet"
        self._add_model_part_parameters(collection, unet_name, unet, config.unet)

        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        # Freeze VAE -- never trained for SDXL
        vae = getattr(model, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)

        # Set requires_grad based on config
        self._set_requires_grad("text_encoder_1", getattr(model, "text_encoder_1", None), config.text_encoder)
        self._set_requires_grad("text_encoder_2", getattr(model, "text_encoder_2", None), config.text_encoder_2)
        self._set_requires_grad("unet", getattr(model, "unet", None), config.unet)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)

        # VAE only needed on train device when not caching latents
        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)

        # Text encoders: on train device if training or not caching
        te_train = getattr(config.text_encoder, "train", False)
        te2_train = getattr(config.text_encoder_2, "train", False)
        _move(model, "text_encoder_1", self.train_device if (te_train or not latent_caching) else self.temp_device)
        _move(model, "text_encoder_2", self.train_device if (te2_train or not latent_caching) else self.temp_device)

        # UNet always on train device
        _move(model, "unet", self.train_device)

        # Set train/eval mode
        _set_mode(model, "text_encoder_1", train=te_train)
        _set_mode(model, "text_encoder_2", train=te2_train)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "unet", train=getattr(config.unet, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: Stable Diffusion 1.5
# ---------------------------------------------------------------------------

class SD15Setup(ModelSetup):
    """Training setup for SD 1.5 models."""

    def create_parameters(
        self,
        model: Any,
        config: Any,
    ) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()
        te = getattr(model, "text_encoder_lora", None) or getattr(model, "text_encoder", None)
        te_name = "text_encoder_lora" if (getattr(model, "text_encoder_lora", None) is not None) else "text_encoder"
        self._add_model_part_parameters(collection, te_name, te, config.text_encoder)

        unet = getattr(model, "unet_lora", None) or getattr(model, "unet", None)
        unet_name = "unet_lora" if (getattr(model, "unet_lora", None) is not None) else "unet"
        self._add_model_part_parameters(collection, unet_name, unet, config.unet)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        vae = getattr(model, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)
        self._set_requires_grad("text_encoder", getattr(model, "text_encoder", None), config.text_encoder)
        self._set_requires_grad("unet", getattr(model, "unet", None), config.unet)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)
        te_train = getattr(config.text_encoder, "train", False)

        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder", self.train_device if (te_train or not latent_caching) else self.temp_device)
        _move(model, "unet", self.train_device)

        _set_mode(model, "text_encoder", train=te_train)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "unet", train=getattr(config.unet, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: SD3 / SD3.5
# ---------------------------------------------------------------------------

class SD3Setup(ModelSetup):
    """Training setup for Stable Diffusion 3.x models."""

    def create_parameters(
        self,
        model: Any,
        config: Any,
    ) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()

        for te_attr, cfg_attr in [
            ("text_encoder_1", "text_encoder"),
            ("text_encoder_2", "text_encoder_2"),
            ("text_encoder_3", "text_encoder_3"),
        ]:
            lora_attr = te_attr + "_lora"
            te = getattr(model, lora_attr, None) or getattr(model, te_attr, None)
            name = lora_attr if getattr(model, lora_attr, None) is not None else te_attr
            part_cfg = getattr(config, cfg_attr, None)
            if part_cfg is not None:
                self._add_model_part_parameters(collection, name, te, part_cfg)

        transformer = getattr(model, "transformer_lora", None) or getattr(model, "transformer", None)
        t_name = "transformer_lora" if getattr(model, "transformer_lora", None) is not None else "transformer"
        self._add_model_part_parameters(collection, t_name, transformer, config.transformer)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        vae = getattr(model, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)
        self._set_requires_grad("transformer", getattr(model, "transformer", None), config.transformer)
        for te_attr, cfg_attr in [
            ("text_encoder_1", "text_encoder"),
            ("text_encoder_2", "text_encoder_2"),
            ("text_encoder_3", "text_encoder_3"),
        ]:
            part_cfg = getattr(config, cfg_attr, None)
            if part_cfg is not None:
                self._set_requires_grad(te_attr, getattr(model, te_attr, None), part_cfg)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)
        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "transformer", self.train_device)

        for te_attr, cfg_attr in [
            ("text_encoder_1", "text_encoder"),
            ("text_encoder_2", "text_encoder_2"),
            ("text_encoder_3", "text_encoder_3"),
        ]:
            part_cfg = getattr(config, cfg_attr, None)
            te_train = getattr(part_cfg, "train", False) if part_cfg else False
            _move(model, te_attr, self.train_device if (te_train or not latent_caching) else self.temp_device)
            _set_mode(model, te_attr, train=te_train)

        _set_mode(model, "vae", train=False)
        _set_mode(model, "transformer", train=getattr(config.transformer, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: Flux / Flux2
# ---------------------------------------------------------------------------

class FluxSetup(ModelSetup):
    """Training setup for Flux / Flux2 models (including Klein variants)."""

    def create_parameters(
        self,
        model: Any,
        config: Any,
    ) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()

        # Text encoder (single for Flux)
        te = getattr(model, "text_encoder_lora", None) or getattr(model, "text_encoder", None)
        te_name = "text_encoder_lora" if getattr(model, "text_encoder_lora", None) is not None else "text_encoder"
        te_cfg = getattr(config, "text_encoder", None)
        if te_cfg is not None:
            self._add_model_part_parameters(collection, te_name, te, te_cfg)

        # Transformer
        transformer = getattr(model, "transformer_lora", None) or getattr(model, "transformer", None)
        t_name = "transformer_lora" if getattr(model, "transformer_lora", None) is not None else "transformer"
        self._add_model_part_parameters(collection, t_name, transformer, config.transformer)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        vae = getattr(model, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)

        te = getattr(model, "text_encoder", None)
        if te is not None:
            te.requires_grad_(False)

        self._set_requires_grad("transformer", getattr(model, "transformer", None), config.transformer)

        # If LoRA, handle adapter requires_grad
        t_lora = getattr(model, "transformer_lora", None)
        if t_lora is not None:
            self._set_requires_grad("transformer_lora", t_lora, config.transformer)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)

        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder", self.train_device if not latent_caching else self.temp_device)
        _move(model, "transformer", self.train_device)

        _set_mode(model, "text_encoder", train=False)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "transformer", train=getattr(config.transformer, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: Z-Image
# ---------------------------------------------------------------------------

class ZImageSetup(ModelSetup):
    """Training setup for Z-Image models."""

    def create_parameters(self, model: Any, config: Any) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()
        transformer = getattr(model, "transformer_lora", None) or getattr(model, "transformer", None)
        t_name = "transformer_lora" if getattr(model, "transformer_lora", None) is not None else "transformer"
        self._add_model_part_parameters(collection, t_name, transformer, config.transformer)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        for attr in ("vae", "text_encoder", "text_encoder_2"):
            mod = getattr(model, attr, None)
            if mod is not None:
                mod.requires_grad_(False)
        self._set_requires_grad("transformer", getattr(model, "transformer", None), config.transformer)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)
        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder_2", self.train_device if not latent_caching else self.temp_device)
        _move(model, "transformer", self.train_device)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "text_encoder", train=False)
        _set_mode(model, "text_encoder_2", train=False)
        _set_mode(model, "transformer", train=getattr(config.transformer, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: LTX2
# ---------------------------------------------------------------------------

class LTX2Setup(ModelSetup):
    """Training setup for LTX2 video models."""

    def create_parameters(self, model: Any, config: Any) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()
        transformer = getattr(model, "transformer_lora", None) or getattr(model, "transformer", None)
        t_name = "transformer_lora" if getattr(model, "transformer_lora", None) is not None else "transformer"
        self._add_model_part_parameters(collection, t_name, transformer, config.transformer)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        for attr in ("vae", "text_encoder"):
            mod = getattr(model, attr, None)
            if mod is not None:
                mod.requires_grad_(False)
        self._set_requires_grad("transformer", getattr(model, "transformer", None), config.transformer)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)
        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder", self.train_device if not latent_caching else self.temp_device)
        _move(model, "transformer", self.train_device)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "text_encoder", train=False)
        _set_mode(model, "transformer", train=getattr(config.transformer, "train", True))


# ---------------------------------------------------------------------------
# Per-model setup: Qwen
# ---------------------------------------------------------------------------

class QwenSetup(ModelSetup):
    """Training setup for Qwen image-edit models."""

    def create_parameters(self, model: Any, config: Any) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()
        transformer = getattr(model, "transformer_lora", None) or getattr(model, "transformer", None)
        t_name = "transformer_lora" if getattr(model, "transformer_lora", None) is not None else "transformer"
        self._add_model_part_parameters(collection, t_name, transformer, config.transformer)

        te = getattr(model, "text_encoder_lora", None) or getattr(model, "text_encoder", None)
        te_name = "text_encoder_lora" if getattr(model, "text_encoder_lora", None) is not None else "text_encoder"
        te_cfg = getattr(config, "text_encoder", None)
        if te_cfg is not None:
            self._add_model_part_parameters(collection, te_name, te, te_cfg)
        return collection

    def setup_model(self, model: Any, config: Any) -> None:
        vae = getattr(model, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)
        self._set_requires_grad("transformer", getattr(model, "transformer", None), config.transformer)
        te_cfg = getattr(config, "text_encoder", None)
        if te_cfg is not None:
            self._set_requires_grad("text_encoder", getattr(model, "text_encoder", None), te_cfg)

    def setup_train_device(self, model: Any, config: Any) -> None:
        latent_caching = getattr(config, "latent_caching", True)
        te_cfg = getattr(config, "text_encoder", None)
        te_train = getattr(te_cfg, "train", False) if te_cfg else False
        _move(model, "vae", self.train_device if not latent_caching else self.temp_device)
        _move(model, "text_encoder", self.train_device if (te_train or not latent_caching) else self.temp_device)
        _move(model, "transformer", self.train_device)
        _set_mode(model, "vae", train=False)
        _set_mode(model, "text_encoder", train=te_train)
        _set_mode(model, "transformer", train=getattr(config.transformer, "train", True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move(model: Any, attr: str, device: torch.device) -> None:
    """Move a model component to *device* if it exists."""
    # Try model-specific _to method first (e.g. model.vae_to())
    to_method = getattr(model, f"{attr}_to", None)
    if callable(to_method):
        to_method(device)
        return
    mod = getattr(model, attr, None)
    if mod is not None and hasattr(mod, "to"):
        mod.to(device)


def _set_mode(model: Any, attr: str, *, train: bool) -> None:
    """Set a model component to train or eval mode."""
    mod = getattr(model, attr, None)
    if mod is None:
        return
    if train:
        mod.train()
    else:
        mod.eval()


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_SETUP_REGISTRY: dict[str, type[ModelSetup]] = {
    # SDXL family
    ModelType.SDXL.value: SDXLSetup,
    ModelType.SDXL_10_BASE.value: SDXLSetup,
    ModelType.SDXL_INPAINTING.value: SDXLSetup,
    # SD 1.5 family
    ModelType.SD15.value: SD15Setup,
    ModelType.SD15_INPAINTING.value: SD15Setup,
    ModelType.SD20.value: SD15Setup,
    ModelType.SD20_BASE.value: SD15Setup,
    ModelType.SD21.value: SD15Setup,
    ModelType.SD21_BASE.value: SD15Setup,
    # SD3 family
    ModelType.SD3.value: SD3Setup,
    ModelType.SD35.value: SD3Setup,
    # Flux family
    ModelType.FLUX_DEV.value: FluxSetup,
    ModelType.FLUX_SCHNELL.value: FluxSetup,
    ModelType.FLUX_2.value: FluxSetup,
    ModelType.FLUX_2_DEV.value: FluxSetup,
    ModelType.FLUX_2_KLEIN.value: FluxSetup,
    ModelType.FLUX_2_KLEIN_4B.value: FluxSetup,
    ModelType.FLUX_2_KLEIN_9B.value: FluxSetup,
    ModelType.FLUX_FILL_DEV.value: FluxSetup,
    # Z-Image
    ModelType.ZIMAGE.value: ZImageSetup,
    ModelType.Z_IMAGE.value: ZImageSetup,
    # LTX2
    ModelType.LTX2.value: LTX2Setup,
    # Qwen
    ModelType.QWEN.value: QwenSetup,
    ModelType.QWEN_IMAGE_EDIT.value: QwenSetup,
}


def create_model_setup(
    model_type: ModelType | str,
    train_device: torch.device | str = "cuda",
    temp_device: torch.device | str = "cpu",
    *,
    debug_mode: bool = False,
) -> ModelSetup:
    """Factory to create the appropriate ``ModelSetup`` for a model type."""
    key = model_type.value if isinstance(model_type, ModelType) else str(model_type).lower()
    cls = _SETUP_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No ModelSetup registered for model type {key!r}.  "
            f"Known types: {sorted(_SETUP_REGISTRY.keys())}"
        )
    return cls(
        train_device=torch.device(train_device),
        temp_device=torch.device(temp_device),
        debug_mode=debug_mode,
    )


__all__ = [
    "ModelSetup",
    "SDXLSetup",
    "SD15Setup",
    "SD3Setup",
    "FluxSetup",
    "ZImageSetup",
    "LTX2Setup",
    "QwenSetup",
    "create_model_setup",
]
