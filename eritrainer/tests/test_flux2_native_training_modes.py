"""Tests for native FLUX.2 training mode selection and adapter parsing."""

from __future__ import annotations

from eritrainer.cli.native_flux2 import (
    _build_adapter_kwargs,
    _create_lr_scheduler,
    _create_optimizer,
    _extract_adapter_config,
)
from eritrainer.training.flux2.image_trainer import Flux2ImageTrainer, Flux2ImageTrainerConfig

import torch
import torch.nn as nn


class _TinyModel:
    def __init__(self) -> None:
        self.transformer = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.vae = nn.Sequential(nn.Linear(4, 4))
        self.text_encoder = nn.Sequential(nn.Linear(4, 4))
        self.text_encoder_2 = None


class _DummyAdapter:
    def __init__(self, params: list[nn.Parameter]) -> None:
        self._params = params

    def get_trainable_params(self):
        return self._params


def _make_trainable_params() -> list[nn.Parameter]:
    layer = nn.Linear(4, 4)
    return list(layer.parameters())


def test_extract_adapter_config_supports_extended_lycoris_types():
    adapter_type, adapter_cfg = _extract_adapter_config(
        {
            "peft_type": "diag-oft",
            "diag_oft": {
                "rank": 8,
                "block_size": 4,
            },
        }
    )

    assert adapter_type == "diag_oft"
    assert adapter_cfg["rank"] == 8


def test_extract_adapter_config_detects_full_finetune():
    adapter_type, adapter_cfg = _extract_adapter_config(
        {
            "training_method": "full_finetune",
            "peft_type": "lora",
        }
    )

    assert adapter_type == "full"
    assert adapter_cfg == {}


def test_build_adapter_kwargs_parses_targets_and_booleans():
    kwargs = _build_adapter_kwargs(
        {
            "target_modules": "to_q,to_k,to_v",
            "rank_dropout": 0.1,
            "module_dropout": 0.2,
            "decompose_both": "true",
            "rescaled": "false",
            "block_size": 8,
        }
    )

    assert kwargs["target_modules"] == ["to_q", "to_k", "to_v"]
    assert kwargs["rank_dropout"] == 0.1
    assert kwargs["module_dropout"] == 0.2
    assert kwargs["decompose_both"] is True
    assert kwargs["rescaled"] is False
    assert kwargs["block_size"] == 8


def test_to_train_mode_full_finetune_unfreezes_transformer():
    model = _TinyModel()
    cfg = Flux2ImageTrainerConfig(
        model_path="/tmp/mock",
        model_variant="klein-4b",
        train_device="cpu",
        temp_device="cpu",
        gradient_checkpointing="off",
    )
    trainer = Flux2ImageTrainer(cfg, model=model)

    trainer.to_train_mode()

    assert all(param.requires_grad for param in model.transformer.parameters())
    assert all(not param.requires_grad for param in model.vae.parameters())
    assert all(not param.requires_grad for param in model.text_encoder.parameters())


def test_to_train_mode_adapter_freezes_backbone():
    model = _TinyModel()
    cfg = Flux2ImageTrainerConfig(
        model_path="/tmp/mock",
        model_variant="klein-4b",
        train_device="cpu",
        temp_device="cpu",
        gradient_checkpointing="off",
    )
    trainer = Flux2ImageTrainer(cfg, model=model)

    trainable_param = next(model.transformer.parameters())
    trainer.adapter = _DummyAdapter([trainable_param])
    trainer.to_train_mode()

    assert trainable_param.requires_grad is True
    frozen = [param for param in model.transformer.parameters() if param is not trainable_param]
    assert frozen
    assert all(not param.requires_grad for param in frozen)


def test_create_optimizer_maps_schedule_free_alias_to_adamw():
    params = _make_trainable_params()
    optimizer, optimizer_name = _create_optimizer(
        params,
        config={},
        optimizer_block={"optimizer": "SCHEDULE_FREE_ADAMW", "weight_decay": 0.01},
        learning_rate=1e-4,
    )
    assert optimizer_name == "adamw"
    assert isinstance(optimizer, torch.optim.AdamW)


def test_create_optimizer_supports_sgd():
    params = _make_trainable_params()
    optimizer, optimizer_name = _create_optimizer(
        params,
        config={},
        optimizer_block={"optimizer": "sgd", "momentum": 0.9, "nesterov": True},
        learning_rate=1e-3,
    )
    assert optimizer_name == "sgd"
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["nesterov"] is True


def test_create_lr_scheduler_supports_linear_with_warmup():
    params = _make_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=1.0)
    scheduler, scheduler_name = _create_lr_scheduler(
        optimizer,
        config={"learning_rate_scheduler": "LINEAR", "learning_rate_warmup_steps": 2},
        scheduler_block={},
        total_optimizer_steps=6,
    )
    assert scheduler is not None
    assert scheduler_name == "linear"
    warmup_initial_lr = float(optimizer.param_groups[0]["lr"])
    assert warmup_initial_lr < 1.0

    lrs: list[float] = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[0] >= warmup_initial_lr
    assert lrs[-1] <= lrs[2]


def test_create_lr_scheduler_constant_without_warmup_returns_none():
    params = _make_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=1.0)
    scheduler, scheduler_name = _create_lr_scheduler(
        optimizer,
        config={"learning_rate_scheduler": "CONSTANT"},
        scheduler_block={},
        total_optimizer_steps=10,
    )
    assert scheduler is None
    assert scheduler_name == "constant"
