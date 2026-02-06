"""Tests for native Z-Image model helpers."""

from __future__ import annotations

from types import SimpleNamespace

from eritrainer.models.zimage import ZImageModel, _format_chat_prompt, _validate_zimage_model_path

import torch

import pytest


class _TokenizerWithTemplate:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ):
        self.calls.append(
            (
                messages,
                {
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                },
            )
        )
        return "<chat>prompt</chat>"


class _TokenizerWithoutTemplate:
    pass


def test_format_chat_prompt_uses_template():
    tokenizer = _TokenizerWithTemplate()
    formatted = _format_chat_prompt(tokenizer, "describe image")

    assert formatted == "<chat>prompt</chat>"
    assert tokenizer.calls
    _, kwargs = tokenizer.calls[0]
    assert kwargs.get("tokenize") is False
    assert kwargs.get("add_generation_prompt") is True
    assert kwargs.get("enable_thinking") is True


def test_format_chat_prompt_fallback_without_template():
    tokenizer = _TokenizerWithoutTemplate()
    prompt = "raw prompt"
    assert _format_chat_prompt(tokenizer, prompt) == prompt


def test_validate_zimage_model_path_requires_components(tmp_path):
    model_root = tmp_path / "zimage"
    model_root.mkdir()
    # Incomplete layout should fail.
    with pytest.raises(FileNotFoundError):
        _validate_zimage_model_path(str(model_root))


def test_validate_zimage_model_path_accepts_complete_layout(tmp_path):
    model_root = tmp_path / "zimage"
    model_root.mkdir()

    (model_root / "scheduler").mkdir()
    (model_root / "tokenizer").mkdir()
    (model_root / "text_encoder").mkdir()
    (model_root / "vae").mkdir()
    (model_root / "transformer").mkdir()

    (model_root / "scheduler" / "scheduler_config.json").write_text("{}")
    (model_root / "tokenizer" / "tokenizer_config.json").write_text("{}")
    (model_root / "text_encoder" / "config.json").write_text("{}")
    (model_root / "vae" / "config.json").write_text("{}")
    (model_root / "transformer" / "config.json").write_text("{}")

    resolved = _validate_zimage_model_path(str(model_root))
    assert resolved == model_root


def test_encode_prompt_features_applies_mask():
    class DummyTokenizer:
        def __call__(self, prompts, **kwargs):
            assert prompts == ["chat prompt"]
            assert kwargs.get("max_length") == 512
            return SimpleNamespace(
                input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
            )

    class DummyTextEncoder:
        def __init__(self) -> None:
            self.device = torch.device("cpu")

        def to(self, device):
            self.device = torch.device(device)
            return self

        def __call__(self, *, input_ids, attention_mask, output_hidden_states, return_dict):
            del input_ids, attention_mask, output_hidden_states, return_dict
            hidden = torch.tensor(
                [
                    [
                        [1.0, 1.5],
                        [2.0, 2.5],
                        [3.0, 3.5],
                        [4.0, 4.5],
                    ]
                ]
            )
            return SimpleNamespace(
                hidden_states=(hidden - 10, hidden, hidden + 10),
                last_hidden_state=hidden + 100,
            )

    pipeline = SimpleNamespace(tokenizer=DummyTokenizer(), text_encoder=DummyTextEncoder())
    model = ZImageModel()

    # Bypass chat-template behavior for deterministic expectations in this unit test.
    prompt_embeds, pooled, mask = model.encode_prompt_features(
        SimpleNamespace(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder),
        prompt="chat prompt",
        device=torch.device("cpu"),
    )

    assert pooled is None
    assert mask is None
    assert prompt_embeds.shape == (2, 2)
    assert torch.equal(prompt_embeds, torch.tensor([[1.0, 1.5], [3.0, 3.5]]))
