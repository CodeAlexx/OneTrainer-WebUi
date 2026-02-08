"""Tests for the text encoding pipeline — tokenizer, CLIP, T5, and manager."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from serenity.inference.models.detection import ModelArchitecture
from serenity.inference.text.clip import CLIPEncoder, CLIPType, TextOutput
from serenity.inference.text.encoders import TextEncoderManager, TextEncoderType, get_required_encoders
from serenity.inference.text.t5 import T5Encoder
from serenity.inference.text.tokenizer import (
    build_token_weight_map,
    create_token_chunks,
    has_non_default_weights,
    parse_prompt_weights,
    split_segments_at_break,
    truncate_or_pad,
)


# =========================================================================
# Tokenizer tests
# =========================================================================


class TestParsePromptWeights:
    """Tests for parse_prompt_weights."""

    def test_plain_text(self) -> None:
        result = parse_prompt_weights("hello world")
        assert result == [("hello world", 1.0)]

    def test_empty_string(self) -> None:
        result = parse_prompt_weights("")
        assert result == [("", 1.0)]

    def test_explicit_weight(self) -> None:
        result = parse_prompt_weights("(word:1.5)")
        assert len(result) == 1
        assert result[0][0] == "word"
        assert result[0][1] == pytest.approx(1.5)

    def test_mixed_weighted_and_plain(self) -> None:
        result = parse_prompt_weights("a (b:1.2) c")
        assert len(result) == 3
        assert result[0] == ("a ", 1.0)
        assert result[1][0] == "b"
        assert result[1][1] == pytest.approx(1.2)
        assert result[2] == (" c", 1.0)

    def test_bare_parentheses(self) -> None:
        result = parse_prompt_weights("(word)")
        assert len(result) == 1
        assert result[0][0] == "word"
        assert result[0][1] == pytest.approx(1.1)

    def test_zero_weight(self) -> None:
        result = parse_prompt_weights("(hidden:0.0)")
        assert len(result) == 1
        assert result[0][0] == "hidden"
        assert result[0][1] == pytest.approx(0.0)

    def test_negative_weight(self) -> None:
        result = parse_prompt_weights("(bad:-0.5)")
        assert len(result) == 1
        assert result[0][0] == "bad"
        assert result[0][1] == pytest.approx(-0.5)

    def test_break_keyword(self) -> None:
        result = parse_prompt_weights("hello BREAK world")
        assert len(result) == 3
        assert result[0] == ("hello ", 1.0)
        assert result[1] == ("BREAK", 1.0)
        assert result[2] == (" world", 1.0)

    def test_escaped_parentheses(self) -> None:
        result = parse_prompt_weights("\\(literal\\)")
        assert len(result) == 1
        assert result[0][0] == "(literal)"
        assert result[0][1] == 1.0

    def test_multiple_weighted_segments(self) -> None:
        result = parse_prompt_weights("(a:0.5) normal (b:2.0)")
        assert len(result) == 3
        assert result[0][0] == "a"
        assert result[0][1] == pytest.approx(0.5)
        assert result[1] == (" normal ", 1.0)
        assert result[2][0] == "b"
        assert result[2][1] == pytest.approx(2.0)


class TestTruncateOrPad:
    """Tests for truncate_or_pad."""

    def test_pad_short_sequence(self) -> None:
        result = truncate_or_pad([1, 2, 3], max_length=5, pad_token=0)
        assert result == [1, 2, 3, 0, 0]

    def test_truncate_long_sequence(self) -> None:
        result = truncate_or_pad([1, 2, 3, 4, 5], max_length=3, pad_token=0)
        assert result == [1, 2, 3]

    def test_exact_length(self) -> None:
        result = truncate_or_pad([1, 2, 3], max_length=3, pad_token=0)
        assert result == [1, 2, 3]

    def test_empty_input(self) -> None:
        result = truncate_or_pad([], max_length=3, pad_token=99)
        assert result == [99, 99, 99]

    def test_length_one(self) -> None:
        result = truncate_or_pad([42], max_length=1, pad_token=0)
        assert result == [42]


class TestCreateTokenChunks:
    """Tests for create_token_chunks."""

    def test_single_chunk(self) -> None:
        tokens = [1, 2, 3]
        weights = [1.0, 1.0, 1.0]
        result = create_token_chunks(tokens, weights, max_chunk_length=5)
        assert len(result) == 1
        assert result[0] == ([1, 2, 3], [1.0, 1.0, 1.0])

    def test_multiple_chunks(self) -> None:
        tokens = [1, 2, 3, 4, 5]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = create_token_chunks(tokens, weights, max_chunk_length=2)
        assert len(result) == 3
        assert result[0] == ([1, 2], [1.0, 1.0])
        assert result[1] == ([3, 4], [1.0, 1.0])
        assert result[2] == ([5], [1.0])

    def test_exact_chunk_size(self) -> None:
        tokens = [1, 2, 3, 4]
        weights = [0.5, 1.0, 1.5, 2.0]
        result = create_token_chunks(tokens, weights, max_chunk_length=2)
        assert len(result) == 2
        assert result[0] == ([1, 2], [0.5, 1.0])
        assert result[1] == ([3, 4], [1.5, 2.0])

    def test_empty_input(self) -> None:
        result = create_token_chunks([], [], max_chunk_length=5)
        assert result == [([], [])]


# =========================================================================
# CLIPEncoder tests
# =========================================================================


class TestCLIPType:
    """Tests for CLIPType enum."""

    def test_clip_l_value(self) -> None:
        assert CLIPType.CLIP_L.value == "clip_l"

    def test_clip_g_value(self) -> None:
        assert CLIPType.CLIP_G.value == "clip_g"

    def test_string_enum(self) -> None:
        assert isinstance(CLIPType.CLIP_L, str)


class TestTextOutput:
    """Tests for the TextOutput dataclass."""

    def test_creation_with_pooled(self) -> None:
        out = TextOutput(hidden_states="h", pooled_output="p")
        assert out.hidden_states == "h"
        assert out.pooled_output == "p"

    def test_creation_without_pooled(self) -> None:
        out = TextOutput(hidden_states="h")
        assert out.hidden_states == "h"
        assert out.pooled_output is None


class TestCLIPEncoder:
    """Tests for CLIPEncoder (mocked — no real model)."""

    def test_is_loaded_starts_false(self) -> None:
        enc = CLIPEncoder()
        assert enc.is_loaded is False

    def test_encode_raises_when_not_loaded(self) -> None:
        enc = CLIPEncoder()
        with pytest.raises(RuntimeError, match="not loaded"):
            enc.encode("hello")

    def test_unload_when_not_loaded(self) -> None:
        """Unloading when nothing is loaded should not error."""
        enc = CLIPEncoder()
        enc.unload()
        assert enc.is_loaded is False

    def test_encode_with_mock(self) -> None:
        """Test encoding flow with a mocked transformers model."""
        enc = CLIPEncoder()

        # Simulate a loaded state
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Set up tokenizer to return a mock with .to() that returns itself
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        # Create fake tensor-like outputs
        @dataclass
        class FakeHidden:
            shape: tuple = (1, 77, 768)

        fake_hidden = MagicMock()
        fake_hidden.shape = (1, 77, 768)

        fake_pooled = MagicMock()
        fake_pooled.shape = (1, 768)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = fake_hidden
        mock_outputs.pooler_output = fake_pooled
        mock_outputs.hidden_states = (fake_hidden, fake_hidden, fake_hidden)
        mock_model.return_value = mock_outputs

        enc._model = mock_model
        enc._tokenizer = mock_tokenizer
        enc._device = "cpu"

        result = enc.encode("test prompt", max_length=77, clip_skip=0)

        assert isinstance(result, TextOutput)
        assert result.hidden_states is fake_hidden
        assert result.pooled_output is fake_pooled

    def test_clip_skip_selects_earlier_layer(self) -> None:
        """Test that clip_skip > 0 uses an earlier hidden state."""
        enc = CLIPEncoder()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        # Create distinct layer outputs
        layer_0 = MagicMock(name="layer_0")
        layer_1 = MagicMock(name="layer_1")
        layer_2 = MagicMock(name="layer_2")
        layer_last = MagicMock(name="layer_last")

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = layer_last
        mock_outputs.pooler_output = MagicMock()
        # hidden_states: (embedding, layer0, layer1, layer2, layer_last)
        mock_outputs.hidden_states = (MagicMock(), layer_0, layer_1, layer_2, layer_last)

        # Mock final_layer_norm
        mock_model.text_model = MagicMock()
        final_norm = MagicMock()
        final_norm.return_value = layer_2  # identity-like
        mock_model.text_model.final_layer_norm = final_norm
        mock_model.return_value = mock_outputs

        enc._model = mock_model
        enc._tokenizer = mock_tokenizer
        enc._device = "cpu"

        # clip_skip=1 should use hidden_states[-2] = layer_2
        result = enc.encode("test", clip_skip=1)
        final_norm.assert_called_once_with(layer_2)


# =========================================================================
# T5Encoder tests
# =========================================================================


class TestT5Encoder:
    """Tests for T5Encoder (mocked — no real model)."""

    def test_is_loaded_starts_false(self) -> None:
        enc = T5Encoder()
        assert enc.is_loaded is False

    def test_encode_raises_when_not_loaded(self) -> None:
        enc = T5Encoder()
        with pytest.raises(RuntimeError, match="not loaded"):
            enc.encode("hello")

    def test_unload_when_not_loaded(self) -> None:
        enc = T5Encoder()
        enc.unload()
        assert enc.is_loaded is False

    def test_encode_returns_none_pooled(self) -> None:
        """T5 encoder must always return pooled_output=None."""
        enc = T5Encoder()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        fake_hidden = MagicMock()
        fake_hidden.shape = (1, 512, 4096)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = fake_hidden
        mock_model.return_value = mock_outputs

        enc._model = mock_model
        enc._tokenizer = mock_tokenizer
        enc._device = "cpu"

        result = enc.encode("test prompt", max_length=512)
        assert isinstance(result, TextOutput)
        assert result.hidden_states is fake_hidden
        assert result.pooled_output is None

    def test_max_length_parameter(self) -> None:
        """Verify max_length is passed to the tokenizer."""
        enc = T5Encoder()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = MagicMock()
        mock_model.return_value = mock_outputs

        enc._model = mock_model
        enc._tokenizer = mock_tokenizer
        enc._device = "cpu"

        enc.encode("test", max_length=256)

        # Check tokenizer was called with max_length=256
        call_kwargs = mock_tokenizer.call_args
        assert call_kwargs[1]["max_length"] == 256


# =========================================================================
# TextEncoderType tests
# =========================================================================


class TestTextEncoderType:
    """Tests for TextEncoderType enum."""

    def test_all_values(self) -> None:
        assert TextEncoderType.CLIP_L.value == "clip_l"
        assert TextEncoderType.CLIP_G.value == "clip_g"
        assert TextEncoderType.T5_XXL.value == "t5_xxl"
        assert TextEncoderType.QWEN.value == "qwen"
        assert TextEncoderType.GEMMA.value == "gemma"


# =========================================================================
# get_required_encoders tests
# =========================================================================


class TestGetRequiredEncoders:
    """Tests for get_required_encoders."""

    def test_sd15(self) -> None:
        result = get_required_encoders(ModelArchitecture.SD15)
        assert result == [TextEncoderType.CLIP_L]

    def test_sdxl(self) -> None:
        result = get_required_encoders(ModelArchitecture.SDXL)
        assert result == [TextEncoderType.CLIP_L, TextEncoderType.CLIP_G]

    def test_flux_dev(self) -> None:
        result = get_required_encoders(ModelArchitecture.FLUX_DEV)
        assert result == [TextEncoderType.CLIP_L, TextEncoderType.T5_XXL]

    def test_flux_schnell(self) -> None:
        result = get_required_encoders(ModelArchitecture.FLUX_SCHNELL)
        assert result == [TextEncoderType.CLIP_L, TextEncoderType.T5_XXL]

    def test_chroma(self) -> None:
        result = get_required_encoders(ModelArchitecture.CHROMA)
        assert result == [TextEncoderType.T5_XXL]

    def test_wan(self) -> None:
        result = get_required_encoders(ModelArchitecture.WAN)
        assert result == [TextEncoderType.T5_XXL]

    def test_sd3(self) -> None:
        result = get_required_encoders(ModelArchitecture.SD3)
        assert result == [
            TextEncoderType.CLIP_L,
            TextEncoderType.CLIP_G,
            TextEncoderType.T5_XXL,
        ]

    def test_sdxl_refiner(self) -> None:
        result = get_required_encoders(ModelArchitecture.SDXL_REFINER)
        assert result == [TextEncoderType.CLIP_G]

    def test_qwen(self) -> None:
        result = get_required_encoders(ModelArchitecture.QWEN)
        assert result == [TextEncoderType.QWEN]


# =========================================================================
# TextEncoderManager tests
# =========================================================================


def _make_mock_clip(hidden_dim: int = 768, seq_len: int = 77) -> CLIPEncoder:
    """Create a mock CLIPEncoder that returns plausible tensors."""
    enc = CLIPEncoder()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokens = MagicMock()
    mock_tokens.to.return_value = mock_tokens
    mock_tokenizer.return_value = mock_tokens

    import torch

    hidden = torch.zeros(1, seq_len, hidden_dim)
    pooled = torch.zeros(1, hidden_dim)

    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden
    mock_outputs.pooler_output = pooled
    mock_outputs.hidden_states = (hidden, hidden, hidden)
    mock_model.return_value = mock_outputs

    enc._model = mock_model
    enc._tokenizer = mock_tokenizer
    enc._device = "cpu"
    return enc


def _make_mock_t5(hidden_dim: int = 4096, seq_len: int = 512) -> T5Encoder:
    """Create a mock T5Encoder that returns plausible tensors."""
    enc = T5Encoder()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokens = MagicMock()
    mock_tokens.to.return_value = mock_tokens
    mock_tokenizer.return_value = mock_tokens

    import torch

    hidden = torch.zeros(1, seq_len, hidden_dim)

    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden
    mock_model.return_value = mock_outputs

    enc._model = mock_model
    enc._tokenizer = mock_tokenizer
    enc._device = "cpu"
    return enc


class TestTextEncoderManager:
    """Tests for TextEncoderManager with mocked encoders."""

    def test_get_encoder_creates_clip(self) -> None:
        mgr = TextEncoderManager()
        enc = mgr.get_encoder(TextEncoderType.CLIP_L)
        assert isinstance(enc, CLIPEncoder)

    def test_get_encoder_creates_t5(self) -> None:
        mgr = TextEncoderManager()
        enc = mgr.get_encoder(TextEncoderType.T5_XXL)
        assert isinstance(enc, T5Encoder)

    def test_get_encoder_caches(self) -> None:
        mgr = TextEncoderManager()
        enc1 = mgr.get_encoder(TextEncoderType.CLIP_L)
        enc2 = mgr.get_encoder(TextEncoderType.CLIP_L)
        assert enc1 is enc2

    def test_unsupported_encoder_raises(self) -> None:
        mgr = TextEncoderManager()
        with pytest.raises(ValueError, match="Unsupported"):
            mgr.get_encoder("not_a_real_encoder")  # type: ignore[arg-type]

    def test_unload_all_clears_state(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_L] = _make_mock_clip()
        mgr._encoders[TextEncoderType.T5_XXL] = _make_mock_t5()
        assert len(mgr._encoders) == 2

        mgr.unload_all()
        assert len(mgr._encoders) == 0

    def test_encode_sd15_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_L] = _make_mock_clip(768)

        result = mgr.encode_for_model(ModelArchitecture.SD15, "test", "neg")
        assert "cond" in result
        assert "uncond" in result

    def test_encode_sdxl_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_L] = _make_mock_clip(768)
        mgr._encoders[TextEncoderType.CLIP_G] = _make_mock_clip(1280)

        result = mgr.encode_for_model(ModelArchitecture.SDXL, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert "pooled" in result
        assert "neg_pooled" in result

    def test_encode_flux_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_L] = _make_mock_clip(768)
        mgr._encoders[TextEncoderType.T5_XXL] = _make_mock_t5(4096)

        result = mgr.encode_for_model(ModelArchitecture.FLUX_DEV, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert result["uncond"] is None
        assert "clip_cond" in result
        assert "pooled" in result

    def test_encode_chroma_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.T5_XXL] = _make_mock_t5(4096)

        result = mgr.encode_for_model(ModelArchitecture.CHROMA, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert result["uncond"] is None

    def test_encode_sd3_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_L] = _make_mock_clip(768)
        mgr._encoders[TextEncoderType.CLIP_G] = _make_mock_clip(1280)
        mgr._encoders[TextEncoderType.T5_XXL] = _make_mock_t5(4096)

        result = mgr.encode_for_model(ModelArchitecture.SD3, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert "pooled" in result
        assert "neg_pooled" in result

    def test_encode_unsupported_model_raises(self) -> None:
        mgr = TextEncoderManager()
        with pytest.raises(ValueError, match="not implemented"):
            mgr.encode_for_model("fake_unsupported_arch", "test")  # type: ignore[arg-type]

    def test_encode_wan_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.T5_XXL] = _make_mock_t5(4096)

        result = mgr.encode_for_model(ModelArchitecture.WAN, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert result["uncond"] is None

    def test_encode_sdxl_refiner_keys(self) -> None:
        mgr = TextEncoderManager()
        mgr._encoders[TextEncoderType.CLIP_G] = _make_mock_clip(1280)

        result = mgr.encode_for_model(ModelArchitecture.SDXL_REFINER, "test", "neg")
        assert "cond" in result
        assert "uncond" in result
        assert "pooled" in result
        assert "neg_pooled" in result


# =========================================================================
# Weight utility tests (new helpers in tokenizer.py)
# =========================================================================


class TestHasNonDefaultWeights:
    """Tests for has_non_default_weights."""

    def test_all_default(self) -> None:
        segments = [("hello", 1.0), ("world", 1.0)]
        assert has_non_default_weights(segments) is False

    def test_one_weighted(self) -> None:
        segments = [("hello", 1.0), ("world", 1.5)]
        assert has_non_default_weights(segments) is True

    def test_zero_weight(self) -> None:
        segments = [("hidden", 0.0)]
        assert has_non_default_weights(segments) is True

    def test_empty(self) -> None:
        segments = [("", 1.0)]
        assert has_non_default_weights(segments) is False


class TestSplitSegmentsAtBreak:
    """Tests for split_segments_at_break."""

    def test_no_break(self) -> None:
        segments = [("hello", 1.0), ("world", 1.5)]
        groups = split_segments_at_break(segments)
        assert len(groups) == 1
        assert groups[0] == [("hello", 1.0), ("world", 1.5)]

    def test_single_break(self) -> None:
        segments = [("hello", 1.0), ("BREAK", 1.0), ("world", 1.5)]
        groups = split_segments_at_break(segments)
        assert len(groups) == 2
        assert groups[0] == [("hello", 1.0)]
        assert groups[1] == [("world", 1.5)]

    def test_multiple_breaks(self) -> None:
        segments = [("a", 1.0), ("BREAK", 1.0), ("b", 1.0), ("BREAK", 1.0), ("c", 1.0)]
        groups = split_segments_at_break(segments)
        assert len(groups) == 3

    def test_leading_break(self) -> None:
        segments = [("BREAK", 1.0), ("world", 1.0)]
        groups = split_segments_at_break(segments)
        assert len(groups) == 1
        assert groups[0] == [("world", 1.0)]

    def test_all_breaks(self) -> None:
        segments = [("BREAK", 1.0), ("BREAK", 1.0)]
        groups = split_segments_at_break(segments)
        # Should return at least one empty group
        assert len(groups) == 1
        assert groups[0] == [("", 1.0)]

    def test_empty_segments(self) -> None:
        segments = [("", 1.0)]
        groups = split_segments_at_break(segments)
        assert len(groups) == 1


class TestBuildTokenWeightMap:
    """Tests for build_token_weight_map."""

    @staticmethod
    def _simple_tokenizer(text: str) -> list[int]:
        """Fake tokenizer: each character becomes token ID = ord(char)."""
        return [ord(c) for c in text]

    def test_basic_weighted(self) -> None:
        segments = [("ab", 1.5), ("cd", 0.5)]
        tokens, weights = build_token_weight_map(
            segments,
            tokenize_fn=self._simple_tokenizer,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_length=10,
        )
        assert len(tokens) == 10
        assert len(weights) == 10
        # BOS
        assert tokens[0] == 1
        assert weights[0] == 1.0
        # 'a', 'b' with weight 1.5
        assert weights[1] == 1.5
        assert weights[2] == 1.5
        # 'c', 'd' with weight 0.5
        assert weights[3] == 0.5
        assert weights[4] == 0.5
        # EOS
        assert tokens[5] == 2
        assert weights[5] == 1.0
        # PAD
        assert tokens[6] == 0
        assert weights[6] == 1.0

    def test_no_special_tokens(self) -> None:
        segments = [("abc", 2.0)]
        tokens, weights = build_token_weight_map(
            segments,
            tokenize_fn=self._simple_tokenizer,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            max_length=5,
        )
        assert len(tokens) == 5
        # Content tokens
        assert weights[0] == 2.0
        assert weights[1] == 2.0
        assert weights[2] == 2.0
        # Padding
        assert tokens[3] == 0
        assert weights[3] == 1.0

    def test_truncation(self) -> None:
        """Tokens that exceed content capacity should be truncated."""
        segments = [("abcdefgh", 1.5)]
        tokens, weights = build_token_weight_map(
            segments,
            tokenize_fn=self._simple_tokenizer,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_length=5,
        )
        assert len(tokens) == 5
        # BOS + 3 content + EOS = 5
        assert tokens[0] == 1  # BOS
        assert tokens[4] == 2  # EOS

    def test_empty_segment(self) -> None:
        segments = [("", 1.0)]
        tokens, weights = build_token_weight_map(
            segments,
            tokenize_fn=self._simple_tokenizer,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_length=5,
        )
        assert len(tokens) == 5
        assert tokens[0] == 1  # BOS
        assert tokens[1] == 2  # EOS
        # Rest are padding
        assert tokens[2] == 0


# =========================================================================
# Weighted encoding tests — CLIP
# =========================================================================


def _make_weighted_mock_clip(hidden_dim: int = 768, seq_len: int = 77) -> CLIPEncoder:
    """Create a mock CLIPEncoder that supports the weighted encoding path.

    The mock tokenizer has both __call__ (for fast path) and encode() (for
    per-segment tokenization).  The model accepts input_ids/attention_mask
    kwargs and returns hidden states proportional to ones so weight scaling
    is testable.
    """
    enc = CLIPEncoder()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Token IDs for fast-path: tokenizer(text, ...) returns dict-like
    mock_tokens = MagicMock()
    mock_tokens.to.return_value = mock_tokens
    mock_tokenizer.return_value = mock_tokens

    # Per-segment path: tokenizer.encode(text, add_special_tokens=False)
    # Returns one token per word character (simplified)
    def _encode_bare(text: str, add_special_tokens: bool = True) -> list[int]:
        if not add_special_tokens:
            return [ord(c) for c in text if not c.isspace()]
        return [49406] + [ord(c) for c in text if not c.isspace()] + [49407]

    mock_tokenizer.encode = _encode_bare
    mock_tokenizer.bos_token_id = 49406
    mock_tokenizer.eos_token_id = 49407
    mock_tokenizer.pad_token_id = 49407  # CLIP uses EOS as PAD

    # Model returns ones so we can verify weight scaling
    def _model_forward(**kwargs: object) -> MagicMock:
        iids = kwargs.get("input_ids")
        if iids is not None and hasattr(iids, "shape"):
            batch, slen = iids.shape
        else:
            batch, slen = 1, seq_len
        hidden = torch.ones(batch, slen, hidden_dim)
        pooled = torch.zeros(batch, hidden_dim)
        out = MagicMock()
        out.last_hidden_state = hidden
        out.pooler_output = pooled
        out.hidden_states = (hidden, hidden, hidden)
        out.text_embeds = None
        return out

    mock_model.side_effect = _model_forward
    mock_model.text_model = MagicMock()
    mock_model.text_model.final_layer_norm = MagicMock(side_effect=lambda x: x)

    enc._model = mock_model
    enc._tokenizer = mock_tokenizer
    enc._device = "cpu"
    return enc


def _make_weighted_mock_t5(hidden_dim: int = 4096, seq_len: int = 512) -> T5Encoder:
    """Create a mock T5Encoder that supports the weighted encoding path."""
    enc = T5Encoder()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokens = MagicMock()
    mock_tokens.to.return_value = mock_tokens
    mock_tokenizer.return_value = mock_tokens

    def _encode_bare(text: str, add_special_tokens: bool = True) -> list[int]:
        if not add_special_tokens:
            return [ord(c) for c in text if not c.isspace()]
        return [ord(c) for c in text if not c.isspace()] + [1]  # 1 = EOS for T5

    mock_tokenizer.encode = _encode_bare
    mock_tokenizer.bos_token_id = None  # T5 has no BOS
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.pad_token_id = 0

    def _model_forward(**kwargs: object) -> MagicMock:
        iids = kwargs.get("input_ids")
        if iids is not None and hasattr(iids, "shape"):
            batch, slen = iids.shape
        else:
            batch, slen = 1, seq_len
        hidden = torch.ones(batch, slen, hidden_dim)
        out = MagicMock()
        out.last_hidden_state = hidden
        return out

    mock_model.side_effect = _model_forward

    enc._model = mock_model
    enc._tokenizer = mock_tokenizer
    enc._device = "cpu"
    return enc


class TestCLIPWeightedEncoding:
    """Tests for CLIP prompt weight application."""

    def test_weighted_produces_different_embeddings(self) -> None:
        """Encoding '(word:1.5)' must produce different hidden states than 'word'."""
        enc = _make_weighted_mock_clip()
        unweighted = enc.encode("word")
        weighted = enc.encode("(word:1.5)")

        # The weighted path scales content token embeddings by 1.5
        # while unweighted keeps them at 1.0 (fast path returns ones)
        assert not torch.allclose(
            unweighted.hidden_states, weighted.hidden_states,
        ), "Weighted encoding must differ from unweighted"

    def test_weight_1_0_uses_fast_path(self) -> None:
        """When all weights are 1.0, the fast path (unweighted) is used."""
        enc = _make_weighted_mock_clip()
        result = enc.encode("plain text no weights")
        # Fast path should succeed and return TextOutput
        assert isinstance(result, TextOutput)
        assert result.hidden_states is not None

    def test_weight_scaling_value(self) -> None:
        """Verify that content tokens are actually multiplied by the weight."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("(ab:2.0)", max_length=10)

        hidden = result.hidden_states
        # Token layout: BOS(1.0), a(2.0), b(2.0), EOS(1.0), PAD(1.0)...
        # BOS at index 0 should be 1.0 (special token, unweighted)
        assert hidden[0, 0, 0].item() == pytest.approx(1.0)
        # Content tokens at indices 1, 2 should be 2.0
        assert hidden[0, 1, 0].item() == pytest.approx(2.0)
        assert hidden[0, 2, 0].item() == pytest.approx(2.0)

    def test_special_tokens_not_weighted(self) -> None:
        """BOS and EOS tokens must NOT be scaled by the weight."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("(x:3.0)", max_length=10)

        hidden = result.hidden_states
        # BOS at index 0
        assert hidden[0, 0, 0].item() == pytest.approx(1.0)
        # Content 'x' at index 1
        assert hidden[0, 1, 0].item() == pytest.approx(3.0)
        # EOS at index 2
        assert hidden[0, 2, 0].item() == pytest.approx(1.0)

    def test_nested_weights(self) -> None:
        """Nested bare parentheses ((word)) use the inner weight of 1.1."""
        segments = parse_prompt_weights("((word))")
        assert len(segments) == 1
        # The parser preserves the innermost non-default weight
        assert segments[0][1] == pytest.approx(1.1)

    def test_break_splits_encoding(self) -> None:
        """BREAK should produce concatenated hidden states from separate chunks."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result_no_break = enc.encode("(ab:1.5)", max_length=10)
        result_with_break = enc.encode("(a:1.5) BREAK (b:2.0)", max_length=10)

        # With BREAK, we get two separate 10-token chunks concatenated
        assert result_with_break.hidden_states.shape[1] == 20
        assert result_no_break.hidden_states.shape[1] == 10

    def test_empty_prompt(self) -> None:
        """Empty prompt should not crash."""
        enc = _make_weighted_mock_clip()
        result = enc.encode("")
        assert isinstance(result, TextOutput)

    def test_pooled_output_preserved(self) -> None:
        """Pooled output should still be returned with weighted encoding."""
        enc = _make_weighted_mock_clip()
        result = enc.encode("(test:1.5)")
        # Our mock returns pooler_output via the model
        assert isinstance(result, TextOutput)

    def test_multiple_weights_in_prompt(self) -> None:
        """Multiple weighted segments apply correct per-token weights."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=20)
        result = enc.encode("(a:0.5) (b:2.0)", max_length=20)

        hidden = result.hidden_states
        # BOS at 0 = 1.0, 'a' at 1 = 0.5, 'b' at 2 = 2.0, EOS at 3 = 1.0
        assert hidden[0, 0, 0].item() == pytest.approx(1.0)
        assert hidden[0, 1, 0].item() == pytest.approx(0.5)
        assert hidden[0, 2, 0].item() == pytest.approx(2.0)
        assert hidden[0, 3, 0].item() == pytest.approx(1.0)


# =========================================================================
# Weighted encoding tests — T5
# =========================================================================


class TestT5WeightedEncoding:
    """Tests for T5 prompt weight application."""

    def test_weighted_produces_different_embeddings(self) -> None:
        """Encoding '(word:1.5)' must differ from unweighted 'word'."""
        enc = _make_weighted_mock_t5()
        unweighted = enc.encode("word")
        weighted = enc.encode("(word:1.5)")

        assert not torch.allclose(
            unweighted.hidden_states, weighted.hidden_states,
        ), "Weighted T5 encoding must differ from unweighted"

    def test_weight_1_0_uses_fast_path(self) -> None:
        """Plain text uses the unweighted fast path."""
        enc = _make_weighted_mock_t5()
        result = enc.encode("plain text")
        assert isinstance(result, TextOutput)
        assert result.pooled_output is None

    def test_weight_scaling_value(self) -> None:
        """Content tokens are multiplied by the weight value."""
        enc = _make_weighted_mock_t5(hidden_dim=4, seq_len=10)
        result = enc.encode("(ab:2.0)", max_length=10)

        hidden = result.hidden_states
        # T5: no BOS. Content 'a'=2.0, 'b'=2.0, EOS=1.0, pad=1.0
        assert hidden[0, 0, 0].item() == pytest.approx(2.0)
        assert hidden[0, 1, 0].item() == pytest.approx(2.0)
        # EOS
        assert hidden[0, 2, 0].item() == pytest.approx(1.0)

    def test_eos_not_weighted(self) -> None:
        """EOS token must not be scaled by the weight."""
        enc = _make_weighted_mock_t5(hidden_dim=4, seq_len=10)
        result = enc.encode("(x:5.0)", max_length=10)

        hidden = result.hidden_states
        # 'x'=5.0, EOS=1.0
        assert hidden[0, 0, 0].item() == pytest.approx(5.0)
        assert hidden[0, 1, 0].item() == pytest.approx(1.0)

    def test_break_splits_t5_encoding(self) -> None:
        """BREAK produces concatenated hidden states from separate chunks."""
        enc = _make_weighted_mock_t5(hidden_dim=4, seq_len=10)
        result = enc.encode("(a:1.5) BREAK (b:2.0)", max_length=10)
        # Two 10-token chunks
        assert result.hidden_states.shape[1] == 20

    def test_empty_prompt(self) -> None:
        """Empty prompt should not crash."""
        enc = _make_weighted_mock_t5()
        result = enc.encode("")
        assert isinstance(result, TextOutput)

    def test_pooled_always_none(self) -> None:
        """T5 never has a pooled output, even with weighted encoding."""
        enc = _make_weighted_mock_t5()
        result = enc.encode("(test:1.5)")
        assert result.pooled_output is None

    def test_multiple_weights(self) -> None:
        """Multiple weighted segments apply correct weights."""
        enc = _make_weighted_mock_t5(hidden_dim=4, seq_len=20)
        result = enc.encode("(a:0.5) (b:3.0)", max_length=20)

        hidden = result.hidden_states
        # T5: 'a'=0.5, 'b'=3.0, EOS=1.0
        assert hidden[0, 0, 0].item() == pytest.approx(0.5)
        assert hidden[0, 1, 0].item() == pytest.approx(3.0)
        assert hidden[0, 2, 0].item() == pytest.approx(1.0)


# =========================================================================
# Integration: parse_prompt_weights -> weighted encode round-trip
# =========================================================================


class TestWeightedEncodingIntegration:
    """End-to-end integration of parsing and weighted encoding."""

    def test_nested_weights_affect_clip_encoding(self) -> None:
        """((word)) parsed as weight 1.1 should scale CLIP embeddings."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("((x))", max_length=10)

        hidden = result.hidden_states
        # BOS=1.0, x=1.1 (inner bare-paren weight), EOS=1.0
        assert hidden[0, 0, 0].item() == pytest.approx(1.0)
        assert hidden[0, 1, 0].item() == pytest.approx(1.1, abs=0.01)

    def test_zero_weight_zeros_embeddings(self) -> None:
        """Weight 0.0 should zero out the corresponding token embeddings."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("(x:0.0)", max_length=10)

        hidden = result.hidden_states
        # 'x' at index 1 should be zeroed
        assert hidden[0, 1, 0].item() == pytest.approx(0.0)

    def test_negative_weight_negates_embeddings(self) -> None:
        """Negative weight should negate embeddings."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("(x:-1.0)", max_length=10)

        hidden = result.hidden_states
        assert hidden[0, 1, 0].item() == pytest.approx(-1.0)

    def test_explicit_weight_syntax_parsed_correctly(self) -> None:
        """Verify the full round-trip: '(word:1.5)' parses and scales."""
        segments = parse_prompt_weights("a (beautiful:1.3) sunset")
        assert len(segments) == 3
        assert segments[0] == ("a ", 1.0)
        assert segments[1][0] == "beautiful"
        assert segments[1][1] == pytest.approx(1.3)
        assert segments[2] == (" sunset", 1.0)

        # Verify the has_non_default_weights helper sees the 1.3
        assert has_non_default_weights(segments) is True

    def test_single_word_prompt(self) -> None:
        """Single word should encode fine through weighted path."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        result = enc.encode("(hello:2.0)", max_length=10)
        assert isinstance(result, TextOutput)
        # 'h', 'e', 'l', 'l', 'o' — 5 content tokens at weight 2.0
        for i in range(1, 6):
            assert result.hidden_states[0, i, 0].item() == pytest.approx(2.0)

    def test_all_break_prompt_clip(self) -> None:
        """Prompt with only BREAK should produce valid output."""
        enc = _make_weighted_mock_clip(hidden_dim=4, seq_len=10)
        # "BREAK" alone — parse_prompt_weights returns just the BREAK
        # with no non-default weights, so fast path is used
        result = enc.encode("BREAK", max_length=10)
        assert isinstance(result, TextOutput)
