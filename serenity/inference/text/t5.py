"""T5 text encoder — T5-XXL for Flux, Chroma, Wan, and SD3."""

from __future__ import annotations

import logging
from typing import Any

from serenity.inference.text.clip import TextOutput

__all__ = [
    "T5Encoder",
]

logger = logging.getLogger(__name__)


class T5Encoder:
    """Load and run a T5 encoder model (typically T5-XXL).

    T5-XXL specifics:
      - Hidden dimension: 4096
      - Default max tokens: 512
      - No pooled output (encoder-only usage)

    Used by: Flux, Chroma, Wan, SD3.

    All ``transformers`` imports are lazy so the module can be imported
    without the library installed.
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a T5 encoder model and tokenizer from *model_path*."""
        try:
            from transformers import AutoTokenizer, T5EncoderModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load T5 models. "
                "Install it with: pip install transformers"
            ) from exc

        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading T5 encoder from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = T5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

    def unload(self) -> None:
        """Release model and tokenizer, freeing memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """``True`` if a model is currently loaded."""
        return self._model is not None

    # -- encoding ------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: int = 512,
    ) -> TextOutput:
        """Tokenize and encode *text*, applying prompt weights to embeddings.

        Parses ``(word:1.5)`` syntax via :func:`parse_prompt_weights` and
        scales per-token hidden states accordingly.  When all weights are
        1.0, the fast path (identical to unweighted encoding) is used.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (T5-XXL default is 512).

        Returns:
            :class:`TextOutput` with hidden states. ``pooled_output`` is
            always ``None`` for T5 (encoder-only, no pooling head).
        """
        import torch

        from serenity.inference.text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("T5Encoder is not loaded. Call load() first.")

        segments = parse_prompt_weights(text)

        # Fast path — no weighting needed
        if not has_non_default_weights(segments):
            return self._encode_unweighted(text, max_length)

        # Split at BREAK boundaries
        groups = split_segments_at_break(segments)

        all_hidden: list[Any] = []
        for group in groups:
            hidden_chunk = self._encode_weighted_group(group, max_length)
            all_hidden.append(hidden_chunk)

        if len(all_hidden) == 1:
            hidden = all_hidden[0]
        else:
            hidden = torch.cat(all_hidden, dim=1)

        return TextOutput(hidden_states=hidden, pooled_output=None)

    def _encode_unweighted(self, text: str, max_length: int) -> TextOutput:
        """Original fast-path encoding with no weight application."""
        import torch

        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**tokens)

        return TextOutput(
            hidden_states=outputs.last_hidden_state,
            pooled_output=None,
        )

    def _encode_weighted_group(
        self,
        group: list[tuple[str, float]],
        max_length: int,
    ) -> Any:
        """Encode a single BREAK-group with per-token weight scaling."""
        import torch

        from serenity.inference.text.tokenizer import build_token_weight_map

        # T5 uses EOS but no BOS; pad token varies
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # T5 typically uses 0 as pad

        def _tokenize_bare(text: str) -> list[int]:
            """Tokenize text without special tokens."""
            return self._tokenizer.encode(text, add_special_tokens=False)

        token_ids, weights = build_token_weight_map(
            group,
            tokenize_fn=_tokenize_bare,
            bos_token_id=None,  # T5 has no BOS
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            max_length=max_length,
        )

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(
            [[1 if t != pad_id else 0 for t in token_ids]],
            dtype=torch.long,
            device=self._device,
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state

        # Apply per-token weights
        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        hidden = hidden * weight_tensor

        return hidden
