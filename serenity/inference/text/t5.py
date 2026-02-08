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
        """Tokenize and encode *text*, returning hidden states.

        Args:
            text: Input prompt string.
            max_length: Maximum token length (T5-XXL default is 512).

        Returns:
            :class:`TextOutput` with hidden states. ``pooled_output`` is
            always ``None`` for T5 (encoder-only, no pooling head).
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("T5Encoder is not loaded. Call load() first.")

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
