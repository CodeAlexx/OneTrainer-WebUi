"""Gemma 2 text encoder — hidden states for Lumina and ZImage models."""

from __future__ import annotations

import logging
from typing import Any

from serenity.inference.text.clip import TextOutput

__all__ = [
    "GemmaEncoder",
]

logger = logging.getLogger(__name__)


class GemmaEncoder:
    """Load and run a Gemma 2 model as a text encoder.

    Gemma 2 specifics:
      - Hidden dimension depends on model size (2304 for 2B, 3584 for 9B)
      - Default max tokens: 256
      - No pooled output (hidden states only)

    Used by: Lumina, ZImage.

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
        """Load a Gemma model and tokenizer from *model_path*."""
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load Gemma models. "
                "Install it with: pip install transformers"
            ) from exc

        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading Gemma encoder from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(
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
        max_length: int = 256,
    ) -> TextOutput:
        """Tokenize and encode *text*, returning hidden states.

        Args:
            text: Input prompt string.
            max_length: Maximum token length (default 256).

        Returns:
            :class:`TextOutput` with hidden states. ``pooled_output`` is
            always ``None`` for Gemma (no pooling head).
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("GemmaEncoder is not loaded. Call load() first.")

        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(
                **tokens,
                output_hidden_states=True,
            )

        return TextOutput(
            hidden_states=outputs.last_hidden_state,
            pooled_output=None,
        )
