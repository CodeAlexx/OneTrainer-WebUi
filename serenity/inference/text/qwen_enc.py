"""Qwen 2.5 text encoder — causal LM hidden states for Qwen-based image gen."""

from __future__ import annotations

import logging
from typing import Any

from serenity.inference.text.clip import TextOutput

__all__ = [
    "QwenEncoder",
]

logger = logging.getLogger(__name__)


class QwenEncoder:
    """Load and run a Qwen 2.5 causal LM as a text encoder.

    Qwen 2.5 specifics:
      - Hidden dimension varies by model size (1536 for 1.5B, 3584 for 7B)
      - Default max tokens: 256
      - No pooled output (hidden states only)

    Used by: Qwen-based image generation models.

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
        """Load a Qwen causal LM and tokenizer from *model_path*."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load Qwen models. "
                "Install it with: pip install transformers"
            ) from exc

        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading Qwen encoder from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
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
        """Tokenize and encode *text*, applying prompt weights to embeddings.

        Parses ``(word:1.5)`` syntax via :func:`parse_prompt_weights` and
        scales per-token hidden states accordingly.  When all weights are
        1.0, the fast path (identical to unweighted encoding) is used.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (default 256).

        Returns:
            :class:`TextOutput` with hidden states. ``pooled_output`` is
            always ``None`` for Qwen (causal LM, no pooling head).
        """
        import torch

        from serenity.inference.text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("QwenEncoder is not loaded. Call load() first.")

        segments = parse_prompt_weights(text)

        if not has_non_default_weights(segments):
            return self._encode_unweighted(text, max_length)

        groups = split_segments_at_break(segments)

        all_hidden: list[Any] = []
        for group in groups:
            all_hidden.append(self._encode_weighted_group(group, max_length))

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
            outputs = self._model(
                **tokens,
                output_hidden_states=True,
            )

        return TextOutput(
            hidden_states=outputs.hidden_states[-1],
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

        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        def _tokenize_bare(text: str) -> list[int]:
            return self._tokenizer.encode(text, add_special_tokens=False)

        token_ids, weights = build_token_weight_map(
            group,
            tokenize_fn=_tokenize_bare,
            bos_token_id=bos_id,
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
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1]

        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)
        hidden = hidden * weight_tensor

        return hidden
