"""CLIP text encoder — supports CLIP-L and CLIP-G (OpenCLIP ViT-bigG)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "CLIPEncoder",
    "CLIPType",
    "TextOutput",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CLIPType(str, Enum):
    """Supported CLIP model variants."""

    CLIP_L = "clip_l"
    CLIP_G = "clip_g"


@dataclass
class TextOutput:
    """Output from a text encoder forward pass.

    Attributes:
        hidden_states: Encoder hidden states, shape ``(batch, seq_len, dim)``.
        pooled_output: Pooled representation, shape ``(batch, dim)``.
            ``None`` for encoders that do not produce a pooled output (e.g. T5).
    """

    hidden_states: Any  # torch.Tensor at runtime
    pooled_output: Any | None = None  # torch.Tensor | None


# ---------------------------------------------------------------------------
# CLIP hidden dimensions
# ---------------------------------------------------------------------------

_CLIP_DIMS: dict[CLIPType, int] = {
    CLIPType.CLIP_L: 768,
    CLIPType.CLIP_G: 1280,
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class CLIPEncoder:
    """Load and run a CLIP text encoder.

    Supports both CLIP-L (768-dim, used by SD15/SDXL/Flux) and
    CLIP-G (1280-dim OpenCLIP ViT-bigG, used by SDXL).

    All ``transformers`` imports are lazy so the module can be imported
    without the library installed.

    Args:
        model_path: HuggingFace repo or local path.  Loaded eagerly if given.
        dtype: Torch dtype for model weights.
        device: Target device string.
        use_projection: When ``True``, load ``CLIPTextModelWithProjection``
            instead of ``CLIPTextModel``.  Required for CLIP-G (SDXL).
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: Any = None,
        device: str = "cpu",
        use_projection: bool = False,
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        self._use_projection = use_projection
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(
        self,
        model_path: str,
        subfolder: str | None = None,
        tokenizer_subfolder: str | None = None,
    ) -> None:
        """Load a CLIP model and tokenizer from *model_path*.

        Args:
            model_path: HuggingFace repo ID or local directory.
            subfolder: Optional subfolder for the model weights
                (e.g. ``"text_encoder_2"`` for SDXL CLIP-G).
            tokenizer_subfolder: Optional subfolder for the tokenizer.
                Defaults to *subfolder* when not specified.
        """
        try:
            from transformers import CLIPTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load CLIP models. "
                "Install it with: pip install transformers"
            ) from exc

        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        tok_sf = tokenizer_subfolder or subfolder
        tok_kwargs: dict[str, str] = {}
        model_kwargs: dict[str, str] = {}
        if tok_sf:
            tok_kwargs["subfolder"] = tok_sf
        if subfolder:
            model_kwargs["subfolder"] = subfolder

        logger.info("Loading CLIP model from %s (subfolder=%s)", model_path, subfolder)
        self._tokenizer = CLIPTokenizer.from_pretrained(model_path, **tok_kwargs)

        if self._use_projection:
            from transformers import CLIPTextModelWithProjection  # type: ignore[import-untyped]

            self._model = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
                **model_kwargs,
            ).to(self._device)
        else:
            from transformers import CLIPTextModel  # type: ignore[import-untyped]

            self._model = CLIPTextModel.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
                **model_kwargs,
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
        max_length: int = 77,
        clip_skip: int = 0,
    ) -> TextOutput:
        """Tokenize and encode *text*, returning hidden states.

        Args:
            text: Input prompt string.
            max_length: Maximum token length (CLIP default is 77).
            clip_skip: Number of final encoder layers to skip.
                0 means use the last hidden state, 1 skips the last layer, etc.

        Returns:
            :class:`TextOutput` with hidden states and pooled output.
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("CLIPEncoder is not loaded. Call load() first.")

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

        # Apply clip_skip — index from the back of hidden_states
        # hidden_states is a tuple with (embedding, layer0, layer1, ..., layerN)
        # Default (clip_skip=0) uses the last hidden state
        if clip_skip > 0 and len(outputs.hidden_states) > clip_skip:
            hidden = outputs.hidden_states[-(clip_skip + 1)]
            # Apply final layer norm if available
            if hasattr(self._model.text_model, "final_layer_norm"):
                hidden = self._model.text_model.final_layer_norm(hidden)
        else:
            hidden = outputs.last_hidden_state

        # CLIPTextModelWithProjection → text_embeds; CLIPTextModel → pooler_output
        if self._use_projection:
            pooled = getattr(outputs, "text_embeds", None)
        else:
            pooled = getattr(outputs, "pooler_output", None)

        return TextOutput(
            hidden_states=hidden,
            pooled_output=pooled,
        )
