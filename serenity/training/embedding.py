"""Textual Inversion / Embedding training utilities.

Supports embedding training for SD1.5, SDXL, and other architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for a single embedding (textual inversion) token."""

    placeholder: str
    num_vectors: int = 1
    initializer_token: str | None = None
    is_output_embedding: bool = False
    train: bool = True


@dataclass
class EmbeddingData:
    """Container for a trained embedding's state."""

    placeholder: str
    vectors: Tensor
    is_output_embedding: bool = False

    @property
    def num_vectors(self) -> int:
        return self.vectors.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.vectors.shape[1]


def create_embedding(
    tokenizer: Any,
    text_encoder: nn.Module,
    config: EmbeddingConfig,
    encode_fn: Any | None = None,
) -> EmbeddingData:
    """Create a new embedding for textual inversion training.

    If *config.initializer_token* is provided, the embedding is
    initialized from that token's existing embedding vector.
    Otherwise, the embedding is initialized from a text encoding pass
    via *encode_fn*, or random initialization as fallback.
    """
    embed_module = text_encoder.get_input_embeddings()
    embedding_dim = embed_module.weight.shape[1]
    dtype = embed_module.weight.dtype

    if config.initializer_token is not None:
        # Initialize from existing token
        init_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
        if not init_ids:
            logger.warning(
                "Initializer token '%s' not found; using random init",
                config.initializer_token,
            )
            vectors = torch.randn(config.num_vectors, embedding_dim, dtype=dtype) * 0.01
        else:
            init_embeds = embed_module.weight[init_ids].detach().clone()
            if init_embeds.shape[0] >= config.num_vectors:
                vectors = init_embeds[: config.num_vectors]
            else:
                # Repeat to fill
                repeats = (config.num_vectors + init_embeds.shape[0] - 1) // init_embeds.shape[0]
                vectors = init_embeds.repeat(repeats, 1)[: config.num_vectors]
    elif encode_fn is not None:
        # Initialize from text encoder output
        try:
            with torch.no_grad():
                encoded = encode_fn(config.placeholder)
                if encoded.ndim == 1:
                    encoded = encoded.unsqueeze(0)
                if encoded.shape[0] >= config.num_vectors:
                    vectors = encoded[: config.num_vectors].detach().clone()
                else:
                    vectors = torch.randn(config.num_vectors, embedding_dim, dtype=dtype) * 0.01
        except (RuntimeError, ValueError):
            logger.warning("encode_fn failed; using random init for '%s'", config.placeholder)
            vectors = torch.randn(config.num_vectors, embedding_dim, dtype=dtype) * 0.01
    else:
        vectors = torch.randn(config.num_vectors, embedding_dim, dtype=dtype) * 0.01

    vectors = vectors.to(dtype=dtype)
    return EmbeddingData(
        placeholder=config.placeholder,
        vectors=vectors,
        is_output_embedding=config.is_output_embedding,
    )


def add_embedding_to_tokenizer(
    tokenizer: Any,
    embedding: EmbeddingData,
) -> list[int]:
    """Register embedding placeholder tokens in the tokenizer.

    For multi-vector embeddings, adds tokens like ``<placeholder>``,
    ``<placeholder>_1``, ``<placeholder>_2``, etc.

    Returns the new token IDs.
    """
    if embedding.num_vectors == 1:
        new_tokens = [embedding.placeholder]
    else:
        new_tokens = [embedding.placeholder] + [
            f"{embedding.placeholder}_{i}" for i in range(1, embedding.num_vectors)
        ]

    tokenizer.add_tokens(new_tokens)
    token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    return token_ids


def resize_text_encoder_embeddings(
    text_encoder: nn.Module,
    tokenizer: Any,
) -> None:
    """Resize the text encoder's input embeddings to match the tokenizer."""
    if hasattr(text_encoder, "resize_token_embeddings"):
        text_encoder.resize_token_embeddings(len(tokenizer))
    else:
        embed = text_encoder.get_input_embeddings()
        if embed is not None and embed.num_embeddings < len(tokenizer):
            new_embed = nn.Embedding(len(tokenizer), embed.embedding_dim, dtype=embed.weight.dtype)
            new_embed.weight.data[: embed.num_embeddings] = embed.weight.data
            text_encoder.set_input_embeddings(new_embed)


def inject_embedding_vectors(
    text_encoder: nn.Module,
    token_ids: list[int],
    embedding: EmbeddingData,
) -> None:
    """Write embedding vectors into the text encoder's embedding layer."""
    embed_module = text_encoder.get_input_embeddings()
    with torch.no_grad():
        for i, tid in enumerate(token_ids):
            embed_module.weight[tid] = embedding.vectors[i].to(
                dtype=embed_module.weight.dtype,
                device=embed_module.weight.device,
            )


def setup_embedding_training(
    tokenizer: Any,
    text_encoder: nn.Module,
    embedding: EmbeddingData,
) -> list[int]:
    """Full setup: register tokens, resize embeddings, inject vectors.

    Returns the token IDs for the new embedding.
    """
    token_ids = add_embedding_to_tokenizer(tokenizer, embedding)
    resize_text_encoder_embeddings(text_encoder, tokenizer)
    inject_embedding_vectors(text_encoder, token_ids, embedding)
    return token_ids


def train_embedding_step(
    text_encoder: nn.Module,
    token_ids: list[int],
    embedding: EmbeddingData,
) -> None:
    """Synchronize trainable embedding vectors back from the text encoder.

    Call this after each optimizer step to keep the EmbeddingData in sync.
    """
    embed_module = text_encoder.get_input_embeddings()
    for i, tid in enumerate(token_ids):
        embedding.vectors[i] = embed_module.weight[tid].detach().clone()


def get_trainable_embedding_params(
    text_encoder: nn.Module,
    token_ids: list[int],
) -> list[Tensor]:
    """Return the embedding weight slices that should be optimized.

    This enables selective gradient computation on only the new tokens.
    """
    embed_module = text_encoder.get_input_embeddings()
    # We need the full embedding weight as the optimizable parameter;
    # the mask restricts gradients to only the new token indices.
    return [embed_module.weight]


def create_embedding_mask(
    text_encoder: nn.Module,
    token_ids: list[int],
) -> Tensor:
    """Create a boolean mask for the trainable embedding indices."""
    embed_module = text_encoder.get_input_embeddings()
    mask = torch.zeros(embed_module.num_embeddings, dtype=torch.bool)
    for tid in token_ids:
        mask[tid] = True
    return mask


def save_embedding(
    embedding: EmbeddingData,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save embedding to a safetensors or torch file."""
    path = Path(path)
    state = {"emb_params": embedding.vectors}

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
            meta = metadata or {}
            meta.setdefault("placeholder", embedding.placeholder)
            meta.setdefault("num_vectors", str(embedding.num_vectors))
            save_file(state, str(path), metadata=meta)
        except ImportError:
            torch.save(state, str(path))
    else:
        torch.save(state, str(path))

    logger.info(
        "Saved embedding '%s' (%d vectors, dim=%d) to %s",
        embedding.placeholder,
        embedding.num_vectors,
        embedding.embedding_dim,
        path,
    )


def load_embedding(
    path: str | Path,
    placeholder: str | None = None,
) -> EmbeddingData:
    """Load an embedding from a safetensors or torch file."""
    path = Path(path)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            state = load_file(str(path))
        except ImportError:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)

    # Try common key names
    vectors = None
    for key in ("emb_params", "emp_params", "emp_params_out", "string_to_param", "*"):
        if key == "*":
            # Take the first tensor
            for v in state.values():
                if isinstance(v, Tensor) and v.ndim == 2:
                    vectors = v
                    break
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, Tensor) and vv.ndim == 2:
                            vectors = vv
                            break
            break
        if key in state:
            v = state[key]
            if isinstance(v, Tensor):
                vectors = v
                break
            elif isinstance(v, dict):
                # string_to_param style
                for vv in v.values():
                    if isinstance(vv, Tensor):
                        vectors = vv
                        break
                break

    if vectors is None:
        raise ValueError(f"Could not find embedding vectors in {path}")

    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)

    name = placeholder or path.stem
    return EmbeddingData(placeholder=name, vectors=vectors)


__all__ = [
    "EmbeddingConfig",
    "EmbeddingData",
    "create_embedding",
    "add_embedding_to_tokenizer",
    "resize_text_encoder_embeddings",
    "inject_embedding_vectors",
    "setup_embedding_training",
    "train_embedding_step",
    "get_trainable_embedding_params",
    "create_embedding_mask",
    "save_embedding",
    "load_embedding",
]
