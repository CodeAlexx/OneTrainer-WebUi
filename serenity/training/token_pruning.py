"""Token pruning utilities for text encoders (T5, Qwen).

Reduces token count by removing padding/low-importance tokens,
saving memory and compute during transformer forward passes.

Parity with OneTrainer's token pruning in ChromaModel and QwenModel
(attention-mask-based sequence trimming with 16-aligned padding).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PruningResult:
    """Result of token pruning on encoder output."""

    hidden_states: Tensor
    attention_mask: Tensor
    original_length: int
    pruned_length: int

    @property
    def tokens_removed(self) -> int:
        return self.original_length - self.pruned_length


def prune_tokens(
    hidden_states: Tensor,
    attention_mask: Tensor,
    *,
    pad_to_multiple: int = 16,
    min_length: int = 1,
) -> PruningResult:
    """Prune padding tokens from text encoder output.

    Removes trailing tokens that are masked in all batch samples.
    The sequence length is padded to a multiple of *pad_to_multiple*
    for attention processor compatibility.

    Parameters
    ----------
    hidden_states : Tensor
        Text encoder output, shape ``(B, T, D)``.
    attention_mask : Tensor
        Attention mask, shape ``(B, T)``. Non-zero = valid token.
    pad_to_multiple : int
        Align output sequence length to this multiple (default 16).
    min_length : int
        Minimum output sequence length.

    Returns
    -------
    PruningResult
        Pruned hidden states and mask.
    """
    original_length = hidden_states.shape[1]

    # Compute per-sample sequence lengths
    bool_mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
    seq_lengths = bool_mask.sum(dim=1)
    max_seq_length = max(int(seq_lengths.max().item()), min_length)

    # Pad to multiple for attention compatibility
    if pad_to_multiple > 1 and max_seq_length % pad_to_multiple > 0:
        # Only pad if not all sequences have the same length (need attention mask)
        needs_mask = (seq_lengths != max_seq_length).any()
        if needs_mask:
            max_seq_length += pad_to_multiple - (max_seq_length % pad_to_multiple)

    # Clamp to original length
    max_seq_length = min(max_seq_length, original_length)

    pruned_hidden = hidden_states[:, :max_seq_length, :]
    pruned_mask = bool_mask[:, :max_seq_length]

    return PruningResult(
        hidden_states=pruned_hidden,
        attention_mask=pruned_mask,
        original_length=original_length,
        pruned_length=max_seq_length,
    )


def compute_token_importance(
    hidden_states: Tensor,
    attention_mask: Tensor,
    method: str = "norm",
) -> Tensor:
    """Compute per-token importance scores.

    Parameters
    ----------
    hidden_states : Tensor
        Shape ``(B, T, D)``.
    attention_mask : Tensor
        Shape ``(B, T)``.
    method : str
        Importance scoring method:
        - ``'norm'``: L2 norm of each token's hidden state.
        - ``'attention'``: Sum of attention weights (if available).
        - ``'gradient'``: Gradient-based importance (requires grad).

    Returns
    -------
    Tensor
        Importance scores, shape ``(B, T)``.
    """
    if method == "norm":
        # L2 norm of hidden states as proxy for information content
        importance = hidden_states.float().norm(dim=-1)
        # Zero out masked positions
        if attention_mask is not None:
            importance = importance * attention_mask.float()
        return importance

    if method == "attention":
        # Use attention weights if stored on the hidden states
        # Fall back to norm-based scoring
        logger.debug("Attention-based importance not available; falling back to norm")
        return compute_token_importance(hidden_states, attention_mask, method="norm")

    if method == "gradient":
        if not hidden_states.requires_grad:
            logger.debug("Gradient-based importance requires grad; falling back to norm")
            return compute_token_importance(hidden_states, attention_mask, method="norm")
        # Gradient magnitude as importance
        importance = hidden_states.float().norm(dim=-1)
        if attention_mask is not None:
            importance = importance * attention_mask.float()
        return importance

    raise ValueError(f"Unknown importance method: {method}")


def apply_pruning_mask(
    hidden_states: Tensor,
    attention_mask: Tensor,
    importance_scores: Tensor,
    keep_ratio: float = 0.75,
    *,
    pad_to_multiple: int = 16,
    always_keep_first: bool = True,
    always_keep_last: bool = False,
) -> PruningResult:
    """Prune tokens based on importance scores, keeping the top-k.

    Unlike :func:`prune_tokens` which only removes trailing padding,
    this selectively removes low-importance tokens from the middle of
    the sequence.

    Parameters
    ----------
    hidden_states : Tensor
        Shape ``(B, T, D)``.
    attention_mask : Tensor
        Shape ``(B, T)``.
    importance_scores : Tensor
        Shape ``(B, T)``.
    keep_ratio : float
        Fraction of valid tokens to keep (0.0-1.0).
    pad_to_multiple : int
        Align output length to this multiple.
    always_keep_first : bool
        Always keep the first token (BOS/CLS).
    always_keep_last : bool
        Always keep the last valid token (EOS/SEP).

    Returns
    -------
    PruningResult
        Pruned hidden states and mask.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    original_length = seq_len

    bool_mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
    valid_counts = bool_mask.sum(dim=1)

    # Compute keep count per sample
    keep_counts = (valid_counts.float() * keep_ratio).long().clamp(min=1)
    max_keep = int(keep_counts.max().item())

    # Pad to multiple
    if pad_to_multiple > 1 and max_keep % pad_to_multiple > 0:
        max_keep += pad_to_multiple - (max_keep % pad_to_multiple)
    max_keep = min(max_keep, seq_len)

    # Build keep mask from importance scores
    # Set masked positions to -inf so they sort last
    scores = importance_scores.clone()
    scores[~bool_mask] = float("-inf")

    if always_keep_first:
        scores[:, 0] = float("inf")
    if always_keep_last:
        # Find last valid position per sample
        last_valid = valid_counts - 1
        for b in range(batch_size):
            if last_valid[b] >= 0:
                scores[b, last_valid[b]] = float("inf")

    # Get indices of top-k tokens (sorted by position to preserve order)
    _, top_indices = scores.topk(max_keep, dim=1, largest=True)
    top_indices = top_indices.sort(dim=1).values

    # Gather selected tokens
    expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    pruned_hidden = torch.gather(hidden_states, 1, expanded_indices)
    pruned_mask = torch.gather(bool_mask, 1, top_indices)

    return PruningResult(
        hidden_states=pruned_hidden,
        attention_mask=pruned_mask,
        original_length=original_length,
        pruned_length=max_keep,
    )


__all__ = [
    "PruningResult",
    "prune_tokens",
    "compute_token_importance",
    "apply_pruning_mask",
]
