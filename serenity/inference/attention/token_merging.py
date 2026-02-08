"""Token merging (ToME) for attention speedup."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["merge_tokens", "unmerge_tokens", "bipartite_soft_matching"]


def bipartite_soft_matching(metric: Tensor, ratio: float) -> tuple[Tensor, Tensor] | tuple[None, None]:
    """Compute bipartite soft matching indices.

    Splits tokens into source (even) and destination (odd) sets,
    finds most similar pairs via dot product, merges top-r pairs.

    Args:
        metric: Token features (B, N, C) for similarity computation.
        ratio: Fraction of tokens to merge (0.0 to 1.0).

    Returns:
        Tuple of (merge_indices, unmerge_indices) for gather/scatter
        operations, or ``(None, None)`` when no merging is needed.
    """
    B, N, C = metric.shape
    r = int(N * ratio / 2)  # Number of tokens to merge
    if r <= 0:
        return None, None

    # Split into source (even indices) and destination (odd indices)
    src = metric[:, ::2]   # (B, src_len, C)
    dst = metric[:, 1::2]  # (B, dst_len, C)

    # Normalize for cosine similarity
    src_norm = src / src.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dst_norm = dst / dst.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Similarity matrix
    scores = torch.bmm(src_norm, dst_norm.transpose(1, 2))  # (B, src_len, dst_len)

    # Find best match for each source token
    node_max, node_idx = scores.max(dim=-1)  # (B, src_len)

    # Select top-r most similar pairs to merge
    _, edge_idx = node_max.topk(r, dim=-1)  # (B, r) -- indices into src

    # Return indices for gather-based merge and scatter-based unmerge
    return edge_idx, node_idx


def merge_tokens(
    x: Tensor,
    merge_info: tuple[Tensor, Tensor] | tuple[None, None] | None,
    mode: str = "mean",
) -> Tensor:
    """Merge tokens according to bipartite matching.

    Args:
        x: Input tokens (B, N, C).
        merge_info: ``(edge_idx, node_idx)`` from
            :func:`bipartite_soft_matching`. ``None`` is a no-op.
        mode: Merge strategy -- ``"mean"`` averages matched pairs.
    """
    if merge_info is None or merge_info[0] is None:
        return x

    edge_idx, node_idx = merge_info
    B, N, C = x.shape
    r = edge_idx.shape[1]

    src = x[:, ::2]   # (B, src_len, C)
    dst = x[:, 1::2]  # (B, dst_len, C)

    # Gather the source tokens being merged
    merged_src = src.gather(1, edge_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, r, C)
    # Find their destination matches
    dst_indices = node_idx.gather(1, edge_idx)  # (B, r)

    # Add merged source tokens to their destination matches
    dst = dst.scatter_add(1, dst_indices.unsqueeze(-1).expand(-1, -1, C), merged_src)

    # Build mask for unmerged source tokens
    src_mask = torch.ones(B, src.shape[1], device=x.device, dtype=torch.bool)
    src_mask.scatter_(1, edge_idx, False)

    # Collect unmerged source tokens
    unmerged_src = src[src_mask].view(B, -1, C)

    if mode == "mean":
        # Count how many tokens were merged into each destination
        counts = torch.ones(B, dst.shape[1], 1, device=x.device)
        counts.scatter_add_(1, dst_indices.unsqueeze(-1), torch.ones(B, r, 1, device=x.device))
        dst = dst / counts

    # Concatenate: remaining source tokens + merged destination tokens
    return torch.cat([unmerged_src, dst], dim=1)


def unmerge_tokens(
    x: Tensor,
    merge_info: tuple[Tensor, Tensor] | tuple[None, None] | None,
    original_length: int,
) -> Tensor:
    """Unmerge tokens back to original sequence length.

    Args:
        x: Merged tokens (B, N_merged, C).
        merge_info: ``(edge_idx, node_idx)`` from matching. ``None`` is
            a no-op.
        original_length: Original sequence length before merging.
    """
    if merge_info is None or merge_info[0] is None:
        return x

    edge_idx, node_idx = merge_info
    B, _, C = x.shape
    r = edge_idx.shape[1]
    src_len = original_length // 2
    dst_len = original_length - src_len
    unmerged_src_len = src_len - r

    # Split back into unmerged source and destination
    unmerged_src = x[:, :unmerged_src_len]  # (B, src_len - r, C)
    dst = x[:, unmerged_src_len:]            # (B, dst_len, C)

    # Reconstruct merged source tokens from their destination matches
    dst_indices = node_idx.gather(1, edge_idx)  # (B, r)
    merged_src = dst.gather(1, dst_indices.unsqueeze(-1).expand(-1, -1, C))  # (B, r, C)

    # Reconstruct full source tensor
    full_src = torch.zeros(B, src_len, C, device=x.device, dtype=x.dtype)
    src_mask = torch.ones(B, src_len, device=x.device, dtype=torch.bool)
    src_mask.scatter_(1, edge_idx, False)

    # Place unmerged source tokens
    unmerged_indices = src_mask.nonzero(as_tuple=False)
    full_src[unmerged_indices[:, 0], unmerged_indices[:, 1]] = unmerged_src.reshape(-1, C)

    # Place merged source tokens
    full_src.scatter_(1, edge_idx.unsqueeze(-1).expand(-1, -1, C), merged_src)

    # Interleave: src at even positions, dst at odd positions
    out = torch.zeros(B, original_length, C, device=x.device, dtype=x.dtype)
    out[:, ::2] = full_src
    out[:, 1::2] = dst

    return out
