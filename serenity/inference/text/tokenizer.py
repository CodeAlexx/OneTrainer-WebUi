"""Prompt parsing, weighting, and tokenization utilities."""

from __future__ import annotations

import logging
import re

__all__ = [
    "create_token_chunks",
    "parse_prompt_weights",
    "truncate_or_pad",
]

logger = logging.getLogger(__name__)

# Default weight for un-annotated text
_DEFAULT_WEIGHT = 1.0
# Weight boost for bare parentheses (word) without explicit weight
_BARE_PAREN_WEIGHT = 1.1


# ---------------------------------------------------------------------------
# Prompt weight parsing
# ---------------------------------------------------------------------------


def parse_prompt_weights(prompt: str) -> list[tuple[str, float]]:
    """Parse prompt weighting syntax into (text, weight) segments.

    Syntax rules:
      - Plain text gets weight 1.0
      - ``(word:1.5)`` sets weight to 1.5
      - ``(word)`` without explicit weight applies 1.1 multiplier
      - Nested parentheses multiply weights
      - ``BREAK`` keyword splits segments
      - Escaped parentheses ``\\(`` ``\\)`` are treated as literal text

    Returns:
        List of (text_segment, weight) tuples.
    """
    if not prompt:
        return [("", _DEFAULT_WEIGHT)]

    result: list[tuple[str, float]] = []
    weight_stack: list[float] = [_DEFAULT_WEIGHT]
    current_text: list[str] = []
    i = 0
    length = len(prompt)

    while i < length:
        char = prompt[i]

        # Handle escaped parentheses
        if char == "\\" and i + 1 < length and prompt[i + 1] in ("(", ")"):
            current_text.append(prompt[i + 1])
            i += 2
            continue

        # Handle BREAK keyword
        if prompt[i:i + 5] == "BREAK":
            # Check that BREAK is word-bounded
            before_ok = i == 0 or not prompt[i - 1].isalnum()
            after_ok = i + 5 >= length or not prompt[i + 5].isalnum()
            if before_ok and after_ok:
                text = "".join(current_text)
                if text:
                    result.append((text, weight_stack[-1]))
                    current_text = []
                result.append(("BREAK", _DEFAULT_WEIGHT))
                i += 5
                continue

        # Opening parenthesis — push weight
        if char == "(":
            text = "".join(current_text)
            if text:
                result.append((text, weight_stack[-1]))
                current_text = []

            # Look ahead for explicit weight `(text:weight)`
            # We need to find the matching close paren
            i += 1
            continue_outer = False

            # Find matching close paren considering nesting
            depth = 1
            j = i
            while j < length and depth > 0:
                if prompt[j] == "\\" and j + 1 < length and prompt[j + 1] in ("(", ")"):
                    j += 2
                    continue
                if prompt[j] == "(":
                    depth += 1
                elif prompt[j] == ")":
                    depth -= 1
                j += 1

            if depth == 0:
                # j is now one past the closing paren
                inner = prompt[i:j - 1]

                # Check for explicit weight at the end: `:float`
                # Find the last colon not inside nested parens
                colon_pos = _find_weight_colon(inner)
                if colon_pos is not None:
                    weight_text = inner[colon_pos + 1:]
                    inner_prompt = inner[:colon_pos]
                    try:
                        w = float(weight_text)
                        weight_stack.append(w)
                        # Recursively parse the inner prompt
                        inner_segments = parse_prompt_weights(inner_prompt)
                        for seg_text, seg_weight in inner_segments:
                            if seg_text:
                                # Inner segments already have default weight;
                                # we multiply by the outer weight
                                result.append((seg_text, w * (seg_weight / _DEFAULT_WEIGHT) if seg_weight == _DEFAULT_WEIGHT else seg_weight))
                        weight_stack.pop()
                        i = j
                        continue
                    except ValueError:
                        pass

                # Bare parentheses — apply 1.1 multiplier
                new_weight = weight_stack[-1] * _BARE_PAREN_WEIGHT
                weight_stack.append(new_weight)
                inner_segments = parse_prompt_weights(inner)
                for seg_text, seg_weight in inner_segments:
                    if seg_text:
                        if seg_weight == _DEFAULT_WEIGHT:
                            result.append((seg_text, new_weight))
                        else:
                            result.append((seg_text, seg_weight))
                weight_stack.pop()
                i = j
                continue

            # Unmatched paren — treat as literal
            current_text.append("(")
            continue

        # Closing parenthesis without matching open — treat as literal
        if char == ")":
            current_text.append(")")
            i += 1
            continue

        current_text.append(char)
        i += 1

    # Flush remaining text
    text = "".join(current_text)
    if text:
        result.append((text, weight_stack[-1]))

    # If nothing was produced, return empty with default weight
    if not result:
        return [("", _DEFAULT_WEIGHT)]

    return result


def _find_weight_colon(text: str) -> int | None:
    """Find the position of a weight-specifying colon in parenthesised text.

    Returns the index of the last colon that sits outside any nested
    parentheses group and is followed by a valid float, or ``None``.
    """
    depth = 0
    last_colon = None
    for i, ch in enumerate(text):
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in ("(", ")"):
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            last_colon = i

    if last_colon is not None:
        remainder = text[last_colon + 1:]
        try:
            float(remainder)
            return last_colon
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Token manipulation
# ---------------------------------------------------------------------------


def truncate_or_pad(
    tokens: list[int],
    max_length: int,
    pad_token: int,
) -> list[int]:
    """Truncate or pad a token list to exactly *max_length*.

    Args:
        tokens: Input token IDs.
        max_length: Target length.
        pad_token: Token ID used for padding.

    Returns:
        Token list of exactly *max_length* elements.
    """
    if len(tokens) >= max_length:
        return tokens[:max_length]
    return tokens + [pad_token] * (max_length - len(tokens))


def create_token_chunks(
    tokens: list[int],
    weights: list[float],
    max_chunk_length: int,
) -> list[tuple[list[int], list[float]]]:
    """Split tokens and weights into fixed-size chunks for multi-pass encoding.

    Args:
        tokens: Flat list of token IDs.
        weights: Corresponding per-token weights (same length as *tokens*).
        max_chunk_length: Maximum tokens per chunk.

    Returns:
        List of (token_chunk, weight_chunk) tuples.
    """
    if not tokens:
        return [([], [])]

    chunks: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(tokens), max_chunk_length):
        end = start + max_chunk_length
        chunks.append((tokens[start:end], weights[start:end]))

    return chunks
