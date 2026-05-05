"""Tokenizer helpers for sentence-level attention aggregation."""

from __future__ import annotations

from typing import Sequence

from transformers import PreTrainedTokenizerBase


def get_sentence_token_boundaries(
    text: str,
    sentences: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[int, int]]:
    if not sentences:
        return []

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Sentence boundary mapping requires a fast tokenizer with offsets."
        )

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    boundaries: list[tuple[int, int]] = []
    search_start = 0

    for sentence in sentences:
        char_start = text.find(sentence, search_start)
        if char_start < 0:
            stripped = sentence.strip()
            char_start = text.find(stripped, search_start)
            if char_start < 0:
                raise ValueError(f"Sentence not found in trace text: {sentence!r}")
            sentence = stripped

        char_end = char_start + len(sentence)
        token_start = None
        token_end = None

        for index, (offset_start, offset_end) in enumerate(offsets):
            if token_start is None and offset_end > char_start:
                token_start = index
            if token_start is not None and offset_start < char_end <= offset_end:
                token_end = index + 1
                break
            if token_start is not None and offset_start >= char_end:
                token_end = index
                break

        if token_start is None:
            raise ValueError(
                f"Could not map sentence start to token index: {sentence!r}"
            )
        if token_end is None:
            token_end = len(offsets)

        boundaries.append((token_start, token_end))
        search_start = char_end

    return boundaries


def average_attention_by_sentence(
    matrix,
    sentence_boundaries: Sequence[tuple[int, int]],
):
    import numpy as np

    size = len(sentence_boundaries)
    averaged = np.zeros((size, size), dtype=np.float32)

    for row_index, (row_start, row_end) in enumerate(sentence_boundaries):
        for col_index, (col_start, col_end) in enumerate(sentence_boundaries):
            region = matrix[row_start:row_end, col_start:col_end]
            if region.size:
                averaged[row_index, col_index] = float(np.mean(region))
    return averaged


def average_attention_heads_by_sentence(
    matrices,
    sentence_boundaries: Sequence[tuple[int, int]],
):
    import numpy as np

    size = len(sentence_boundaries)
    if size == 0:
        return np.zeros((matrices.shape[0], 0, 0), dtype=np.float32)

    boundaries = np.asarray(sentence_boundaries, dtype=np.int64)
    row_starts = boundaries[:, 0]
    row_ends = boundaries[:, 1]
    col_starts = boundaries[:, 0]
    col_ends = boundaries[:, 1]
    row_lengths = np.maximum(row_ends - row_starts, 1)
    col_lengths = np.maximum(col_ends - col_starts, 1)
    denominators = (row_lengths[:, None] * col_lengths[None, :]).astype(np.float32)

    prefix = np.pad(matrices, ((0, 0), (1, 0), (1, 0)), mode="constant")
    prefix = prefix.cumsum(axis=1).cumsum(axis=2)
    averaged = np.empty((matrices.shape[0], size, size), dtype=np.float32)

    for row_index, (row_start, row_end) in enumerate(zip(row_starts, row_ends)):
        row_sums = (
            prefix[:, row_end, col_ends]
            - prefix[:, row_start, col_ends]
            - prefix[:, row_end, col_starts]
            + prefix[:, row_start, col_starts]
        )
        averaged[:, row_index, :] = row_sums / denominators[row_index]

    return averaged
