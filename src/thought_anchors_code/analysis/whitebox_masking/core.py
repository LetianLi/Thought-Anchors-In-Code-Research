"""Core causal matrix computation via attention suppression."""

from __future__ import annotations

import numpy as np
import torch

from thought_anchors_code.analysis.whitebox_attention.tokenization import (
    get_sentence_token_boundaries,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.analysis.whitebox_masking.hooks import QwenAttentionHookManager
from thought_anchors_code.analysis.whitebox_masking.kl_divergence import (
    sentence_mean_log_kl,
)


def compute_causal_matrix(
    rollout: CodeRollout,
    model,
    tokenizer,
    temperature: float = 0.6,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Compute the M×M causal influence matrix for one rollout.

    Returns:
        causal_matrix: [M, M] float32 array; NaN on diagonal and upper triangle.
                       Lower triangle (i < j): mean log-KL when sentence i is suppressed,
                       column-normalised (subtract per-column nanmean).
        sentences: list of M sentence strings
        boundaries: list of M (token_start, token_end) half-open tuples
    """
    sentences = split_reasoning_steps(rollout.reasoning)
    if len(sentences) < 2:
        m = len(sentences)
        return np.full((m, m), np.nan, dtype=np.float32), sentences, []

    text = rollout.reasoning
    boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    M = len(sentences)

    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    device = _input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        base_logits = model(**inputs, use_cache=False).logits[0].cpu()  # [seq_len, vocab]

    causal_matrix = np.full((M, M), np.nan, dtype=np.float32)

    for i in range(M):
        src_start, src_end = boundaries[i]
        if src_start >= src_end:
            continue

        with torch.no_grad():
            with QwenAttentionHookManager(model, [src_start, src_end]):
                masked_logits = model(**inputs, use_cache=False).logits[0].cpu()

        for j in range(i + 1, M):
            tgt_start, tgt_end = boundaries[j]
            causal_matrix[i, j] = sentence_mean_log_kl(
                base_logits, masked_logits, tgt_start, tgt_end, temperature
            )

        del masked_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del base_logits

    # Column normalise: subtract nanmean of each column's lower-triangle entries.
    # j=0 has no prior sources (empty slice) so we skip it.
    for j in range(1, M):
        col = causal_matrix[:j, j]
        col_mean = np.nanmean(col)
        if not np.isnan(col_mean):
            causal_matrix[:j, j] -= col_mean

    return causal_matrix, sentences, boundaries


def _input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
