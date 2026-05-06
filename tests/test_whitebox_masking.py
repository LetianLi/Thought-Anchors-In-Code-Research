"""Smoke tests for whitebox_masking causal matrix computation."""

from __future__ import annotations

import numpy as np
import pytest

from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
)
from thought_anchors_code.analysis.whitebox_masking import compute_causal_matrix
from thought_anchors_code.config import DEFAULT_MODEL_ID, ROLLOUT_DIR
from thought_anchors_code.engine.model_loader import get_local_model


ROLLOUT_FILE = ROLLOUT_DIR / "humaneval_qwen3_5_0_8b_full.jsonl"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return get_local_model(DEFAULT_MODEL_ID, float32=True, device_map="cpu")


@pytest.fixture(scope="module")
def correct_rollouts():
    if not ROLLOUT_FILE.exists():
        pytest.skip(f"Rollout file not found: {ROLLOUT_FILE}")
    rollouts = load_rollouts_jsonl(ROLLOUT_FILE)
    correct = [r for r in rollouts if r.is_correct is True]
    if len(correct) < 2:
        pytest.skip("Need at least 2 correct rollouts.")
    return correct[:2]


def test_causal_matrix_shape_and_structure(correct_rollouts, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    for rollout in correct_rollouts:
        matrix, sentences, boundaries = compute_causal_matrix(rollout, model, tokenizer)
        M = len(sentences)
        assert matrix.shape == (M, M), f"Expected ({M},{M}), got {matrix.shape}"

        # Upper triangle including diagonal must be NaN
        for i in range(M):
            for j in range(i + 1):
                assert np.isnan(matrix[i, j]), f"Expected NaN at ({i},{j}), got {matrix[i,j]}"

        # Lower triangle must be finite (no NaN, no Inf)
        lower_mask = np.zeros((M, M), dtype=bool)
        for i in range(M):
            for j in range(i + 1, M):
                lower_mask[i, j] = True
        if lower_mask.any():
            finite_vals = matrix[lower_mask]
            assert not np.any(np.isnan(finite_vals)), "NaN found in lower triangle"
            assert not np.any(np.isinf(finite_vals)), "Inf found in lower triangle"

        # Boundaries count should match sentence count
        assert len(boundaries) == M
