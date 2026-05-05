import numpy as np

from thought_anchors_code.analysis.whitebox_attention.attention_extraction import (
    _expand_sparse_attention_layers,
)
from thought_anchors_code.analysis.whitebox_attention.tokenization import (
    average_attention_by_sentence,
    average_attention_heads_by_sentence,
)
from thought_anchors_code.analysis.whitebox_attention.receiver_heads import (
    get_all_vertical_scores,
    get_vertical_scores,
)


def test_get_vertical_scores_ignores_local_band() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0, 0.0],
            [0.6, 0.8, 1.0, 0.0],
            [0.9, 0.4, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    scores = get_vertical_scores(matrix, proximity_ignore=2, control_depth=False)
    assert np.isclose(scores[0], 0.75)
    assert np.isclose(scores[1], 0.4)
    assert np.isnan(scores[2])
    assert np.isnan(scores[3])
    assert scores.shape == (4,)


def test_get_all_vertical_scores_matches_single_head_path() -> None:
    matrices = np.array(
        [
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.7, 1.0, 0.0, 0.0],
                    [0.6, 0.8, 1.0, 0.0],
                    [0.9, 0.4, 0.3, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.2, 1.0, 0.0, 0.0],
                    [0.4, 0.6, 1.0, 0.0],
                    [0.8, 0.5, 0.1, 1.0],
                ],
            ]
        ],
        dtype=np.float32,
    )

    batched = get_all_vertical_scores(matrices, proximity_ignore=2, control_depth=False)

    assert batched.shape == (1, 2, 4)
    for layer in range(matrices.shape[0]):
        for head in range(matrices.shape[1]):
            expected = get_vertical_scores(
                matrices[layer, head], proximity_ignore=2, control_depth=False
            )
            np.testing.assert_allclose(batched[layer, head], expected)


def test_expand_sparse_attention_layers_preserves_actual_layer_indices() -> None:
    class Config:
        layer_types = [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]

    class Model:
        config = Config()

    compressed = np.ones((2, 3, 4, 4), dtype=np.float32)
    compressed[1] *= 2

    expanded = _expand_sparse_attention_layers(compressed, Model())

    assert expanded.shape == (5, 3, 4, 4)
    assert np.all(np.isnan(expanded[0]))
    assert np.all(np.isnan(expanded[1]))
    assert np.all(expanded[2] == 1)
    assert np.all(np.isnan(expanded[3]))
    assert np.all(expanded[4] == 2)


def test_average_attention_heads_by_sentence_matches_single_head_path() -> None:
    matrices = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    boundaries = [(0, 2), (2, 4)]

    batched = average_attention_heads_by_sentence(matrices, boundaries)

    expected = np.asarray(
        [average_attention_by_sentence(matrix, boundaries) for matrix in matrices],
        dtype=np.float32,
    )
    np.testing.assert_allclose(batched, expected)
