import numpy as np

from thought_anchors_code.analysis.whitebox_attention.attention_extraction import (
    _expand_sparse_attention_layers,
)
from thought_anchors_code.analysis.whitebox_attention.receiver_heads import (
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
