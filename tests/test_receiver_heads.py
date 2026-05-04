import numpy as np

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
