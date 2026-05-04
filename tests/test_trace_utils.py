from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
)


def test_split_reasoning_steps_splits_sentences_and_bullets() -> None:
    text = (
        "Plan the solution. Then check edge cases.\n1. Write helper.\n2. Return result"
    )
    assert split_reasoning_steps(text) == [
        "Plan the solution. ",
        "Then check edge cases.\n",
        "1. Write helper.",
        "2. Return result",
    ]


def test_split_reasoning_steps_stops_before_code_tag() -> None:
    text = "Plan first. Then implement.\n<code>def foo():\n    return 1\n</code>"
    assert split_reasoning_steps(text) == ["Plan first. ", "Then implement.\n"]
