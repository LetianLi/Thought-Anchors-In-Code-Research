from __future__ import annotations

import json

from thought_anchors_code.analysis.blackbox_resampling.core import (
    build_continuation_prompt,
    build_resampling_jobs,
    enumerate_sentence_interventions,
    read_completed_resampling_keys,
)
from thought_anchors_code.analysis.blackbox_resampling.summarize import (
    summarize_resampling_file,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
    truncate_rollouts_to_sentence_percentile,
)


def _rollout() -> CodeRollout:
    return CodeRollout(
        model_id="model",
        dataset_name="openai_humaneval",
        task_id="HumanEval/1",
        sample_id=0,
        complete=True,
        prompt="Solve it.",
        raw="",
        reasoning="First, parse input. Then handle edge cases. Finally return output.",
        answer="def f(): pass",
        is_correct=True,
    )


def test_truncate_rollouts_to_sentence_percentile_caps_long_rollouts() -> None:
    rollouts = [
        CodeRollout(
            model_id="model",
            dataset_name="openai_humaneval",
            task_id=f"HumanEval/{index}",
            sample_id=0,
            complete=True,
            prompt="Solve it.",
            raw="",
            reasoning=" ".join(["One."] * count),
            answer="def f(): pass",
            is_correct=True,
        )
        for index, count in enumerate([1, 2, 3, 4], start=1)
    ]

    truncated, max_sentences = truncate_rollouts_to_sentence_percentile(rollouts)

    assert max_sentences == 3
    assert [len(split_reasoning_steps(rollout.reasoning)) for rollout in truncated] == [
        1,
        2,
        3,
        3,
    ]


def test_enumerate_sentence_interventions_includes_every_sentence() -> None:
    summaries = {
        ("HumanEval/1", 0): {
            "sentence_scores": [0.1, 0.2, 0.3],
            "code_sentence_scores": [0.01, 0.9, 0.02],
        }
    }

    interventions = enumerate_sentence_interventions(
        [_rollout()],
        summaries,
    )

    assert [
        (intervention.selection, intervention.sentence_index)
        for intervention in interventions
    ] == [
        ("sentence", 0),
        ("sentence", 1),
        ("sentence", 2),
    ]
    assert interventions[1].code_sentence_score == 0.9
    assert interventions[1].sentence_text == "Then handle edge cases."


def test_build_resampling_jobs_expands_interventions_by_resample_count() -> None:
    rollout = _rollout()
    interventions = enumerate_sentence_interventions(
        [rollout],
        {("HumanEval/1", 0): {"sentence_scores": [0.1, 0.2, 0.3]}},
    )

    jobs = build_resampling_jobs(
        interventions[:2],
        rollout_map={("openai_humaneval", "HumanEval/1", 0): rollout},
        rows_by_task_id={"HumanEval/1": {"task_id": "HumanEval/1"}},
        tokenizer=None,
        num_resamples=3,
    )

    assert len(jobs) == 6
    assert [job.resample_id for job in jobs[:3]] == [0, 1, 2]
    assert [job.intervention.sentence_index for job in jobs] == [0, 0, 0, 1, 1, 1]


def test_build_continuation_prompt_uses_prefix_before_omitted_sentence() -> None:
    prompt, prefix_count, suffix_count = build_continuation_prompt(
        _rollout(),
        1,
        tokenizer=None,
    )

    assert prompt == "Solve it.<reasoning>\nFirst, parse input. "
    assert prefix_count == 1
    assert suffix_count == 1


def test_read_completed_resampling_keys(tmp_path) -> None:
    output = tmp_path / "resampling.jsonl"
    output.write_text(
        json.dumps(
            {
                "dataset_name": "mbpp",
                "task_id": "12",
                "sample_id": 0,
                "sentence_index": 3,
                "selection": "sentence",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_completed_resampling_keys(output) == {("mbpp", "12", 0, 3, "sentence")}


def test_summarize_resampling_file_groups_by_selection(tmp_path) -> None:
    output = tmp_path / "resampling.jsonl"
    rows = [
        {
            "dataset_name": "mbpp",
            "task_id": "12",
            "sample_id": 0,
            "sentence_index": 1,
            "selection": "sentence",
            "sentence_score": 0.4,
            "code_sentence_score": 0.8,
            "original_is_correct": True,
            "resamples": [{"is_correct": False}, {"is_correct": True}],
        },
        {
            "dataset_name": "mbpp",
            "task_id": "13",
            "sample_id": 0,
            "sentence_index": 2,
            "selection": "sentence",
            "sentence_score": 0.5,
            "code_sentence_score": 0.7,
            "original_is_correct": True,
            "resamples": [{"is_correct": True}, {"is_correct": True}],
        },
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = summarize_resampling_file(output)

    assert summary == [
        {
            "dataset_name": "mbpp",
            "sentence_index": 1,
            "original_is_correct": True,
            "interventions": 1,
            "mean_sentence_score": 0.4,
            "mean_code_sentence_score": 0.8,
            "mean_resample_pass_rate": 0.5,
            "mean_pass_rate_delta": -0.5,
        },
        {
            "dataset_name": "mbpp",
            "sentence_index": 2,
            "original_is_correct": True,
            "interventions": 1,
            "mean_sentence_score": 0.5,
            "mean_code_sentence_score": 0.7,
            "mean_resample_pass_rate": 1,
            "mean_pass_rate_delta": 0,
        },
    ]
