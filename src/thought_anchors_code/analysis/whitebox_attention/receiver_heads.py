"""Receiver-head scoring logic adapted for code reasoning rollouts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
from scipy import stats

from thought_anchors_code.analysis.whitebox_attention.attention_extraction import (
    build_sentence_attention_cache,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
)
from thought_anchors_code.analysis.whitebox_attention.types import (
    CodeRollout,
    ReceiverHead,
    RolloutAttentionSummary,
)


def get_vertical_scores(
    averaged_attention: np.ndarray,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    matrix = averaged_attention.copy()
    matrix[np.triu_indices_from(matrix, k=1)] = np.nan
    matrix[np.triu_indices_from(matrix, k=-proximity_ignore + 1)] = np.nan

    if control_depth:
        per_row = np.sum(~np.isnan(matrix), axis=1)
        matrix = stats.rankdata(matrix, axis=1, nan_policy="omit") / per_row[:, None]

    vertical_scores = []
    for column in range(matrix.shape[1]):
        values = matrix[column + proximity_ignore :, column]
        vertical_scores.append(np.nanmean(values) if values.size else np.nan)
    return np.asarray(vertical_scores, dtype=np.float32)


def get_all_vertical_scores(
    averaged_attention: np.ndarray,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    """Compute vertical scores for all layer/head matrices at once."""
    matrix = averaged_attention.copy()
    sentence_count = matrix.shape[-1]
    upper_rows, upper_cols = np.triu_indices(sentence_count, k=1)
    local_rows, local_cols = np.triu_indices(sentence_count, k=-proximity_ignore + 1)
    matrix[..., upper_rows, upper_cols] = np.nan
    matrix[..., local_rows, local_cols] = np.nan

    if control_depth:
        per_row = np.sum(~np.isnan(matrix), axis=-1)
        matrix = stats.rankdata(matrix, axis=-1, nan_policy="omit") / per_row[..., None]

    vertical_scores = np.full(matrix.shape[:-1], np.nan, dtype=np.float32)
    for column in range(sentence_count):
        values = matrix[..., column + proximity_ignore :, column]
        if values.shape[-1] == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vertical_scores[..., column] = np.nanmean(values, axis=-1)
    return vertical_scores


def get_trace_vertical_scores(
    trace: CodeRollout,
    model_name_or_path: str,
    cache_dir: Path | None = None,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> tuple[list[str], np.ndarray]:
    sentences = split_reasoning_steps(trace.reasoning)
    if len(sentences) <= proximity_ignore:
        raise ValueError(
            f"task_id={trace.task_id}, sample_id={trace.sample_id} has only {len(sentences)} reasoning steps; need more than proximity_ignore={proximity_ignore}."
        )
    matrices = build_sentence_attention_cache(
        text=trace.reasoning,
        sentences=sentences,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )
    layers, heads = matrices.shape[:2]
    scores = get_all_vertical_scores(
        matrices,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )
    if scores.shape != (layers, heads, len(sentences)):
        raise ValueError(
            f"Unexpected vertical score shape {scores.shape}; expected {(layers, heads, len(sentences))}."
        )
    return sentences, scores


def analyze_receiver_heads_once(
    rollouts: Sequence[CodeRollout],
    model_name_or_path: str,
    cache_dir: Path | None = None,
    top_k: int = 20,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> tuple[list[ReceiverHead], list[RolloutAttentionSummary]]:
    scored_rollouts: list[tuple[CodeRollout, np.ndarray]] = []
    kurtosis_by_trace = []
    for rollout in rollouts:
        try:
            _, vertical_scores = _compute_rollout_vertical_scores(
                rollout,
                model_name_or_path,
                cache_dir,
                proximity_ignore,
                control_depth,
            )
        except ValueError as error:
            warnings.warn(
                f"Skipping rollout during receiver-head analysis: {error}", stacklevel=2
            )
            continue
        scored_rollouts.append((rollout, vertical_scores))
        kurtosis_by_trace.append(
            stats.kurtosis(
                vertical_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
            )
        )

    receiver_heads = _rank_receiver_heads_from_kurtoses(kurtosis_by_trace, top_k=top_k)
    rollout_order = {
        (rollout.task_id, rollout.sample_id): index for index, rollout in enumerate(rollouts)
    }
    scored_rollouts.sort(
        key=lambda item: rollout_order[(item[0].task_id, item[0].sample_id)]
    )
    summaries = [
        _summarize_scores_with_receiver_heads(
            rollout,
            vertical_scores,
            receiver_heads,
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
        )
        for rollout, vertical_scores in scored_rollouts
    ]
    return receiver_heads, summaries


def analyze_receiver_heads_to_jsonl(
    rollouts: Sequence[CodeRollout],
    model_name_or_path: str,
    output_path: Path,
    cache_dir: Path | None = None,
    top_k: int = 20,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    resume: bool = True,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = read_completed_attention_summary_keys(output_path) if resume else set()
    if not resume:
        output_path.write_text("", encoding="utf-8")

    scored_rollouts: list[tuple[CodeRollout, np.ndarray]] = []
    kurtosis_by_trace = []
    for rollout in rollouts:
        try:
            _, vertical_scores = _compute_rollout_vertical_scores(
                rollout,
                model_name_or_path,
                cache_dir,
                proximity_ignore,
                control_depth,
            )
        except ValueError as error:
            warnings.warn(
                f"Skipping rollout during receiver-head analysis: {error}", stacklevel=2
            )
            continue
        scored_rollouts.append((rollout, vertical_scores))
        kurtosis_by_trace.append(
            stats.kurtosis(
                vertical_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
            )
        )

    receiver_heads = _rank_receiver_heads_from_kurtoses(kurtosis_by_trace, top_k=top_k)
    rollout_order = {
        (rollout.task_id, rollout.sample_id): index for index, rollout in enumerate(rollouts)
    }
    scored_rollouts.sort(
        key=lambda item: rollout_order[(item[0].task_id, item[0].sample_id)]
    )

    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for rollout, vertical_scores in scored_rollouts:
            key = (rollout.model_id, rollout.dataset_name, rollout.task_id, rollout.sample_id)
            if key in completed_keys:
                continue
            summary = _summarize_scores_with_receiver_heads(
                rollout,
                vertical_scores,
                receiver_heads,
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
            )
            payload = asdict(summary)
            payload["dataset_name"] = rollout.dataset_name
            payload["task_id"] = rollout.task_id
            payload["model_id"] = rollout.model_id
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            written += 1
    return written


def read_completed_attention_summary_keys(
    output_path: str | Path,
) -> set[tuple[str, str, str, int]]:
    path = Path(output_path)
    keys: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            keys.add(
                (
                    str(payload.get("model_id")),
                    str(payload.get("dataset_name")),
                    str(payload.get("task_id")),
                    int(payload.get("sample_id", 0)),
                )
            )
    return keys


def _compute_rollout_vertical_scores(
    rollout: CodeRollout,
    model_name_or_path: str,
    cache_dir: Path | None,
    proximity_ignore: int,
    control_depth: bool,
) -> tuple[CodeRollout, np.ndarray]:
    _, vertical_scores = get_trace_vertical_scores(
        trace=rollout,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )
    return rollout, vertical_scores


def _rank_receiver_heads_from_kurtoses(
    kurtosis_by_trace: Sequence[np.ndarray], top_k: int
) -> list[ReceiverHead]:
    if not kurtosis_by_trace:
        raise ValueError("No valid rollouts available for receiver-head ranking.")

    mean_kurtosis = np.nanmean(np.asarray(kurtosis_by_trace, dtype=np.float32), axis=0)
    flat = mean_kurtosis.reshape(-1)
    valid = np.flatnonzero(~np.isnan(flat))
    if len(valid) == 0:
        raise ValueError("Receiver-head ranking produced only NaN kurtosis scores.")
    top_k = min(top_k, len(valid))
    top_flat = valid[np.argpartition(flat[valid], -top_k)[-top_k:]]
    top_flat = top_flat[np.argsort(flat[top_flat])[::-1]]

    _, heads = mean_kurtosis.shape
    return [
        ReceiverHead(
            layer=int(flat_index // heads),
            head=int(flat_index % heads),
            kurtosis=float(mean_kurtosis[int(flat_index // heads), int(flat_index % heads)]),
        )
        for flat_index in top_flat
    ]


def _summarize_scores_with_receiver_heads(
    trace: CodeRollout,
    vertical_scores: np.ndarray,
    receiver_heads: Sequence[ReceiverHead],
    model_name_or_path: str | None = None,
    cache_dir: Path | None = None,
) -> RolloutAttentionSummary:
    receiver_head_scores = np.asarray(
        [vertical_scores[receiver.layer, receiver.head, :] for receiver in receiver_heads],
        dtype=np.float32,
    )
    valid_head_rows = ~np.all(np.isnan(receiver_head_scores), axis=1)
    if not np.any(valid_head_rows):
        raise ValueError(
            f"task_id={trace.task_id}, sample_id={trace.sample_id} produced only NaN receiver-head scores."
        )

    filtered_scores = receiver_head_scores[valid_head_rows]
    valid_sentence_counts = np.sum(~np.isnan(filtered_scores), axis=0)
    sentence_score_sums = np.nansum(filtered_scores, axis=0)
    sentence_scores = np.divide(
        sentence_score_sums,
        valid_sentence_counts,
        out=np.full(filtered_scores.shape[1], np.nan, dtype=np.float32),
        where=valid_sentence_counts > 0,
    )
    if np.all(np.isnan(sentence_scores)):
        raise ValueError(
            f"task_id={trace.task_id}, sample_id={trace.sample_id} produced only NaN sentence scores."
        )

    return RolloutAttentionSummary(
        task_id=trace.task_id,
        sample_id=trace.sample_id,
        sentence_scores=sentence_scores.tolist(),
        receiver_head_scores=[float(score) for score in np.nanmean(filtered_scores, axis=1)],
        code_sentence_scores=(
            get_code_to_reasoning_scores(
                trace=trace,
                receiver_heads=receiver_heads,
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
            ).tolist()
            if model_name_or_path is not None
            else None
        ),
    )


def get_code_to_reasoning_scores(
    trace: CodeRollout,
    receiver_heads: Sequence[ReceiverHead],
    model_name_or_path: str,
    cache_dir: Path | None = None,
) -> np.ndarray:
    reasoning_sentences = split_reasoning_steps(trace.reasoning)
    answer = (trace.answer or "").strip()
    if not reasoning_sentences or not answer:
        return np.full(len(reasoning_sentences), np.nan, dtype=np.float32)

    code_sentence = f"\n<code>\n{answer}\n</code>"
    text = trace.reasoning + code_sentence
    sentences = [*reasoning_sentences, code_sentence]
    matrices = build_sentence_attention_cache(
        text=text,
        sentences=sentences,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )

    code_index = len(sentences) - 1
    per_head_scores = []
    for receiver in receiver_heads:
        per_head_scores.append(matrices[receiver.layer, receiver.head, code_index, :code_index])
    if not per_head_scores:
        return np.full(len(reasoning_sentences), np.nan, dtype=np.float32)
    return np.nanmean(np.asarray(per_head_scores, dtype=np.float32), axis=0)


def rank_receiver_heads(
    rollouts: Sequence[CodeRollout],
    model_name_or_path: str,
    cache_dir: Path | None = None,
    top_k: int = 20,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> list[ReceiverHead]:
    kurtosis_by_trace = []
    for rollout in rollouts:
        try:
            _, vertical_scores = get_trace_vertical_scores(
                trace=rollout,
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
            )
        except ValueError as error:
            warnings.warn(
                f"Skipping rollout during receiver-head ranking: {error}", stacklevel=2
            )
            continue
        kurtosis_by_trace.append(
            stats.kurtosis(
                vertical_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
            )
        )

    return _rank_receiver_heads_from_kurtoses(kurtosis_by_trace, top_k=top_k)


def summarize_trace_with_receiver_heads(
    trace: CodeRollout,
    receiver_heads: Sequence[ReceiverHead],
    model_name_or_path: str,
    cache_dir: Path | None = None,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> RolloutAttentionSummary:
    if not receiver_heads:
        raise ValueError("No receiver heads were selected for summary export.")

    _, vertical_scores = get_trace_vertical_scores(
        trace=trace,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )
    receiver_head_scores = np.asarray(
        [
            vertical_scores[receiver.layer, receiver.head, :]
            for receiver in receiver_heads
        ],
        dtype=np.float32,
    )
    valid_head_rows = ~np.all(np.isnan(receiver_head_scores), axis=1)
    if not np.any(valid_head_rows):
        raise ValueError(
            f"task_id={trace.task_id}, sample_id={trace.sample_id} produced only NaN receiver-head scores."
        )

    filtered_scores = receiver_head_scores[valid_head_rows]
    valid_sentence_counts = np.sum(~np.isnan(filtered_scores), axis=0)
    sentence_score_sums = np.nansum(filtered_scores, axis=0)
    sentence_scores = np.divide(
        sentence_score_sums,
        valid_sentence_counts,
        out=np.full(filtered_scores.shape[1], np.nan, dtype=np.float32),
        where=valid_sentence_counts > 0,
    )
    if np.all(np.isnan(sentence_scores)):
        raise ValueError(
            f"task_id={trace.task_id}, sample_id={trace.sample_id} produced only NaN sentence scores."
        )

    return RolloutAttentionSummary(
        task_id=trace.task_id,
        sample_id=trace.sample_id,
        sentence_scores=sentence_scores.tolist(),
        receiver_head_scores=[
            float(score) for score in np.nanmean(filtered_scores, axis=1)
        ],
    )


def export_receiver_head_summary(
    output_path: Path,
    rollouts: Sequence[CodeRollout],
    receiver_heads: Sequence[ReceiverHead],
    model_name_or_path: str,
    cache_dir: Path | None = None,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            try:
                summary = summarize_trace_with_receiver_heads(
                    trace=rollout,
                    receiver_heads=receiver_heads,
                    model_name_or_path=model_name_or_path,
                    cache_dir=cache_dir,
                    proximity_ignore=proximity_ignore,
                    control_depth=control_depth,
                )
            except ValueError as error:
                warnings.warn(
                    f"Skipping rollout during summary export: {error}", stacklevel=2
                )
                continue
            payload = asdict(summary)
            payload["dataset_name"] = rollout.dataset_name
            payload["task_id"] = rollout.task_id
            payload["model_id"] = rollout.model_id
            handle.write(json.dumps(payload) + "\n")
