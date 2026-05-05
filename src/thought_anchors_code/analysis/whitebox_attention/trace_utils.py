"""Utilities for loading code rollouts and segmenting reasoning sentences."""

from __future__ import annotations

import json
from pathlib import Path
import re
import warnings
from dataclasses import replace

import numpy as np

from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout


SENTENCE_END_RE = re.compile(r"(?<=[.!?])(?=(?:\s|$))")
LINE_STEP_RE = re.compile(r"^(\s*(?:[-*]|\d+[.)])\s+)(.*)$")


def split_reasoning_steps(text: str) -> list[str]:
    """Split a reasoning trace into sentence-like steps while preserving spacing."""
    content = _strip_code_sections(text)
    if not content.strip():
        return []

    steps: list[str] = []
    for line in content.splitlines(keepends=True):
        bullet_match = LINE_STEP_RE.match(line)
        if bullet_match:
            prefix, content = bullet_match.groups()
            parts = _split_sentence_like(content)
            if not parts:
                steps.append(line)
                continue
            for index, part in enumerate(parts):
                steps.append(f"{prefix if index == 0 else ''}{part}")
            continue
        steps.extend(_split_sentence_like(line))
    return [step for step in steps if step.strip()]


def _split_sentence_like(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.start() + 1
        parts.append(text[start:end])
        start = end
    if start < len(text):
        parts.append(text[start:])
    return parts


def _strip_code_sections(text: str) -> str:
    index = text.find("<code>")
    if index >= 0:
        return text[:index]
    return text


def load_rollouts_jsonl(path: str | Path) -> list[CodeRollout]:
    rollout_path = Path(path)
    rollouts: list[CodeRollout] = []
    with rollout_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            model_id = payload.get("model_id")
            dataset_name = payload.get("dataset_name") or payload.get("dataset")
            task_id = (
                payload.get("task_id") or payload.get("problem_id") or payload.get("id")
            )
            sample_id = payload.get("sample_id")
            complete = payload.get("complete")
            reasoning = (
                payload.get("reasoning")
                or payload.get("full_cot")
                or payload.get("trace")
            )
            if not model_id:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing model_id.",
                    stacklevel=2,
                )
                continue
            if not dataset_name:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing dataset_name.",
                    stacklevel=2,
                )
                continue
            if not task_id:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing task_id.",
                    stacklevel=2,
                )
                continue
            if sample_id is None:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing sample_id.",
                    stacklevel=2,
                )
                continue
            if complete is None:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing complete flag.",
                    stacklevel=2,
                )
                continue
            if not reasoning:
                warnings.warn(
                    f"Skipping line {line_number} in {rollout_path}: missing reasoning text.",
                    stacklevel=2,
                )
                continue
            rollouts.append(
                CodeRollout(
                    model_id=str(model_id),
                    dataset_name=str(dataset_name),
                    task_id=str(task_id),
                    sample_id=int(sample_id),
                    complete=bool(complete),
                    prompt=str(payload.get("prompt") or ""),
                    raw=str(payload.get("raw") or payload.get("generated_text") or ""),
                    reasoning=str(reasoning),
                    answer=str(payload.get("answer") or payload.get("code") or ""),
                    is_correct=payload.get("is_correct"),
                )
            )
    return rollouts


def truncate_rollouts_to_sentence_percentile(
    rollouts: list[CodeRollout],
    percentile: float = 75.0,
) -> tuple[list[CodeRollout], int | None]:
    counts = [len(split_reasoning_steps(rollout.reasoning)) for rollout in rollouts]
    counts = [count for count in counts if count > 0]
    if not counts:
        return rollouts, None

    max_sentences = max(1, int(np.percentile(np.asarray(counts, dtype=float), percentile)))
    truncated = []
    for rollout in rollouts:
        steps = split_reasoning_steps(rollout.reasoning)
        if len(steps) > max_sentences:
            truncated.append(replace(rollout, reasoning="".join(steps[:max_sentences])))
        else:
            truncated.append(rollout)
    return truncated, max_sentences
