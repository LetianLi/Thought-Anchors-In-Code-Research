"""Shared data structures for black-box reasoning resampling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceIntervention:
    dataset_name: str
    task_id: str
    sample_id: int
    sentence_index: int
    sentence_text: str
    selection: str
    sentence_score: float | None
    code_sentence_score: float | None


@dataclass(frozen=True)
class ResampleOutcome:
    resample_id: int
    raw: str
    reasoning: str
    answer: str
    complete: bool
    is_correct: bool | None


@dataclass(frozen=True)
class ResamplingResult:
    model_id: str
    dataset_name: str
    task_id: str
    sample_id: int
    sentence_index: int
    selection: str
    sentence_text: str
    sentence_score: float | None
    code_sentence_score: float | None
    original_answer: str | None
    original_is_correct: bool | None
    prefix_sentence_count: int
    suffix_sentence_count: int
    resamples: list[ResampleOutcome]
