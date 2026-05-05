"""Shared data structures for white-box attention analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeRollout:
    model_id: str
    dataset_name: str
    task_id: str
    sample_id: int
    complete: bool
    prompt: str
    raw: str
    reasoning: str
    answer: str | None = None
    is_correct: bool | None = None

    @property
    def full_text(self) -> str:
        return self.reasoning

    @property
    def rollout_key(self) -> str:
        task_slug = self.task_id.replace("/", "_").replace(" ", "_")
        return f"{self.dataset_name}__{task_slug}__s{self.sample_id}"


@dataclass(frozen=True)
class ReceiverHead:
    layer: int
    head: int
    kurtosis: float


@dataclass(frozen=True)
class RolloutAttentionSummary:
    task_id: str
    sample_id: int
    sentence_scores: list[float]
    receiver_head_scores: list[float]
    code_sentence_scores: list[float] | None = None


def default_trace_cache_dir(base_dir: Path, rollout_key: str) -> Path:
    return base_dir / rollout_key
