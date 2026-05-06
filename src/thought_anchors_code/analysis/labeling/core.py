"""Core utilities for labeling reasoning rollout sentences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable

from tqdm import tqdm

from thought_anchors_code.analysis.labeling.providers import LLMClient
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    load_rollouts_jsonl,
    split_reasoning_steps,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.config import PROJECT_ROOT


KNOWN_FUNCTION_TAGS = {
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",
    "final_answer_emission",
    "self_checking",
    "unknown",
}

STRICT_JSON_SUFFIX = (
    "\n\nReturn only valid JSON. Do not wrap the JSON in Markdown. "
    "Use exactly the sentence indices shown above."
)


@dataclass(frozen=True)
class SentenceLabel:
    function_tags: list[str]
    depends_on: list[str]


@dataclass(frozen=True)
class LabeledRollout:
    model_id: str
    dataset_name: str
    task_id: str
    sample_id: int
    complete: bool
    is_correct: bool | None
    prompt: str
    reasoning: str
    answer: str | None
    sentences: list[dict[str, str]]
    labels: dict[str, SentenceLabel]
    label_provider: str
    label_model: str
    label_prompt: str
    raw_label_response: str
    validation_warnings: list[str]
    labeled_at: str


@dataclass(frozen=True)
class LabelingJob:
    rollout: CodeRollout
    sentences: list[str]
    label_prompt: str

    @property
    def key(self) -> tuple[str, str, int]:
        return label_key(self.rollout)


@dataclass(frozen=True)
class LabelingFailure:
    job: LabelingJob
    error: str


@dataclass(frozen=True)
class RolloutLabelingSummary:
    total_rollouts: int
    complete_rollouts: int
    incomplete_rollouts: int
    rollouts_with_reasoning: int
    selected_rollouts: int
    already_labeled: int
    pending_jobs: int


def run_labeling_to_jsonl(
    *,
    rollout_file: str | Path,
    output_path: str | Path,
    client: LLMClient,
    prompt_path: str | Path | None = None,
    limit_rollouts: int | None = None,
    resume: bool = True,
    include_incomplete: bool = False,
    system_prompt: str | None = None,
    request_interval_seconds: float = 0.0,
    strict_json_instruction: bool = True,
    batch_size: int = 1,
    concurrency: int = 1,
) -> int:
    """Label rollout sentences and append one JSON object per rollout."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = read_completed_label_keys(destination) if resume else set()
    if not resume:
        destination.write_text("", encoding="utf-8")
    elif not destination.exists():
        destination.write_text("", encoding="utf-8")

    classification_prompt = load_classification_prompt(prompt_path)
    rollouts = load_rollouts_jsonl(rollout_file)
    jobs = build_labeling_jobs(
        rollouts,
        classification_prompt=classification_prompt,
        completed_keys=completed_keys,
        include_incomplete=include_incomplete,
        limit_rollouts=limit_rollouts,
        strict_json_instruction=strict_json_instruction,
    )
    written = 0
    batch_size = max(1, batch_size)
    concurrency = max(1, concurrency)
    progress = tqdm(total=len(jobs), desc="[labeling] rollouts", unit="rollout")
    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start : batch_start + batch_size]
        labeled_rollouts, failures = label_jobs_concurrently(
            batch,
            client=client,
            system_prompt=system_prompt,
            max_workers=min(concurrency, len(batch)),
        )
        for labeled in labeled_rollouts:
            append_labeled_rollout_jsonl(labeled, destination)
            written += 1
        for failure in failures:
            tqdm.write(
                "[labeling] failed "
                f"{failure.job.rollout.dataset_name}/{failure.job.rollout.task_id} "
                f"sample={failure.job.rollout.sample_id}: {failure.error}"
            )
        progress.update(len(batch))
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
    progress.close()
    return written


def summarize_rollout_file_for_labeling(
    *,
    rollout_file: str | Path,
    output_path: str | Path | None = None,
    include_incomplete: bool = False,
    resume: bool = True,
    limit_rollouts: int | None = None,
) -> RolloutLabelingSummary:
    rollouts = load_rollouts_jsonl(rollout_file)
    selected = select_rollouts_for_labeling(
        rollouts,
        include_incomplete=include_incomplete,
        limit_rollouts=limit_rollouts,
    )
    completed_keys = (
        read_completed_label_keys(output_path)
        if resume and output_path is not None
        else set()
    )
    pending = [
        rollout
        for rollout in selected
        if label_key(rollout) not in completed_keys
        and split_reasoning_steps(rollout.reasoning)
    ]
    return RolloutLabelingSummary(
        total_rollouts=len(rollouts),
        complete_rollouts=sum(1 for rollout in rollouts if rollout.complete),
        incomplete_rollouts=sum(1 for rollout in rollouts if not rollout.complete),
        rollouts_with_reasoning=sum(
            1 for rollout in rollouts if split_reasoning_steps(rollout.reasoning)
        ),
        selected_rollouts=len(selected),
        already_labeled=sum(1 for rollout in selected if label_key(rollout) in completed_keys),
        pending_jobs=len(pending),
    )


def select_rollouts_for_labeling(
    rollouts: list[CodeRollout],
    *,
    include_incomplete: bool = False,
    limit_rollouts: int | None = None,
) -> list[CodeRollout]:
    selected = rollouts if include_incomplete else [rollout for rollout in rollouts if rollout.complete]
    if limit_rollouts is not None:
        selected = selected[:limit_rollouts]
    return selected


def build_labeling_jobs(
    rollouts: list[CodeRollout],
    *,
    classification_prompt: str,
    completed_keys: set[tuple[str, str, int]] | None = None,
    include_incomplete: bool = False,
    limit_rollouts: int | None = None,
    strict_json_instruction: bool = True,
) -> list[LabelingJob]:
    completed = completed_keys or set()
    jobs: list[LabelingJob] = []
    for rollout in select_rollouts_for_labeling(
        rollouts,
        include_incomplete=include_incomplete,
        limit_rollouts=limit_rollouts,
    ):
        if label_key(rollout) in completed:
            continue
        sentences = split_reasoning_steps(rollout.reasoning)
        if not sentences:
            continue
        label_prompt = build_labeling_prompt(
            classification_prompt,
            problem=extract_problem_text(rollout),
            sentences=sentences,
            strict_json_instruction=strict_json_instruction,
        )
        jobs.append(
            LabelingJob(
                rollout=rollout,
                sentences=sentences,
                label_prompt=label_prompt,
            )
        )
    return jobs


def label_job(
    job: LabelingJob,
    *,
    client: LLMClient,
    system_prompt: str | None = None,
) -> LabeledRollout:
    rollout = job.rollout
    raw_response = client.generate(job.label_prompt, system_prompt=system_prompt)
    labels_payload = extract_json_object(raw_response)
    normalized_labels, warnings = normalize_label_payload(
        labels_payload,
        len(job.sentences),
    )
    return LabeledRollout(
        model_id=rollout.model_id,
        dataset_name=rollout.dataset_name,
        task_id=rollout.task_id,
        sample_id=rollout.sample_id,
        complete=rollout.complete,
        is_correct=rollout.is_correct,
        prompt=rollout.prompt,
        reasoning=rollout.reasoning,
        answer=rollout.answer,
        sentences=[
            {"index": str(index), "text": sentence}
            for index, sentence in enumerate(job.sentences, start=1)
        ],
        labels=normalized_labels,
        label_provider=client.config.provider,
        label_model=client.config.model,
        label_prompt=job.label_prompt,
        raw_label_response=raw_response,
        validation_warnings=warnings,
        labeled_at=datetime.now(UTC).isoformat(),
    )


def label_jobs_concurrently(
    jobs: list[LabelingJob],
    *,
    client: LLMClient,
    system_prompt: str | None = None,
    max_workers: int = 5,
    on_success: Callable[[LabeledRollout], None] | None = None,
    on_failure: Callable[[LabelingFailure], None] | None = None,
) -> tuple[list[LabeledRollout], list[LabelingFailure]]:
    if not jobs:
        return [], []
    successes: list[LabeledRollout] = []
    failures: list[LabelingFailure] = []
    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {
            executor.submit(label_job, job, client=client, system_prompt=system_prompt): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                labeled = future.result()
            except Exception as exc:  # noqa: BLE001 - keep batch progress moving.
                failure = LabelingFailure(job=job, error=str(exc))
                failures.append(failure)
                if on_failure is not None:
                    on_failure(failure)
                continue
            successes.append(labeled)
            if on_success is not None:
                on_success(labeled)
    return successes, failures


def build_labeling_prompt(
    classification_prompt: str,
    *,
    problem: str,
    sentences: list[str],
    strict_json_instruction: bool = True,
) -> str:
    numbered_sentences = "\n".join(
        f"{index}. {sentence.strip()}" for index, sentence in enumerate(sentences, start=1)
    )
    prompt = classification_prompt.replace("<PROBLEM>", problem.strip())
    prompt = prompt.replace("<SENTENCES>", numbered_sentences)
    if strict_json_instruction:
        prompt = f"{prompt}{STRICT_JSON_SUFFIX}"
    return prompt


def load_classification_prompt(prompt_path: str | Path | None = None) -> str:
    path = Path(prompt_path) if prompt_path is not None else PROJECT_ROOT / "prompt.py"
    spec = importlib.util.spec_from_file_location("thought_anchor_label_prompt", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = getattr(module, "CLASSIFICATION_PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path} must define a non-empty CLASSIFICATION_PROMPT string")
    return prompt


def extract_problem_text(rollout: CodeRollout) -> str:
    if not rollout.prompt:
        return ""
    prompt = rollout.prompt
    task_match = re.search(
        r"\bTask:\s*(.*?)(?:\n\nStarter code:|\n\nTests / context:|\n\nRespond using this format:|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    if task_match:
        return task_match.group(1).strip()
    return prompt.split("Respond using this format:", maxsplit=1)[0].strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("LLM response did not contain a JSON object")


def normalize_label_payload(
    payload: dict[str, Any],
    sentence_count: int,
) -> tuple[dict[str, SentenceLabel], list[str]]:
    warnings: list[str] = []
    labels: dict[str, SentenceLabel] = {}
    for index in range(1, sentence_count + 1):
        key = str(index)
        raw_entry = _get_label_entry(payload, key, index)
        if not isinstance(raw_entry, dict):
            warnings.append(f"Missing or invalid label for sentence {key}; using unknown.")
            labels[key] = SentenceLabel(function_tags=["unknown"], depends_on=[])
            continue
        labels[key] = SentenceLabel(
            function_tags=_normalize_function_tags(raw_entry, key, warnings),
            depends_on=_normalize_dependencies(raw_entry, key, index, sentence_count, warnings),
        )
    return labels, warnings


def _get_label_entry(payload: dict[str, Any], key: str, index: int) -> Any:
    if key in payload:
        return payload[key]
    if index in payload:
        return payload[index]
    for container_key in ("labels", "sentences", "annotations"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            if key in nested:
                return nested[key]
            if index in nested:
                return nested[index]
    return None


def _normalize_function_tags(
    raw_entry: dict[str, Any],
    key: str,
    warnings: list[str],
) -> list[str]:
    raw_tags = raw_entry.get("function_tags") or raw_entry.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list):
        warnings.append(f"Sentence {key} function_tags was not a list; using unknown.")
        return ["unknown"]
    tags = []
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag:
            continue
        if tag not in KNOWN_FUNCTION_TAGS:
            warnings.append(f"Sentence {key} had unknown tag '{tag}'; replacing with unknown.")
            tag = "unknown"
        if tag not in tags:
            tags.append(tag)
    return tags or ["unknown"]


def _normalize_dependencies(
    raw_entry: dict[str, Any],
    key: str,
    index: int,
    sentence_count: int,
    warnings: list[str],
) -> list[str]:
    raw_dependencies = raw_entry.get("depends_on") or []
    if isinstance(raw_dependencies, (str, int)):
        raw_dependencies = [raw_dependencies]
    if not isinstance(raw_dependencies, list):
        warnings.append(f"Sentence {key} depends_on was not a list; using [].")
        return []
    dependencies = []
    for raw_dependency in raw_dependencies:
        dependency_text = str(raw_dependency).strip()
        if not dependency_text:
            continue
        if not dependency_text.isdigit():
            warnings.append(f"Sentence {key} had non-numeric dependency '{dependency_text}'.")
            continue
        dependency_index = int(dependency_text)
        if dependency_index < 1 or dependency_index > sentence_count:
            warnings.append(
                f"Sentence {key} dependency {dependency_text} was outside sentence range."
            )
            continue
        if dependency_index >= index:
            warnings.append(
                f"Sentence {key} dependency {dependency_text} was not an earlier sentence."
            )
            continue
        normalized = str(dependency_index)
        if normalized not in dependencies:
            dependencies.append(normalized)
    return dependencies


def read_completed_label_keys(output_path: str | Path) -> set[tuple[str, str, int]]:
    path = Path(output_path)
    keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = payload.get("task_id")
            dataset_name = payload.get("dataset_name")
            sample_id = payload.get("sample_id")
            if task_id is None or dataset_name is None or sample_id is None:
                continue
            if not payload_has_concrete_labels(payload):
                continue
            keys.add((str(dataset_name), str(task_id), int(sample_id)))
    return keys


def payload_has_concrete_labels(payload: dict[str, Any]) -> bool:
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        return False
    for label in labels.values():
        if not isinstance(label, dict):
            continue
        tags = label.get("function_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            continue
        if any(str(tag).strip() and str(tag).strip() != "unknown" for tag in tags):
            return True
    return False


def label_key(rollout: CodeRollout) -> tuple[str, str, int]:
    return (rollout.dataset_name, rollout.task_id, rollout.sample_id)


def append_labeled_rollout_jsonl(
    labeled_rollout: LabeledRollout,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(labeled_rollout)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return destination


def write_labeled_rollouts_jsonl(
    labeled_rollouts: Iterable[LabeledRollout],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("", encoding="utf-8")
    for labeled_rollout in labeled_rollouts:
        append_labeled_rollout_jsonl(labeled_rollout, destination)
    return destination

