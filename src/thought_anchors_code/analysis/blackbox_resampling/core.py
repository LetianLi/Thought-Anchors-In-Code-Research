"""Black-box sentence resampling for causal thought-anchor checks."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import torch
from tqdm import tqdm

from thought_anchors_code.analysis.blackbox_resampling.types import (
    ResampleOutcome,
    ResamplingResult,
    SentenceIntervention,
)
from thought_anchors_code.analysis.whitebox_attention.trace_utils import (
    split_reasoning_steps,
    truncate_rollouts_to_sentence_percentile,
)
from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.engine import get_local_model, get_model_input_device
from thought_anchors_code.rollouts.collect import (
    _format_model_prompt,
    _get_eos_token_ids,
    _set_generation_seed,
    _trim_after_eos,
    parse_reasoning_and_code,
)
from thought_anchors_code.rollouts.evaluate import evaluate_generated_code


def load_attention_summaries(path: str | Path) -> dict[tuple[str, int], dict]:
    summaries: dict[tuple[str, int], dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            summaries[(str(payload["task_id"]), int(payload.get("sample_id", 0)))] = payload
    return summaries


def enumerate_sentence_interventions(
    rollouts: Sequence[CodeRollout],
    attention_summaries: Mapping[tuple[str, int], dict],
) -> list[SentenceIntervention]:
    """Create one resampling intervention per sentence in each rollout."""
    interventions: list[SentenceIntervention] = []
    for rollout in rollouts:
        if not rollout.complete:
            warnings.warn(
                f"Skipping task_id={rollout.task_id}, sample_id={rollout.sample_id}: rollout is incomplete.",
                stacklevel=2,
            )
            continue
        sentences = split_reasoning_steps(rollout.reasoning)
        if not sentences:
            continue
        summary = attention_summaries.get((rollout.task_id, rollout.sample_id))
        sentence_scores = _coerce_score_list(
            summary.get("sentence_scores") if summary is not None else []
        )
        code_scores = _coerce_score_list(
            summary.get("code_sentence_scores") if summary is not None else []
        )
        for sentence_index in range(len(sentences)):
            interventions.append(
                _intervention_from_index(
                    rollout,
                    sentences,
                    sentence_scores,
                    code_scores,
                    sentence_index,
                    selection="sentence",
                )
            )
    return dedupe_interventions(interventions)


def dedupe_interventions(
    interventions: Iterable[SentenceIntervention],
) -> list[SentenceIntervention]:
    seen: set[tuple[str, str, int, int, str]] = set()
    deduped: list[SentenceIntervention] = []
    for intervention in interventions:
        key = (
            intervention.dataset_name,
            intervention.task_id,
            intervention.sample_id,
            intervention.sentence_index,
            intervention.selection,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(intervention)
    return deduped


def build_continuation_prompt(
    rollout: CodeRollout,
    sentence_index: int,
    *,
    tokenizer=None,
    omit_sentence: bool = True,
) -> tuple[str, int, int]:
    """Build model input for continuing after an omitted reasoning sentence."""
    sentences = split_reasoning_steps(rollout.reasoning)
    if sentence_index < 0 or sentence_index >= len(sentences):
        raise IndexError(
            f"sentence_index={sentence_index} is outside 0..{len(sentences) - 1}"
        )
    kept_prefix = "".join(sentences[:sentence_index])
    suffix_count = len(sentences) - sentence_index - (1 if omit_sentence else 0)
    if tokenizer is None:
        model_prompt = rollout.prompt
    else:
        model_prompt = _format_model_prompt(tokenizer, rollout.prompt)
    prefix = f"{model_prompt}<reasoning>\n{kept_prefix}"
    if kept_prefix and not kept_prefix.endswith((" ", "\n")):
        prefix += " "
    return prefix, sentence_index, suffix_count


def run_blackbox_resampling_to_jsonl(
    rollouts: Sequence[CodeRollout],
    rows_by_task_id: Mapping[str, dict],
    attention_summary_path: str | Path,
    output_path: str | Path,
    *,
    model_name_or_path: str,
    num_resamples: int = 3,
    limit_rollouts: int | None = None,
    truncate_to_percentile: float = 75.0,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = 0,
    batch_size: int = 4,
    resume: bool = True,
    evaluate: bool = True,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_resampling_keys(destination) if resume else set()
    if not resume:
        destination.write_text("", encoding="utf-8")

    selected_rollouts = list(rollouts[:limit_rollouts] if limit_rollouts else rollouts)
    if truncate_to_percentile is not None:
        selected_rollouts, max_sentences = truncate_rollouts_to_sentence_percentile(
            selected_rollouts, percentile=truncate_to_percentile
        )
        if max_sentences is not None:
            tqdm.write(
                f"[blackbox] Truncated reasoning traces to p{int(truncate_to_percentile)} sentence count: {max_sentences}"
            )
    summaries = load_attention_summaries(attention_summary_path)
    interventions = enumerate_sentence_interventions(selected_rollouts, summaries)
    pending = [
        intervention
        for intervention in interventions
        if intervention_key(intervention) not in completed
    ]
    if not pending:
        return 0

    tqdm.write(f"[blackbox] Loading model: {model_name_or_path}")
    model, tokenizer = get_local_model(
        model_name_or_path=model_name_or_path,
        float32=False,
        device_map="auto",
    )
    input_device = get_model_input_device(model)
    eos_token_ids = _get_eos_token_ids(tokenizer)
    rollout_map = {
        (rollout.dataset_name, rollout.task_id, rollout.sample_id): rollout
        for rollout in selected_rollouts
    }
    jobs = build_resampling_jobs(
        pending,
        rollout_map=rollout_map,
        rows_by_task_id=rows_by_task_id,
        tokenizer=tokenizer,
        num_resamples=num_resamples,
    )
    outputs_by_key: dict[tuple[str, str, int, int, str], list[ResampleOutcome]] = {
        intervention_key(intervention): [] for intervention in pending
    }
    written = 0
    with destination.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(
            tqdm(
                _batched(jobs, batch_size),
                total=math.ceil(len(jobs) / batch_size),
                desc="[blackbox] batches",
                unit="batch",
            )
        ):
            if seed is not None and seed >= 0:
                _set_generation_seed(seed + batch_index)
            outcomes = generate_resample_batch(
                model,
                tokenizer,
                batch,
                input_device=input_device,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                evaluate=evaluate,
            )
            for job, outcome in zip(batch, outcomes, strict=True):
                key = intervention_key(job.intervention)
                outputs_by_key[key].append(outcome)
                if len(outputs_by_key[key]) < num_resamples:
                    continue
                intervention = job.intervention
                rollout = rollout_map[
                    (
                        intervention.dataset_name,
                        intervention.task_id,
                        intervention.sample_id,
                    )
                ]
                result = ResamplingResult(
                    model_id=model_name_or_path,
                    dataset_name=intervention.dataset_name,
                    task_id=intervention.task_id,
                    sample_id=intervention.sample_id,
                    sentence_index=intervention.sentence_index,
                    selection=intervention.selection,
                    sentence_text=intervention.sentence_text,
                    sentence_score=intervention.sentence_score,
                    code_sentence_score=intervention.code_sentence_score,
                    original_answer=rollout.answer,
                    original_is_correct=rollout.is_correct,
                    prefix_sentence_count=job.prefix_sentence_count,
                    suffix_sentence_count=job.suffix_sentence_count,
                    resamples=sorted(
                        outputs_by_key[key], key=lambda item: item.resample_id
                    ),
                )
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
    return written


class ResamplingJob:
    def __init__(
        self,
        *,
        intervention: SentenceIntervention,
        prompt: str,
        row: dict,
        prefix_sentence_count: int,
        suffix_sentence_count: int,
        resample_id: int,
    ) -> None:
        self.intervention = intervention
        self.prompt = prompt
        self.row = row
        self.prefix_sentence_count = prefix_sentence_count
        self.suffix_sentence_count = suffix_sentence_count
        self.resample_id = resample_id


def build_resampling_jobs(
    interventions: Sequence[SentenceIntervention],
    *,
    rollout_map: Mapping[tuple[str, str, int], CodeRollout],
    rows_by_task_id: Mapping[str, dict],
    tokenizer,
    num_resamples: int,
) -> list[ResamplingJob]:
    jobs: list[ResamplingJob] = []
    for intervention in interventions:
        rollout = rollout_map[
            (intervention.dataset_name, intervention.task_id, intervention.sample_id)
        ]
        row = rows_by_task_id.get(intervention.task_id)
        if row is None:
            warnings.warn(
                f"Skipping task_id={intervention.task_id}: dataset row not found.",
                stacklevel=2,
            )
            continue
        prompt, prefix_count, suffix_count = build_continuation_prompt(
            rollout,
            intervention.sentence_index,
            tokenizer=tokenizer,
            omit_sentence=True,
        )
        for resample_id in range(num_resamples):
            jobs.append(
                ResamplingJob(
                    intervention=intervention,
                    prompt=prompt,
                    row=row,
                    prefix_sentence_count=prefix_count,
                    suffix_sentence_count=suffix_count,
                    resample_id=resample_id,
                )
            )
    return jobs


def generate_resample_batch(
    model,
    tokenizer,
    jobs: Sequence[ResamplingJob],
    *,
    input_device,
    eos_token_ids,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    evaluate: bool,
) -> list[ResampleOutcome]:
    tokenizer.padding_side = "left"
    prompts = [job.prompt for job in jobs]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    padded_prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    outcomes = []
    for batch_index, job in enumerate(jobs):
        generated_ids = _trim_after_eos(
            outputs[batch_index][padded_prompt_tokens:], eos_token_ids
        )
        raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_reasoning_and_code(
            generated,
            task_id=job.intervention.task_id,
            sample_id=job.intervention.sample_id,
        )
        reasoning, answer, complete = parsed or ("", "", False)
        is_correct = (
            evaluate_generated_code(answer, job.row, job.intervention.dataset_name)
            if evaluate
            else None
        )
        outcomes.append(
            ResampleOutcome(
                resample_id=job.resample_id,
                raw=raw,
                reasoning=reasoning,
                answer=answer,
                complete=complete,
                is_correct=is_correct,
            )
        )
    return outcomes


def generate_resample_once(
    model,
    tokenizer,
    prompt: str,
    *,
    input_device,
    eos_token_ids,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    task_id: str,
    sample_id: int,
    row: dict,
    dataset_name: str,
    evaluate: bool,
    resample_id: int,
) -> ResampleOutcome:
    intervention = SentenceIntervention(
        dataset_name=dataset_name,
        task_id=task_id,
        sample_id=sample_id,
        sentence_index=0,
        sentence_text="",
        selection="sentence",
        sentence_score=None,
        code_sentence_score=None,
    )
    return generate_resample_batch(
        model,
        tokenizer,
        [
            ResamplingJob(
                intervention=intervention,
                prompt=prompt,
                row=row,
                prefix_sentence_count=0,
                suffix_sentence_count=0,
                resample_id=resample_id,
            )
        ],
        input_device=input_device,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        evaluate=evaluate,
    )[0]


def read_completed_resampling_keys(
    output_path: str | Path,
) -> set[tuple[str, str, int, int, str]]:
    path = Path(output_path)
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int, int, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            keys.add(
                (
                    str(payload["dataset_name"]),
                    str(payload["task_id"]),
                    int(payload.get("sample_id", 0)),
                    int(payload["sentence_index"]),
                    str(payload.get("selection", "sentence")),
                )
            )
    return keys


def intervention_key(
    intervention: SentenceIntervention,
) -> tuple[str, str, int, int, str]:
    return (
        intervention.dataset_name,
        intervention.task_id,
        intervention.sample_id,
        intervention.sentence_index,
        intervention.selection,
    )


def _intervention_from_index(
    rollout: CodeRollout,
    sentences: Sequence[str],
    sentence_scores: Sequence[float | None],
    code_scores: Sequence[float | None],
    sentence_index: int,
    *,
    selection: str,
) -> SentenceIntervention:
    return SentenceIntervention(
        dataset_name=rollout.dataset_name,
        task_id=rollout.task_id,
        sample_id=rollout.sample_id,
        sentence_index=sentence_index,
        sentence_text=sentences[sentence_index].strip(),
        selection=selection,
        sentence_score=_score_at(sentence_scores, sentence_index),
        code_sentence_score=_score_at(code_scores, sentence_index),
    )


def _coerce_score_list(values) -> list[float | None]:
    if not values:
        return []
    scores = []
    for value in values:
        if value is None:
            scores.append(None)
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            scores.append(None)
            continue
        scores.append(score if math.isfinite(score) else None)
    return scores


def _score_at(scores: Sequence[float | None], index: int) -> float | None:
    if index >= len(scores):
        return None
    return scores[index]


def _batched(items: Sequence[ResamplingJob], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
