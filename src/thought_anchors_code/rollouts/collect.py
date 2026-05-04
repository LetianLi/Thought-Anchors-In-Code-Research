"""Rollout collection utilities for code reasoning tasks."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import time
from typing import Iterable
import warnings

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from thought_anchors_code.analysis.whitebox_attention.types import CodeRollout
from thought_anchors_code.config import ROLLOUT_DIR
from thought_anchors_code.engine import get_local_model, get_model_input_device
from thought_anchors_code.rollouts.prompting import build_code_reasoning_prompt


REASONING_BLOCK_RE = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
OPEN_REASONING_RE = re.compile(r"<reasoning>\s*", re.IGNORECASE)
OPEN_CODE_RE = re.compile(r"<code>\s*", re.IGNORECASE)


def collect_rollouts(
    rows: Iterable[dict],
    model_name_or_path: str,
    dataset_name: str,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    output_path: str | Path | None = None,
) -> list[CodeRollout]:
    row_list = list(rows)
    tqdm.write(f"[rollouts] Loading model: {model_name_or_path}")
    model, tokenizer = get_local_model(
        model_name_or_path=model_name_or_path,
        float32=False,
        device_map="auto",
    )
    input_device = get_model_input_device(model)
    tqdm.write("[rollouts] Model and tokenizer loaded")
    rollouts: list[CodeRollout] = []
    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")

    progress = tqdm(row_list, desc=f"[rollouts] {dataset_name}", unit="example")
    for index, row in enumerate(progress):
        task_id = str(row.get("task_id") or row.get("id") or index)
        progress.set_postfix(task_id=task_id)
        prompt = build_code_reasoning_prompt(
            task_prompt=_extract_task_prompt(row, dataset_name),
            starter_code=row.get("prompt")
            if dataset_name == "openai_humaneval"
            else None,
            test_context=_extract_test_context(row, dataset_name),
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[1])
        tqdm.write(
            f"[rollouts] {index + 1}/{len(row_list)} task={task_id} prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens}"
        )
        sample_id = 0
        token_bar = tqdm(
            total=max_new_tokens,
            desc=f"[tokens] {task_id}.{sample_id}",
            unit="tok",
            leave=False,
            position=1,
        )
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
        start_time = time.perf_counter()
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [
                            TokenTqdmCallback(
                                prompt_tokens=prompt_tokens, progress_bar=token_bar
                            )
                        ]
                    ),
                )
            finally:
                token_bar.close()
        elapsed = time.perf_counter() - start_time
        generated_tokens = int(outputs[0].shape[0] - inputs["input_ids"].shape[1])
        raw_generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        parsed = parse_reasoning_and_code(
            generated, task_id=task_id, sample_id=sample_id
        )
        reasoning, answer, complete = parsed
        tqdm.write(
            f"[rollouts] completed task={task_id} generated_tokens={generated_tokens} elapsed_s={elapsed:.1f} reasoning_chars={len(reasoning)} code_chars={len(answer or '')} complete={complete}"
        )
        rollouts.append(
            CodeRollout(
                model_id=model_name_or_path,
                dataset_name=dataset_name,
                task_id=task_id,
                sample_id=sample_id,
                complete=complete,
                prompt=prompt,
                raw=raw_generated,
                reasoning=reasoning,
                answer=answer,
                is_correct=None,
            )
        )
        if destination is not None:
            append_rollout_jsonl(rollouts[-1], destination)
    return rollouts


def parse_reasoning_and_code(
    generated_text: str,
    *,
    task_id: str | None = None,
    sample_id: int | None = None,
) -> tuple[str, str, bool] | None:
    stripped = generated_text.strip()
    if not stripped:
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: generation was empty.",
            stacklevel=2,
        )
        return "", "", False

    reasoning_match = REASONING_BLOCK_RE.search(generated_text)
    code_match = CODE_BLOCK_RE.search(generated_text)

    if reasoning_match is not None:
        reasoning = reasoning_match.group(1).strip()
        if code_match is not None:
            return reasoning, code_match.group(1).strip(), True

        open_code_match = OPEN_CODE_RE.search(generated_text, reasoning_match.end())
        if open_code_match is not None:
            answer = generated_text[open_code_match.end() :].strip()
            warnings.warn(
                f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: <code> block was not closed.",
                stacklevel=2,
            )
            return reasoning, answer, False

        return reasoning, "", True

    open_reasoning_match = OPEN_REASONING_RE.search(generated_text)
    if open_reasoning_match is None:
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: missing <reasoning> tag, using raw generation as reasoning.",
            stacklevel=2,
        )
        return stripped, "", False

    open_code_match = OPEN_CODE_RE.search(generated_text, open_reasoning_match.end())
    if open_code_match is not None:
        reasoning = generated_text[
            open_reasoning_match.end() : open_code_match.start()
        ].strip()
        answer = generated_text[open_code_match.end() :].strip()
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: <reasoning> and <code> blocks were not closed.",
            stacklevel=2,
        )
        return reasoning, answer, False

    reasoning = generated_text[open_reasoning_match.end() :].strip()
    warnings.warn(
        f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: <reasoning> block was not closed.",
        stacklevel=2,
    )
    return reasoning, "", False


def write_rollouts_jsonl(
    rollouts: Iterable[CodeRollout], output_path: str | Path
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("", encoding="utf-8")
    for rollout in rollouts:
        append_rollout_jsonl(rollout, destination)
    return destination


def append_rollout_jsonl(rollout: CodeRollout, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(rollout)) + "\n")
    return destination


def default_rollout_path(dataset_name: str) -> Path:
    return ROLLOUT_DIR / f"{dataset_name}_rollouts.jsonl"


class TokenTqdmCallback(StoppingCriteria):
    def __init__(self, prompt_tokens: int, progress_bar: tqdm) -> None:
        self.prompt_tokens = prompt_tokens
        self.progress_bar = progress_bar
        self.previous_tokens = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        generated_tokens = max(0, int(input_ids.shape[1]) - self.prompt_tokens)
        delta = generated_tokens - self.previous_tokens
        if delta > 0:
            self.progress_bar.update(delta)
            self.previous_tokens = generated_tokens
        return False


def _extract_task_prompt(row: dict, dataset_name: str) -> str:
    if dataset_name == "mbpp":
        return str(row["text"])
    if dataset_name == "openai_humaneval":
        return str(row["prompt"])
    raise ValueError(f"Unsupported dataset for rollout collection: {dataset_name}")


def _extract_test_context(row: dict, dataset_name: str) -> str | None:
    if dataset_name == "mbpp":
        tests = row.get("test_list") or []
        return "\n".join(tests)
    if dataset_name == "openai_humaneval":
        return row.get("test")
    return None
