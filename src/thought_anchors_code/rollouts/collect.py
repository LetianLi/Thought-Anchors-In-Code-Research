"""Rollout collection utilities for code reasoning tasks."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
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
from thought_anchors_code.rollouts.evaluate import evaluate_generated_code
from thought_anchors_code.rollouts.prompting import build_code_reasoning_prompt


REASONING_BLOCK_RE = re.compile(
    r"<reasoning>?\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE
)
CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
FENCED_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_REASONING_RE = re.compile(r"<reasoning>?\s*", re.IGNORECASE)
OPEN_CODE_RE = re.compile(r"<code>\s*", re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>\s*", re.IGNORECASE)


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
    batch_size: int = 1,
    token_progress: bool = True,
    seed: int | None = 0,
    output_path: str | Path | None = None,
    resume: bool = True,
    evaluate: bool = True,
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
    completed_keys: set[tuple[str, str, str, int]] = set()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if resume and destination.exists():
            completed_keys = read_completed_rollout_keys(destination)
            tqdm.write(
                f"[rollouts] Resuming {destination}; found {len(completed_keys)} existing rollouts"
            )
        else:
            destination.write_text("", encoding="utf-8")

    pending = []
    for index, row in enumerate(row_list):
        task_id = str(row.get("task_id") or row.get("id") or index)
        sample_id = 0
        rollout_key = (model_name_or_path, dataset_name, task_id, sample_id)
        if rollout_key in completed_keys:
            tqdm.write(f"[rollouts] Skipping existing task={task_id}")
            continue
        prompt = build_code_reasoning_prompt(
            task_prompt=_extract_task_prompt(row, dataset_name),
            starter_code=row.get("prompt")
            if dataset_name == "openai_humaneval"
            else None,
            test_context=_extract_test_context(row, dataset_name),
        )
        model_prompt = _format_model_prompt(tokenizer, prompt)
        pending.append((index, row, task_id, sample_id, prompt, model_prompt))

    if not pending:
        return rollouts

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    tokenizer.padding_side = "left"

    progress = tqdm(total=len(pending), desc=f"[rollouts] {dataset_name}", unit="example")
    for batch_start in range(0, len(pending), batch_size):
        if seed is not None and seed >= 0:
            _set_generation_seed(seed)
        batch = pending[batch_start : batch_start + batch_size]
        model_prompts = [item[5] for item in batch]
        inputs = tokenizer(model_prompts, return_tensors="pt", padding=True)
        padded_prompt_tokens = int(inputs["input_ids"].shape[1])
        prompt_token_counts = [
            int(mask.sum().item()) for mask in inputs.get("attention_mask", [])
        ]
        task_ids = ",".join(item[2] for item in batch)
        tqdm.write(
            f"[rollouts] batch={batch_start // batch_size + 1} size={len(batch)} tasks={task_ids} prompt_tokens={prompt_token_counts} max_new_tokens={max_new_tokens}"
        )
        token_bar = None
        stopping_criteria = None
        if token_progress:
            token_bar = tqdm(
                total=max_new_tokens,
                desc=f"[tokens] batch {batch_start // batch_size + 1}",
                unit="tok",
                leave=False,
                position=1,
            )
            stopping_criteria = StoppingCriteriaList(
                [
                    TokenTqdmCallback(
                        prompt_tokens=padded_prompt_tokens,
                        progress_bar=token_bar,
                    )
                ]
            )
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
        eos_token_ids = _get_eos_token_ids(tokenizer)
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
                    eos_token_id=eos_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria,
                )
            finally:
                if token_bar is not None:
                    token_bar.close()
        elapsed = time.perf_counter() - start_time
        total_generated_tokens = 0
        for batch_index, (_, row, task_id, sample_id, prompt, _) in enumerate(batch):
            generated_ids = outputs[batch_index][padded_prompt_tokens:]
            generated_ids = _trim_after_eos(generated_ids, eos_token_ids)
            generated_tokens = int(generated_ids.shape[0])
            total_generated_tokens += generated_tokens
            raw_generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed = parse_reasoning_and_code(
                generated, task_id=task_id, sample_id=sample_id
            )
            reasoning, answer, complete = parsed
            is_correct = (
                evaluate_generated_code(answer, row, dataset_name) if evaluate else None
            )
            tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0
            tqdm.write(
                f"[rollouts] completed task={task_id} generated_tokens={generated_tokens} elapsed_s={elapsed:.1f} tokens_per_second={tokens_per_second:.2f} reasoning_chars={len(reasoning)} code_chars={len(answer or '')} complete={complete} is_correct={is_correct}"
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
                    is_correct=is_correct,
                )
            )
            if destination is not None:
                append_rollout_jsonl(rollouts[-1], destination)
        if len(batch) > 1 and elapsed > 0:
            tqdm.write(
                f"[rollouts] batch throughput generated_tokens={total_generated_tokens} elapsed_s={elapsed:.1f} aggregate_tokens_per_second={total_generated_tokens / elapsed:.2f}"
            )
        progress.update(len(batch))
    progress.close()
    return rollouts


def read_completed_rollout_keys(output_path: str | Path) -> set[tuple[str, str, str, int]]:
    path = Path(output_path)
    keys: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not payload.get("complete") or payload.get("is_correct") is False:
                continue
            keys.add(
                (
                    str(payload.get("model_id")),
                    str(payload.get("dataset_name")),
                    str(payload.get("task_id")),
                    int(payload.get("sample_id", 0)),
                )
            )
    return keys


def _format_model_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [
        {
            "role": "system",
            "content": "You are a precise Python coding assistant. Follow the requested output format exactly and do not repeat placeholder text.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _get_eos_token_ids(tokenizer) -> list[int] | int | None:
    token_ids = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)
    if not token_ids:
        return None
    return token_ids[0] if len(token_ids) == 1 else token_ids


def _trim_after_eos(token_ids: torch.Tensor, eos_token_ids: list[int] | int | None) -> torch.Tensor:
    if eos_token_ids is None:
        return token_ids
    eos_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    for index, token_id in enumerate(token_ids.tolist()):
        if int(token_id) in eos_ids:
            return token_ids[: index + 1]
    return token_ids


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
    think_match = THINK_BLOCK_RE.search(generated_text)

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

    if code_match is not None:
        reasoning = generated_text[: code_match.start()].strip()
        reasoning = OPEN_REASONING_RE.sub("", reasoning, count=1).strip()
        reasoning = OPEN_THINK_RE.sub("", reasoning, count=1).strip()
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: missing closed <reasoning> block, but found <code> block.",
            stacklevel=2,
        )
        return reasoning, code_match.group(1).strip(), False

    if think_match is not None:
        reasoning = think_match.group(1).strip()
        post_think = generated_text[think_match.end() :].strip()
        fenced_code_match = FENCED_CODE_RE.search(post_think)
        if fenced_code_match is not None:
            return reasoning, fenced_code_match.group(1).strip(), True
        return reasoning, post_think, True

    open_reasoning_match = OPEN_REASONING_RE.search(generated_text)
    open_think_match = OPEN_THINK_RE.search(generated_text)
    if open_reasoning_match is None and open_think_match is None:
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: missing <reasoning> tag, using raw generation as reasoning.",
            stacklevel=2,
        )
        return stripped, "", False

    if open_reasoning_match is None:
        reasoning = generated_text[open_think_match.end() :].strip()
        warnings.warn(
            f"Storing incomplete rollout for task_id={task_id}, sample_id={sample_id}: <think> block was not closed.",
            stacklevel=2,
        )
        return reasoning, "", False

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
        handle.write(json.dumps(asdict(rollout), ensure_ascii=False) + "\n")
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


def _set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
