"""Executable correctness checks for generated benchmark solutions."""

from __future__ import annotations

import multiprocessing as mp
import re
import textwrap
from typing import Any


DEFAULT_EVAL_TIMEOUT_SECONDS = 5.0


def evaluate_generated_code(
    answer: str | None,
    row: dict[str, Any],
    dataset_name: str,
    timeout_seconds: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> bool:
    code = _extract_python_code(answer or "")
    if not code:
        return False

    queue: mp.Queue[bool] = mp.Queue(maxsize=1)
    process = mp.Process(target=_run_eval, args=(queue, code, row, dataset_name))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    if process.exitcode != 0 or queue.empty():
        return False
    return bool(queue.get())


def _run_eval(
    queue: mp.Queue[bool], code: str, row: dict[str, Any], dataset_name: str
) -> None:
    try:
        namespace: dict[str, Any] = {}
        if dataset_name == "mbpp":
            setup_code = row.get("test_setup_code") or ""
            if setup_code.strip():
                exec(setup_code, namespace)
            exec(code, namespace)
            tests = list(row.get("test_list") or []) + list(
                row.get("challenge_test_list") or []
            )
            for test in tests:
                exec(test, namespace)
        elif dataset_name == "openai_humaneval":
            prompt = str(row.get("prompt") or "")
            exec(prompt + "\n" + code, namespace)
            exec(str(row["test"]), namespace)
            namespace["check"](namespace[str(row["entry_point"])])
        else:
            raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")
    except Exception:
        queue.put(False)
    else:
        queue.put(True)


def _extract_python_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    stripped = re.split(
        r"\n(?=assert\s|Explanation\b|This code\b|The function\b)",
        stripped,
        maxsplit=1,
    )[0].strip()
    return textwrap.dedent(stripped).strip()
